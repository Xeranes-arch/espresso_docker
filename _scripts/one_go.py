#############################################################################################################################
#############################################################################################################################

# README
# Generates an Ellipsoid geometry raspberry particle filled as dense as can be without overlaps.

# You should only really need to change the AXES to what you want.
# If funky results, enable save_visualization. Written as series of frames for paraview.

# With the current configuration 99% of the calculation time is spent making sure the max nr of particles is approached slowly.
# The surface more so than the inside. Even so, *no guarantee at all* that it finds the best nr.
# If you don't care much about having the highest possible density, you can massively reduce the time this takes.

#############################################################################################################################
#############################################################################################################################
# region: Import

from functions import *
import os
from pathlib import Path

import time
import numba
import numpy as np
import matplotlib.pyplot as plt

import espressomd
espressomd.assert_features(
    [
        "ROTATION",
        "ROTATIONAL_INERTIA",
        "EXTERNAL_FORCES",
        "MASS",
        "VIRTUAL_SITES_RELATIVE",
        "LENNARD_JONES"
    ]
)
LINE = "\n___________________________"

# endregion
#############################################################################################################################
# region: User input

# Redefining print to only obtain barebones output via override = True when DEBUG = FALSE
# Also enables absolute cut off points when itteration nr reached
DEBUG = True  # toggle this


def print(*args, debug_override=False, **kwargs):
    if DEBUG or debug_override:
        __builtins__.print(*args, **kwargs)


# Writes vtk animation frames, doesn't hamper performance much at all
save_visualization = True
# save_visualization = False

##################

# Define the semi-axes of the ellipsoid. [a, b, b]
# Only Spheroids a != b = c

# for running from outside with args if you want several geometries cued up with cue_AXES.py
# a = float(sys.argv[1])
# b = float(sys.argv[2])
# n_particles = float(sys.argv[3])

# Manual aspect ratio
a = 2
b = 1

# n_particles = 512
n_particles = 64

# endregion

#############################################################################################################################
numba.set_num_threads(4)  # or the number of cores you want
# region: Numba test
run_test = False
# Numba achieves a 20x speed up the way I set it up and it only eats ~34% of my cpu vs 25% before.
print("Number of threads: ", numba.get_num_threads())
if run_test:
    @njit(parallel=True)
    def parallel_sum(n):
        total = 0.0
        for i in prange(n):  # Parallel loop
            total += i * 0.5
        return total

    n = 100_000_000  # Large enough to see parallel speedup

    # Single-threaded version (for comparison)

    def serial_sum(n):
        total = 0.0
        for i in range(n):
            total += i * 0.5
        return total

    # Time serial version
    start = time.time()
    serial_result = serial_sum(n)
    serial_time = time.time() - start

    # Time parallel version
    start = time.time()
    parallel_result = parallel_sum(n)
    parallel_time = time.time() - start

    # Verify results match
    assert np.isclose(serial_result, parallel_result), "Results don't match!"

    print(f"Sum result: {parallel_result:.2f}")
    print(f"Serial time: {serial_time:.4f} seconds")
    print(f"Parallel time: {parallel_time:.4f} seconds")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")

    if parallel_time < serial_time * 0.7:  # Expect at least 30% speedup
        print("✅ Parallelization is working!")
    else:
        print("❌ Parallelization may not be working properly")

    exit()

# endregion
#############################################################################################################################
# region: System parameters

# size of the simulation box, arbitrary, should be larger than the raspberry...
box_l = 90.0
kT = 0.001
time_step = 0.001
gamma = 10
force_cap = 1000

# Interaction parameter
eps_ss = 1  # LJ epsilon
sig_ss = 1  # LJ sigma

# endregion
#############################################################################################################################
# region: AXES
AXES = np.array(
    [b, b, a]
)
# find upper and lower bound volumes of ellipsoid
MAX_VOLUME_FRACTION = 0.74048  # Hcp packing 3D
ell_vol = ellipsoid_volume(AXES)
# lower bound is volume of hcp fraction per particle
ell_vol_lower_bound = n_particles * 4 / \
    3 * np.pi * (1/2)**3 / MAX_VOLUME_FRACTION
# upper bound is volume of 1 per particle
ell_vol_upper_bound = n_particles
# convert to scaling factors
scaling_factor = (ell_vol_lower_bound / ell_vol)**(1/3)
upper_bound_factor = (ell_vol_upper_bound / ell_vol)**(1/3)

AXES = np.array([scaling_factor * i for i in AXES])
# would now theoretically encompass all spheres, but we want the ell surface through the centers of outermost particles
AXES = AXES - 0.5
print("Lowerbound for Axes: \n", AXES)
og_AXES = list(AXES)
# endregion
#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = 3  # Skin parameter for the Verlet lists
system.periodicity = [True, True, True]

# the LJ potential (WCA potential) between surface beads
system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=1, sigma=1, cutoff=2.5, shift=-1, offset=0.077)
# system.non_bonded_inter[0, 0].wca.set_params(epsilon=eps_ss, sigma=sig_ss)

# Going to even lower temp regime 0.00001 might make sense, but this seems to function well
system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)

print(LINE)
print("# Creating raspberry", AXES, debug_override=True)

print("Particle_NR: ", n_particles)
# Generate list of positions
tolerance = 0
while True:
    arr_of_points = hex_ellipsoid_points(
        AXES[0]+tolerance, AXES[1]+tolerance, AXES[2]+tolerance)

    if (len(arr_of_points) > n_particles):
        break
    tolerance += 0.01

# Place particle
for i, pos in enumerate(arr_of_points):
    if i > n_particles:
        break
    system.part.add(pos=pos)

# endregion

################## Iterating ###################

# Empty folder for recording frames for Paraview
if save_visualization:
    os.makedirs("_data/vtk_frames/one_go", exist_ok=True)
    folder_path = Path("_data/vtk_frames/one_go")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
nth_step_recorded = 25  # record every n-th simulation step

# Large overlaps with WCA could explode the system. Baby steps.
system.force_cap = force_cap

print(LINE)
print("Relaxation of particles")
t0 = time.time()

k = 0  # iteration variable
check_nth = 100  # nr of iterations before checking progress
thermal_kick = 1
mid_temp = False
low_temp = False

high_score_smallest = 0
smallest_list = []  # list for plotting progress afterwards
highest_unsuccessful_factor = [0]
smallest_success_factor = [upper_bound_factor]
current_factor = 1

no_improvement_counter = 0  # goes up when no new high scores
size_up = []
size_down = []

plot = []
t = [0]

while True:

    system.integrator.run(1)

    # now put all particles back on the surface
    particle_positions = system.part.all().pos
    t1 = time.time()
    new_poss = shift_particle(particle_positions, AXES, if_outside=True)
    t.append((time.time() - t1 + t[-1])/(k+1))
    for part, pos in zip(system.part.all(), new_poss):
        part.pos = pos

    # Record frame
    if not k % nth_step_recorded:
        if save_visualization:
            writevtk(
                f"_data/vtk_frames/one_go/frame_{int(k/nth_step_recorded)}.vtk", system, half_box_shift=True)

    k += 1
    if not k % check_nth:
        # Calculate metrics
        mean, rmsd, smallest = mean_rmsd_smallest(
            system.part.all().id, system)

        print(LINE)
        print("smallest: ", smallest)
        print("mean: ", mean)
        print("rmsd: ", rmsd)
        smallest_list.append(smallest)

        if smallest > high_score_smallest:
            no_improvement_counter = 0
            high_score_smallest = smallest
        elif not thermal_kick:
            no_improvement_counter += 1

        # Successful configuration
        if smallest > 1:
            plot.append((current_factor, 1))
            high_score_smallest = 0
            no_improvement_counter = 0
            print(LINE)
            print("SUCCESS: ", current_factor)
            # Finish condition
            # 0.001
            if abs(smallest_success_factor[-1] - highest_unsuccessful_factor[-1]) < 0.1:
                plt.clf()
                for i, item in enumerate(plot):
                    y, c = item
                    if c == 1:
                        col = "orange"
                        marker = "+"
                    else:
                        col = "blue"
                        marker = "o"
                    plt.scatter(i, y, c=col, marker=marker)
                plt.savefig("factors.png")
                plt.clf()
                break
            # Try to find smaller bound
            else:
                smallest_success_factor.append(round(float(current_factor), 3))
                current_factor = (
                    current_factor + highest_unsuccessful_factor[-1]) / 2

                # Execute size down
                print("GOING DOWN to:", current_factor)
                AXES = np.array([i * current_factor for i in og_AXES])
                shuffle_pos(system, AXES)
                # for part in system.part.all():
                #     part.pos = [0,0,0]

                system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
                low_temp = False
                mid_temp = False

                thermal_kick = True
                print("THERMAL KICK ON")
                system.thermostat.set_langevin(
                    kT=kT*10, gamma=gamma, seed=42)
                high_score_smallest = 0
                continue

    # Not a successful size
    if no_improvement_counter == 10:
        no_improvement_counter = 0

        print(LINE)
        print("UNSUCCESSFUL: ", current_factor)
        highest_unsuccessful_factor.append(current_factor)
        plot.append((current_factor, 0))

        current_factor = (current_factor + smallest_success_factor[-1])/2
        AXES = np.array([i * current_factor for i in og_AXES])
        shuffle_pos(system, AXES)
        # for part in system.part.all():
        #     part.pos = [0, 0, 0]
        print("\n")
        print(LINE)
        print("Factor up: ", round(current_factor, 3))
        high_score_smallest = 0

        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
        low_temp = False
        mid_temp = False

        thermal_kick = True
        print("thermal kick ON")
        system.thermostat.set_langevin(kT=kT*10, gamma=gamma/10, seed=42)

        size_up.append(k)

    if high_score_smallest > 0.99 and not mid_temp:
        print("mid temp regime engaged")
        system.thermostat.set_langevin(kT=kT/10, gamma=gamma*10, seed=42)
        mid_temp = True
        thermal_kick = 0

    # if high_score_smallest > 0.999 and not low_temp:
    #     print("Low temp regime engaged")
    #     system.thermostat.set_langevin(kT=kT/100, gamma=gamma*100, seed=42)
    #     low_temp = True
    #     thermal_kick = 0

    if thermal_kick:
        thermal_kick += 1
        if thermal_kick == 1000:
            thermal_kick = 0
            system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
            print(LINE)
            print("thermal kick OFF")

filename = f"_data/coordinates/{n_particles}_ratio_{a}_{b}.txt"
filename = filename.replace(f"._", f".0_")
filename = filename.replace(f".]", f".0]")
np.savetxt(filename, system.part.select(lambda p: p.id > 0).pos)
print(LINE)
print(filename)
print("contains coordinates of raspberry")

if save_visualization:
    plt.plot(np.arange(len(smallest_list)), smallest_list)
    for val in size_up:
        plt.axvline(x=int(val/check_nth-1), color='red',
                    linestyle='--', linewidth=1.0)
    for val in size_down:
        plt.axvline(x=int(val/check_nth-1), color='blue',
                    linestyle='--', linewidth=1.0)
    plt.ylim(0.8, 1.02)
    plt.savefig(f"_data/relaxation{a}_{b}.png")

t = time.time() - t0
filename = "geom.json"
entry = {
    "type": "one_go",
    "pnr": len(system.part.all())-1,
    "factor": current_factor,
    "k": k,
    "rmsd": rmsd,
    "time": t,
    "kT": kT,
    "gamma": gamma,
    "force cap": force_cap,
    "comment": "epsilon = 20"
}
save_to_json(filename, entry)

print("Reached the end.")
print(LINE)
