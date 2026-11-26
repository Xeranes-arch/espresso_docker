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
t0 = time.time()

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
a = 1
b = 1

# n_particles = 512
n_particles = 64

# endregion
#############################################################################################################################
numba.set_num_threads(1)  # or the number of cores you want
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
# region: Functions


def kick():
    global thermal_kick
    thermal_kick = True
    print("THERMAL KICK ON")
    system.thermostat.set_langevin(
        kT=kT*1000, gamma=gamma, seed=42)


def plot_tried_factors():
    global factors_list
    plt.clf()
    for i, item in enumerate(factors_list):
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


# endregion
#############################################################################################################################
# region: System parameters


# size of the simulation box, arbitrary, should be larger than the raspberry...
box_l = 90.0
kT = 0.001
time_step = 0.001
gamma = 10
force_cap = 10000

# Interaction parameter
eps_ss = 1  # LJ epsilon
sig_ss = 1  # LJ sigma

# endregion
#############################################################################################################################
# region: AXES
axes = np.array(
    [b, b, a]
)
# find upper and lower bound volumes of ellipsoid
MAX_VOLUME_FRACTION = 0.74048  # Hcp packing 3D
ell_vol = ellipsoid_volume(axes)

# lower bound is volume of hcp fraction per particle
ell_vol_lower_bound = n_particles * np.pi/6 / MAX_VOLUME_FRACTION
# upper bound is volume of 1 per particle
ell_vol_upper_bound = n_particles
# convert to scaling factors to be aplied to ellipsoid axes
scaling_factor = (ell_vol_lower_bound / ell_vol)**(1/3)
upper_bound_factor = (ell_vol_upper_bound / ell_vol)**(1/3)

effective_factor = axes[0]*scaling_factor-0.5
upper_bound_factor /= effective_factor

# variable for Ellipsoid dimensions
axes = np.array([scaling_factor * i for i in axes])
# would now theoretically encompass all spheres, but we want the ell surface through the centers of outermost particles
axes = axes - 0.5
# copy of original so we can calculate new value of axes later
OG_AXES = list(axes)

# endregion
#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = 3  # Skin parameter for the Verlet lists
system.periodicity = [True, True, True]

# the LJ potential between surface beads
system.non_bonded_inter[0, 0].lennard_jones.set_params(  # sigma at 1.0 has overlaps forever, too high and density is suboptimal.
    # nr is possibly worth messing with
    epsilon=1, sigma=1.1, cutoff=2.5, shift="auto")

# Going to even lower temp regime 0.00001 might make sense, but this seems to function well
system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)

print(LINE)
print("# Creating raspberry", axes, debug_override=True)

print("Particle_NR: ", n_particles)
# Generate list of positions
spacing = 1
while True:
    arr_of_points = hex_ellipsoid_points(
        axes[0], axes[1], axes[2], spacing=spacing)
    if (len(arr_of_points) > n_particles):
        break
    spacing -= 0.01
print("INITIAL SPACING:", spacing)

# Place particles
for i, pos in enumerate(arr_of_points):
    if i > n_particles:
        break
    system.part.add(pos=pos)

# Large overlaps could explode the system. Baby steps.
system.force_cap = force_cap

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

k = 0  # iteration variable
check_nth = 100  # nr of iterations before checking progress
thermal_kick = 1  # 0 means off, if set to 1 counts up to x and resets itself
checkpoint_shake = False  # flag for one time thermal kick at 0.99 score reached

high_score_smallest = 0
smallest_list = []  # list for plotting progress afterwards
highest_unsuccessful_factor = [1.0]
smallest_success_factor = [upper_bound_factor]

current_factor = (
    smallest_success_factor[-1] + highest_unsuccessful_factor[-1]) / 2
factor_escape_margin = (upper_bound_factor - 1) / 100
axes = np.array([i * current_factor for i in OG_AXES])

# goes up when no new high scores, determines when configuration can't be found
no_improvement_counter = 0
no_impr_lim = 15
og_reset_surf = 1
surface_reset_nth = og_reset_surf

# tracker lists for when factor ellipsoid is scaled up or down
size_up = []
size_down = []
factors_list = [(1.0, 0), (upper_bound_factor, 1)]

# id of current particle that will try to push into the ellipsoid shell for better configuration
pushing_particle_id = 0
dist_curr = 0  # its distance from the shell
push_flag = True
nr_of_part_in_shell = 0

print(LINE)
print("Starting simulation")
print(LINE)

t_check = 0
t_vis = 0
t_surface = 0
t_noconfig = 0

flag20 = True
flag50 = True

triv = False
extra_frame = False
while True:
    if triv:
        triv = False
        print(LINE)
        print("TRIVVVVVVVVVVVVVVV CONF")
        print(LINE)

    ###
    if extra_frame:
        # Overlap list
        overlap_ids = detect_overlap_ids(
            particle_positions,
            axes,
        )

        # colour particles with overlap
        system.part.all().director = [0, 0, -1]
        if len(overlap_ids) > 0:
            system.part.by_ids(overlap_ids).director = [0, 0, 1]
        if save_visualization:
            writevtk(
                f"_data/vtk_frames/one_go/frame_{int(k/nth_step_recorded)}.vtk", system, half_box_shift=True, dir=True)
            extra_frame = False
    ###

    t0 = time.time()
    # Record frame
    if not k % nth_step_recorded or extra_frame:
        if save_visualization:
            writevtk(
                f"_data/vtk_frames/one_go/frame_{int(k/nth_step_recorded)}.vtk", system, half_box_shift=True, dir=True)
            extra_frame = False
    t_vis += time.time()-t0

    # Single step
    system.integrator.run(1)

    t0 = time.time()
    # Put all particles back on the surface
    if not k % surface_reset_nth:  # seems generally better, lets escape minima better and reduces time spentby a lot
        particle_positions = system.part.all().pos
        new_poss = shift_particle(particle_positions, axes, if_outside=True)
        for part, pos in zip(system.part.all(), new_poss):
            part.pos = pos
        t_surface += time.time()-t0

    t0 = time.time()
    k += 1
    if not k % check_nth:

        particle_positions = system.part.all().pos

        # region: ############### Pushing section ###############
        # Distance to shell
        new_dist_curr = round(np.linalg.norm(
            system.part.by_id(pushing_particle_id).pos), 2)

        # as long as pushing outside progresses, no killing system
        if new_dist_curr > dist_curr:
            no_improvement_counter = 0
            dist_curr = new_dist_curr

        # Reset forces
        system.part.by_id(pushing_particle_id).ext_force = [0, 0, 0]

        if push_flag:
            # Find new particle to push outwards, should remain the same as long as it is progressing
            pushing_particle_id = get_particle_closest_to_boundary(
                particle_positions, axes, margin=0.1
            )
            position_vector = np.array(
                system.part.by_id(pushing_particle_id).pos)

            # apply force
            # nr is possibly worth messing with
            system.part.by_id(
                pushing_particle_id).ext_force = 100 * position_vector

        # Shell count
        nr_of_part_in_shell = count_particles_in_shell(
            particle_positions, axes, margin=0.2
        )

        # Overlap list
        overlap_ids = detect_overlap_ids(
            particle_positions,
            axes,
        )

        # colour particles with overlap
        system.part.all().director = [0, 0, -1]
        if len(overlap_ids) > 0:
            system.part.by_ids(overlap_ids).director = [0, 0, 1]

            # disable pushing particles outside when internal stress is low or after sufficient time with no improvement
            if (particles_not_in_shell(
                particle_positions,
                overlap_ids,
                axes,
            ) <= 1 or no_improvement_counter == no_impr_lim-1) and push_flag:  # nr is possibly worth messing with
                print("DISABLE PUSHING")
                no_improvement_counter = 0
                # surface_reset_nth = og_reset_surf
                push_flag = False
                kick()
        system.part.by_id(pushing_particle_id).director = [1, 0, 0]

        # endregion ######################################################

        # Calculate metrics
        mean, rmsd, smallest = mean_rmsd_smallest(
            system.part.all().id, system)  # TODO remove the useless shit
        smallest_list.append(smallest)

        # Detect improvement
        if smallest > high_score_smallest:
            no_improvement_counter = 0
            high_score_smallest = smallest
            print("HIGHSCORE: ", smallest)
        # No tracking lack of improvement during thermal kick
        elif not thermal_kick:
            no_improvement_counter += 1

        # Detect successful configuration
        if smallest > 1:

            print(LINE)
            print("SUCCESS: ", current_factor)
            factors_list.append((current_factor, 1))
            best_configuration = system.part.all().pos

            # Reset flags
            high_score_smallest = 0
            no_improvement_counter = 0
            push_flag = True

            # Finishing condition approaching unsuccessfull floor
            if abs(smallest_success_factor[-1] - highest_unsuccessful_factor[-1]) < factor_escape_margin:
                plot_tried_factors()
                break
            # Try to find smaller bound configuration
            else:
                smallest_success_factor.append(float(current_factor))
                current_factor = (
                    current_factor + highest_unsuccessful_factor[-1]) / 2

                # Execute size down
                print("FACTOR DOWN:", current_factor)
                axes = np.array([i * current_factor for i in OG_AXES])

                # track timestamp for plotting of evolution of smallest dist
                size_down.append(k)

                # Reset positions
                triv = shuffle_pos(system, axes)
                extra_frame = True
                # for part in system.part.all():
                #     part.pos = [0,0,0]

                # Reset trackers/flags
                high_score_smallest = 0
                checkpoint_shake = False
                surface_reset_nth = og_reset_surf

                # Thermal kick
                kick()

                continue
    t_check += time.time() - t0

    t0 = time.time()
    # Detect no valid configuration being found
    if (high_score_smallest > 0.9 and no_improvement_counter == no_impr_lim) or (high_score_smallest < 0.9 and no_improvement_counter == 10):
        no_improvement_counter = 0
        push_flag = True

        print(LINE)
        print("UNSUCCESSFUL: ", current_factor)
        highest_unsuccessful_factor.append(current_factor)
        factors_list.append((current_factor, 0))

        current_factor = (current_factor + smallest_success_factor[-1])/2

        # track timestamp for plotting of evolution of smallest dist
        size_up.append(k)

        # Finishing condition approaching successfull ceiling
        if abs(current_factor - smallest_success_factor[-1]) < factor_escape_margin:
            plot_tried_factors()
            break

        # Apply size increase
        print("FACTOR UP: ", current_factor)
        axes = np.array([i * current_factor for i in OG_AXES])

        # Particle positions reset
        triv = shuffle_pos(system, axes)
        extra_frame = True
        # for part in system.part.all():
        #     part.pos = [0, 0, 0]

        # Reset trackers/flags
        high_score_smallest = 0
        checkpoint_shake = False
        surface_reset_nth = og_reset_surf

        # Thermal kick
        kick()

        continue
    t_noconfig += time.time()-t0

    # One time shuffle when close to valid configuration
    if high_score_smallest > 0.99 and not checkpoint_shake:
        checkpoint_shake = True
        # surface_reset_nth = og_reset_surf
        kick()

    if no_improvement_counter == 20 and flag20:
        flag20 = False
        kick()

    if no_improvement_counter == 50 and flag50:
        flag50 = False
        kick()

    # Disabeling thermal kick
    if thermal_kick:
        thermal_kick += 1
        if thermal_kick == 1000:
            system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
            print(LINE)
            print("thermal kick OFF")
            thermal_kick = 0

    # Reporting progress
    if not k % 100:
        print(LINE)
        print("k = ", k)
        print("Particles in shell:", nr_of_part_in_shell)
        print("Pushing particles into shell: ", push_flag)
        print("No imporvement counter: ", no_improvement_counter)
        print(LINE)

# Timings
print(
    "t_check", t_check, "\n",
    "t_vis", t_vis, "\n",
    "t_surface", t_surface, "\n",
    "t_noconfig", t_noconfig, "\n",
)

# Primary output of final coordinates of particles
filename = f"_data/coordinates/{n_particles}_ratio_{a}_{b}.txt"
filename = filename.replace(f"._", f".0_")
filename = filename.replace(f".]", f".0]")
np.savetxt(filename, best_configuration)
print(LINE)
print(filename)
print("contains coordinates of raspberry")


# smallest neighbour distance evolution over time
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


# Saving performance metrics to .json
t_check = time.time() - t0
filename = "geom.json"
entry = {
    "type": "one_go",
    "pnr": len(system.part.all())-2,
    "factor": current_factor,
    "k": k,
    "rmsd": rmsd,
    "time": t_check,
    "kT": kT,
    "gamma": gamma,
    "force cap": force_cap,
    "comment": "epsilon = 20"
}
save_to_json(filename, entry)

print("Reached the end.")
print(LINE)
print("factor found: ", smallest_success_factor[-1])
print("with no_impr_lim: ", no_impr_lim)
