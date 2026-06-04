#############################################################################################################################
# #############################################################################################################################

# README
# Generates an Ellipsoid geometry raspberry particle filled as dense as can be, but without overlaps.
# Step 1: In a shell on the surface.
# Step 2: Filling the inside.

# You should only really need to change the AXES to what you want.
# If funky results, enable save_visualization. Written as series of frames for paraview.

# With the current configuration 99% of the calculation time is spent making sure the max nr of particles is approached slowly.
# The surface more so than the inside. Even so, *no guarantee at all* that it finds the best nr.
# If you don't care much about having the highest possible density, you can massively reduce the time this takes.
# Gotta mess with the parameters yourself.

# TODO I am sure there easily are better starting configurations / lower bounds for initial particle nr than what I am doing.
# I am guilty of "Just throw compute at it over night."

#############################################################################################################################
#############################################################################################################################
# region: Import

import os
import sys
from pathlib import Path

import espressomd

import time
import numba
import numpy as np
import matplotlib.pyplot as plt

from functions import *

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

# endregion
# region: User input

# Redefining print to only obtain barebones output when DEBUG = FALSE
# Also enables absolute cut off points when itteration nr reached
DEBUG = True  # toggle this


def print(*args, override=False, **kwargs):
    if DEBUG or override:
        __builtins__.print(*args, **kwargs)

# Define the semi-axes of the ellipsoid. [a, b, b]
# Only Spheroids a != b = c, because otherwise the resetting to surface breaks, I think.
# Run time consideration: 5,5,5 took ~4h. 3,2,2 10min


# for running from outside with args if you want several geometries cued up with cue_AXES.py
a = float(sys.argv[1])
b = float(sys.argv[2])

# Manual aspüect ratio
# a = 2
# b = 1

n_particles = 512

# Writes vtk animation frames of the shell process and inner process, doesn't hamper performance much at all
save_visualization = True
# endregion

#############################################################################################################################
# region: Numba setup


# Numba achieves a 20x speed up the way I set it up and it only eats ~34% of my cpu vs 25% before.
numba.set_num_threads(4)  # or the number of cores you want
print("Number of threads: ", numba.get_num_threads())

# @njit(parallel=True)
# def test_parallel(n):
#     for i in prange(n):
#         print(i)  # Will this print multiple numbers at the same time?
#         # simulate work
#         x = 0
#         for _ in range(1000000):
#             x += 1
#     return


# test_parallel(100)
# exit()
# results should show up (1,2,3,4,5,25,26,27,28,51,52,53,6,7,8) then multithreading is working correctly.

# endregion

#############################################################################################################################

# region: Constants

# output formatting
LINE = "\n___________________________"
# Hex packing 2D
MAX_SURFACE_FRACTION = 0.907
# Hcp packing 3D
MAX_VOLUME_FRACTION = 0.74048

# endregion
# region: System parameters

# size of the simulation box, arbitrary, should be larger than the raspberry...
box_l = 90.0
center = [box_l / 2 for _ in range(3)]

skin = 3  # Skin parameter for the Verlet lists
time_step = 0.001

# Interaction parameter
eps_ss = 1  # LJ epsilon
sig_ss = 1  # LJ sigma


AXES = np.array(
    [b, b, a]
)

ell_vol = ellipsoid_volume(AXES)
sphere_vol_lower_bound = n_particles * 4 / \
    3 * np.pi * (1/2)**3 / MAX_VOLUME_FRACTION
scaling_factor = (sphere_vol_lower_bound / ell_vol)**(1/3)
AXES = np.array([scaling_factor * i for i in AXES])
# would now theoretically encompass all spheres, but we want the ell surface through the centers of outermost particles
AXES = AXES - 0.5
# hcp is highly idealistic, work up from there.
# it's not? HUH?
# !!! huh?

print("Lowerbound for Axes: \n", AXES)

# endregion

#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]

# the LJ potential (WCA potential) between surface beads
system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss+0.1
)

# Seed
seed_pass = np.random.randint(0, 2**16 - 1)
print("Seed Pass: ", seed_pass)

# Going to even lower temp regime 0.00001 might make sense, but this seems to function well
system.thermostat.set_langevin(kT=0.001, gamma=1, seed=seed_pass)

print(LINE)
print("# Creating raspberry", AXES, override=True)
center = system.box_l / 2

# endregion

#############################################################################################################################
# region: Outer Shell

# Nr of particles for shell, is an upper bound for if the ellipsoid surface was flat 2D and hex packaging.
# There is surely a smaller upper bound, but the particles have to spread all the same. I don't expect a high speedup from a better initial guess.

print("Particle_NR: ", n_particles)

tolerance = 0
while True:
    arr_of_points = hex_ellipsoid_points(
        AXES[0]+tolerance, AXES[1]+tolerance, AXES[2]+tolerance, spacing=0.99)

    print("Lattice sites: ", len(arr_of_points))
    if (len(arr_of_points) > n_particles):
        break
    tolerance += 0.01


for i, pos in enumerate(arr_of_points):
    if i > n_particles:
        break
    system.part.add(pos=pos)

################## Iterating ###################

# Empty folder for recording frames for Paraview
if save_visualization:
    os.makedirs("_data/vtk_frames/one_go", exist_ok=True)
    folder_path = Path("_data/vtk_frames/one_go")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
nth_step_recorded = 25  # record every 25th simulation step

# Large overlaps with WCA could explode the system. Baby steps.
system.force_cap = 1000

print(LINE)
print("Relaxation of particles")
t0 = time.time()

k = 0  # iteration variable
check_nth = 100  # nr of iterations before checking progress

high_score_smallest = 0
smallest_list = []  # list for plotting progress afterwards

no_improvement_counter = 0  # goes up when no new high scores
size_up = []
size_down = []

may_grow = True

while True:

    # if DEBUG and k == 10000:
    #     print(LINE)
    #     print(LINE)
    #     print("DEBUG BREAK")
    #     print(LINE)
    #     print(LINE)
    #     break

    system.integrator.run(1)

    # now put all particles back on the surface
    particle_positions = system.part.all().pos
    new_poss = shift_particle(particle_positions, AXES, if_outside=True)
    for part, pos in zip(system.part.all(), new_poss):
        part.pos = pos

    # Record frame
    if not k % nth_step_recorded:
        if save_visualization:
            writevtk(
                f"_data/vtk_frames/one_go/frame_{int(k/nth_step_recorded)}.vtk", system)

    k += 1

    if not k % check_nth:
        # Calculate metrics
        mean, std, smallest = mean_std_smallest(
            system.part.all().id, system)

        print(LINE)
        print("smallest: ", smallest)
        print("mean: ", mean)
        print("std: ", std)

        if smallest > 1:
            high_score_smallest = 0
            no_improvement_counter = 0
            if len(size_up) == 0:
                AXES = np.array([i * 0.95 for i in AXES])
                size_down.append(k)
            else:
                break

        smallest_list.append(smallest)

        if smallest > high_score_smallest:
            no_improvement_counter = 0
            high_score_smallest = smallest
        else:
            no_improvement_counter += 1

    if no_improvement_counter == 10 and may_grow:
        no_improvement_counter = 0
        AXES = np.array([i * 1.05 for i in AXES])
        print("\n")
        print(LINE)
        print("Increased Ellipsoid size, to: ")
        print(AXES)
        size_up.append(k)

    if high_score_smallest > 0.99 and may_grow:
        print("Low temp regime engaged")
        system.thermostat.set_langevin(kT=0.0001, gamma=10, seed=seed_pass)
        may_grow = False

filename = f"_data/coordinates/{n_particles}_ratio_{a}_{b}.txt"

filename = filename.replace(f"._", f".0_")
filename = filename.replace(f".]", f".0]")

positions = []
for part in system.part:
    # Except central particle
    if not part.id:
        continue
    positions.append(part.pos)
np.savetxt(filename, np.array(positions))

print(LINE)
print(filename)
print("contains coordinates of raspberry")

print("Reached the end.")
print(LINE)

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
