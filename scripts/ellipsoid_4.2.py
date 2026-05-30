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
from espressomd.virtual_sites import VirtualSitesRelative

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

# Manual
# a = 3
# b = 2


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
    [a, b, b]
)
radius_col = max(AXES)

# endregion

#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]

# the LJ potential (WCA potential) between surface beads
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss+0.1
)

# Seed
seed_pass = np.random.randint(0, 2**32 - 1)
print("Seed Pass: ", seed_pass)

# Going to even lower temp regime 0.00001 might make sense, but this seems to function well
system.thermostat.set_langevin(kT=0.001, gamma=40.0, seed=seed_pass)

print(LINE)
print("# Creating raspberry", AXES, override=True)
center = system.box_l / 2
colPos = (0, 0, 0)

# Place the central particle
system.part.add(id=0, pos=colPos, type=0, fix=(
    True, True, True), rotation=(1, 1, 1))

# endregion

#############################################################################################################################
# region: Outer Shell

# Nr of particles for shell, is an upper bound for if the ellipsoid surface was flat 2D and hex packaging.
# There is surely a smaller upper bound, but the particles have to spread all the same. I don't expect a high speedup from a better initial guess.
n_surface_part = (
    int((ell_surface_area(AXES) * MAX_SURFACE_FRACTION / ((sig_ss / 2) ** 2 * np.pi)))
)
print("Surface Beads: ", n_surface_part)


# Create surface beads uniformly distributed over the surface
# Corrction: That's not uniform. Doesn't matter tho, so I'm leaving it in.
for i in range(1, n_surface_part + 1):
    u = np.random.rand()
    v = np.random.rand()
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x_s = np.sin(phi) * np.cos(theta)
    y_s = np.sin(phi) * np.sin(theta)
    z_s = np.cos(phi)

    x = AXES[0] * x_s
    y = AXES[1] * y_s
    z = AXES[2] * z_s

    colSurfPos = np.array([x, y, z])
    system.part.add(id=i, pos=colSurfPos, type=1)

################## Iterating ###################

# Empty folder for recording frames for Paraview
if save_visualization:
    os.makedirs("data/vtk_frames/shell_animation", exist_ok=True)
    folder_path = Path("data/vtk_frames/shell_animation")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
nth_step_recorded = 25  # record every 25th simulation step

# Large overlaps with WCA could explode the system. Baby steps.
system.force_cap = 1000

print(LINE)
print("Relaxation of the surface particles")
t0 = time.time()

k = 0  # iteration variable
check_nth = 100  # nr of iterations before checking progress

high_score_mean = 0
high_score_smallest = 0
smallest_list = []  # list for plotting progress afterwards
removed_at = []  # records time when particles are removed

no_improvement_counter = 0  # goes up when no new high scores
wait_nr = 5  # target value of no_improvement_counter before removing a particle

high_t = True
t_counter = 0  # counts iterations with high temperature
phase_duration = 10  # duration of high temperature state (* check_nth)


while True:

    if DEBUG and k == 1000:
        print(LINE)
        print(LINE)
        print("DEBUG BREAK")
        print(LINE)
        print(LINE)

        break

    system.integrator.run(1)

    # now put all particles back on the surface
    particle_positions = system.part.select(type=1).pos
    new_poss = shift_particle(particle_positions, AXES)
    for part, pos in zip(system.part.select(type=1), new_poss):
        part.pos = pos

    # Record frame
    if not k % nth_step_recorded:
        if save_visualization:
            writevtk(
                f"data/vtk_frames/shell_animation/frame_{int(k/nth_step_recorded)}.vtk", system)

    k += 1
    if not k % check_nth:

        # Check what to do about thermally exited mode
        if high_t:
            # Cooler system after 2/3 of time
            if t_counter == int(phase_duration/3*2):
                system.thermostat.set_langevin(
                    kT=0.01, gamma=1, seed=seed_pass)
            # Reset to low T domain
            if t_counter == phase_duration:
                system.thermostat.set_langevin(
                    kT=0.001, gamma=40.0, seed=seed_pass)
                high_t = False
                t_counter = 0
                print(LINE)
                print("THERMAL KICK OVER")
            else:
                t_counter += 1

        # Calculate metrics
        mean, std, smallest = mean_std_smallest(
            system.part.select(type=1).id, system)
        rel_mean = abs(high_score_mean-mean)/(1-high_score_mean)
        rel_smallest = abs(high_score_smallest-smallest) / \
            (1-high_score_smallest)

        # Record history for plot
        if not k % 100:
            smallest_list.append(smallest)

        # Set longer thermal kicks after certain point
        # Because the fewer the particles, the slower they are to spread out by themselves
        if high_score_mean > 0.98:
            phase_duration = 50

        ### Particle Removal ###
        # After smallest distance = 0.99 removing another particle is overkill. Just noise to be settled. Make exception after long enough.
        if (no_improvement_counter == wait_nr and high_score_smallest < 0.99 and high_score_mean < 0.999) or no_improvement_counter == 100:

            # Find particle furthest from center to be removed.
            # Avoids disturbing already formed compact lattices.
            furthest = 0
            for part in system.part:
                if np.linalg.norm(part.pos) > furthest:
                    furthest = np.linalg.norm(part.pos)
                    id_of_furthest = part.id
            system.part.by_id(id_of_furthest).remove()

            # Remove particle
            print(LINE)
            print("PARTICLE REMOVED")
            print("Nr of surface particles: ", len(system.part)-1)

            if not k % 100:
                removed_at.append(k)
            no_improvement_counter = 0

            # Initiate thermal kick to skip long equilibriation after removal
            print(LINE)
            print("THERMAL KICK")
            system.thermostat.set_langevin(
                kT=0.1, gamma=1, seed=seed_pass)
            high_t = True

        ### Monitor progress section (only outside high_t) ###
        if not high_t:

            # Catch insignificant progress, for speed up purposes, not in the final stretch
            # Doesn't apply near the end by nature, but maybe not always I dunno
            if rel_mean < 0.05 and high_score_mean - mean < 0:
                no_improvement_counter += 1
                print(LINE)
                print("little progress")
                print("rel_mean: ", round(rel_mean, 3))
                print("rel_smallest: ", round(rel_smallest, 3))
            # Improved smallest dist
            elif smallest > high_score_smallest:
                high_score_smallest = smallest
                no_improvement_counter = 0

                print("Steps: ", k)
                print("Mean: ", mean)
                print("Smallest: ", smallest)
                print("Nr of surface particles: ", len(system.part)-1)
            # Or improved mean dist
            elif mean > high_score_mean:
                high_score_mean = mean
                no_improvement_counter = 0

                print("Steps: ", k)
                print("Mean: ", mean)
                print("Smallest: ", smallest)
                print("Nr of surface particles: ", len(system.part)-1)
            # If no improvements and no miniscule ones: register lack of progress
            else:
                print(LINE)
                print("NO PROGRESS")
                no_improvement_counter += 1
                continue
        else:
            print("Steps: ", k)

        # Exit condition
        if smallest > 1:
            break

print(LINE)
print("Relaxation steps taken: ", k)
print("Time for shell: ", time.time() - t0, override=True)

# Restore low temperature regime
system.thermostat.set_langevin(kT=0.001, gamma=40.0, seed=seed_pass)

smallest_list = smallest_list[5:]
if save_visualization:
    plt.plot(np.arange(len(smallest_list)), smallest_list)
    for i in removed_at:
        plt.axvline(x=int(i/100), color='red', linestyle='--', linewidth=1.0)
    plt.savefig("data/shell_relaxation.png")

# endregion

#############################################################################################################################
# region: Virtual sites

# Select the desired implementation for virtual sites
system.virtual_sites = VirtualSitesRelative()
# min_global_cut needs to be bigger than all virtual bond lengths
system.min_global_cut = radius_col

# Calculate the center of mass position (com) and the moment of inertia (momI) of the colloid
com = np.average(system.part.select(type=1).pos, axis=0)
momIx = 0
momIy = 0
momIz = 0

for i in system.part.select(type=1).id:
    momIx += np.power(np.linalg.norm(com[1:] -
                      system.part.by_id(i).pos[1:]), 2)
    momIy += np.power(
        np.linalg.norm(
            np.array(
                [
                    com[0] - system.part.by_id(i).pos[0],
                    com[2] - system.part.by_id(i).pos[2],
                ]
            )
        ),
        2,
    )
    momIz += np.power(np.linalg.norm(com[:2] -
                      system.part.by_id(i).pos[:2]), 2)

# note that the real (as in non virtual, not physically) particle must be at the center of mass of the colloid because of the integrator
print("\n# moving central particle from {} to {}".format(
    system.part.by_id(0).pos, com))
system.part.by_id(0).fix = [False, False, False]
system.part.by_id(0).pos = com
system.part.by_id(0).mass = len(system.part.select(type=1))

system.part.by_id(0).rinertia = np.array([momIx, momIy, momIz])
print("System rotational moment of inertia: ", system.part.by_id(0).rinertia)

# Convert the surface particles to virtual sites related to the central particle
for p in system.part.select(type=1):
    p.vs_auto_relate_to(0)

system.part.by_id(0).pos = (0, 0, 0)
system.integrator.run(0)

# endregion

#############################################################################################################################
# region: Inner

# type 2 particles fill the volume
# goal here is to fill the ellipsoid with as many particles as allowed by hcp, relax them and then reduce the nr if needed
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=0.0, sigma=1.0, cutoff=1.0, shift=0.0
)
system.non_bonded_inter[1, 2].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss
)
system.non_bonded_inter[2, 2].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss
)


################################
# If perfect Hcp: last sphere has to be placed at shell - tetraheder hight
arr_of_points = hex_ellipsoid_points(
    AXES[0], AXES[1], AXES[1], box_l)

for pos in arr_of_points:
    system.part.add(pos=pos, type=2)

system.force_cap = 100

system.part.by_id(0).pos = center
system.part.by_id(0).fix = [1, 1, 1]
system.part.by_id(0).rotation = [0, 0, 0]


print("Relaxation of the filling particles")
t0 = time.time()


# Empty folder for recording frames for Paraview
if save_visualization:
    os.makedirs("data/vtk_frames/inside_animation", exist_ok=True)
    folder_path = Path("data/vtk_frames/inside_animation")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()

k = 0

c_counter = 0
t_counter = 0
high_t = False
forces = []
second_stage = False
phase_duration = 300
while True:
    if DEBUG and k == 1000:
        print(LINE)
        print(LINE)
        print("DEBUG BREAK")
        print(LINE)
        print(LINE)
        break

    system.integrator.run(1)

    # Disable thermal kick
    if high_t and t_counter == phase_duration:
        system.thermostat.set_langevin(kT=0.001, gamma=40.0, seed=seed_pass)
        high_t = False
        t_counter = 0
        print(LINE)
        print("THERMAL KICK OFF")
    elif high_t:
        t_counter += 1

    if not c_counter % phase_duration and c_counter and second_stage and not high_t:
        # Find particle furthest from center t be removed.
        # Avoids disturbing already formed surface lattices.
        furthest = 0
        for part in system.part.select(type=2):
            if np.linalg.norm([i - j for (i, j) in zip(part.pos, center)]) > furthest:
                furthest = np.linalg.norm(
                    [i - j for (i, j) in zip(part.pos, center)])
                id_of_furthest = part.id

        # Remove particle
        system.part.by_id(id_of_furthest).remove()
        print(LINE)
        print("PARTICLE REMOVED")

        # Thermal kick
        system.thermostat.set_langevin(kT=0.1, gamma=1, seed=seed_pass)
        high_t = True
        print(LINE)
        print("THERMAL KICK ON")

        c_counter = 0
    elif not high_t:
        c_counter += 1

    if not k % 25:
        if save_visualization:
            writevtk(f"data/vtk_frames/inside_animation/inside{k}.vtk", system)
            forces.append(max([np.linalg.norm(part.f)
                               for part in system.part.select(type=2)]))

    # Remove all particles outside and transition to second stage
    if k % 100 == 0 and not second_stage:
        if max([np.linalg.norm(part.f) for part in system.part.select(type=2)]) < 70 and not second_stage:
            second_stage = True
            print(LINE)
            print("ENTERING SECOND STAGE")
            for part in system.part.select(type=2):
                x, y, z = part.pos
                x, y, z = [i - j for (i, j) in zip(part.pos, center)]
                a, b, c = AXES

                s = (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c)
                if s > 1:
                    part.remove()
                    k += 25
                    if save_visualization:
                        writevtk(
                            f"data/vtk_frames/inside_animation/inside{k}.vtk", system)

    if not k % 100:
        mean, std, smallest = mean_std_smallest(
            system.part.all().id[1:], system)
        print("Steps: ", k)
        print("Mean: ", mean)
        print("Std: ", std)
        print("Smallest: ", smallest)
        print("Nr of filling particles: ", len(system.part.select(type=2)))

        if mean > 0.95 or smallest > 0.90:
            phase_duration = 1500
        if smallest > 1:
            break

    k += 1
if save_visualization:
    plt.plot(np.arange(len(forces)), forces)
    plt.savefig("data/max_forces.png")

print("Relaxation steps taken: ", k)
print("Time for inside: ", time.time() - t0, override=True)

print(LINE)
print("Shell particles: ", len(system.part.select(type=1)))
print("Inner particles: ", len(system.part.select(type=2)))
print("Steps taken: ", k)

# endregion

#############################################################################################################################
# region: Output

# write coordinates to textfile
print(str(AXES).replace(" ", "_"))
filename = "data/raspberry_coordinates" + str(AXES).replace(" ", "_") + ".txt"

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
# endregion
