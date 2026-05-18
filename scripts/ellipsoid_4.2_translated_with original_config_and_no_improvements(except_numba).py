# Currently translating to 4.2
# Only then! correcting mistakes and shortening to necessary bits
# Then expanding to different aspect ratios

# Seems to have nothing to do with dense packing so far

import random
import os
import time
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

import numba
from numba import njit, prange

from espressomd.io.writer import vtf
from espressomd.virtual_sites import VirtualSitesRelative
from espressomd import lb

import espressomd.accumulators
import espressomd.observables

from espressomd import interactions
import espressomd

espressomd.assert_features(
    [
        "ROTATION",
        "ROTATIONAL_INERTIA",
        "EXTERNAL_FORCES",
        "MASS",
        "VIRTUAL_SITES_RELATIVE",
        "LENNARD_JONES",
    ]
)

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
# results should show up (1,2,3,4,5,25,26,27,28,51,52,53,6,7,8) then multithreading is working.


# System parameters
#############################################################
box_l = 90.0  # size of the simulation box, pretty arbitrary, should be larger than the raspberry....
number_of_raspberries = 1  # leave this at 1


skin = 3  # Skin parameter for the Verlet lists
time_step = 0.001
eq_tstep = 0.001

# equilibration is n_cycles*integ_steps each for outer shell and inner shells
n_cycle = 10
# n_cycle = 2
integ_steps = 300
# integ_steps = 100


# Interaction parameters (Lennard-Jones for each raspberry)
#############################################################

# the subscript c is for colloid and s is for salt (also used for the surface beads)

eps_ss = 1  # LJ epsilon between the colloid's surface particles.
sig_ss = 1  # LJ sigma between the colloid's surface particles.
radius_col = 3
harmonic_radius = radius_col


# Define the semi-axes of the ellipsoid:
a = 2.0
b = 1.5
c = 1.2
AXES = np.array([a, b, c])  # Give the lenth of the axes
# LJ sigma between the colloid's central particle and surface particles (colloid's radius).
sig_cs = radius_col

#############################################################
# Number of particles making up each raspberry (surface particles + the central particle). use liberally

# we approximate the surface area of the ellipsoid with the Knud Thomsens Formula (See wikipedia):


def ell_surface_area(axes):
    """axes has to be an array in the form [a,b,c] with the semi axes of the ellipsoid"""
    knud_exponent = 1.6075
    surf_area = (
        4.0
        * np.pi
        * (
            (
                (axes[0] * axes[1]) ** knud_exponent
                + (axes[0] * axes[2]) ** knud_exponent
                + (axes[1] * axes[2]) ** knud_exponent
            )
            / (3.0)
        )
        ** (1.0 / knud_exponent)
    )
    return surf_area


# n_col_part = int(5 * ell_surface_area(axes)) #no special reason for factor 5, use whatever rows your boat....
n_col_part = 82
n_fill_part = 42  # round(4*np.pi*pow(radius_col-sig_ss/2., 3)/(3*pow(agrid, 3)))+5
print("Surface Beads: ", n_col_part)
print("Filling Beads: ", n_fill_part)


# System setup
#############################################################
system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]


# the LJ potential (WCA potential) between surface beads causes them to be roughly equidistant on the
# colloid surface
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss
)

# Seed
seed_pass = np.random.randint(0, 2**32 - 1)  # valid 32bit range
print("Seed Pass: ", seed_pass)

# for the warmup we use a Langevin thermostat with an extremely low temperature and high friction coefficient
# such that the trajectories roughly follow the gradient of the potential while not accelerating too much
system.thermostat.set_langevin(kT=0.00001, gamma=40.0, seed=seed_pass)

print("# Creating raspberry")
center = system.box_l / 2
colPos = (0, 0, 0)


# Place the central particle
system.part.add(id=0, pos=colPos, type=0, fix=(True, True, True), rotation=(1, 1, 1))


##################################################################
############ define the 'potential' function #####################
##################################################################
@njit
def ellipsoid_potential_numba(t, axes, position):
    """axes should be an array with the a,b,c axes for the ellipsoid. position is the actual particle position.
    The function returns the nearest surface position as an array [x y z]"""
    return (
        (((position[0] * axes[0]) / (t + axes[0] ** 2.0))) ** 2.0
        + (((position[1] * axes[1]) / (t + axes[1] ** 2.0))) ** 2.0
        + (((position[2] * axes[2]) / (t + axes[2] ** 2.0))) ** 2.0
        - 1
    )


def ellipsoid_potential(t, axes, position):
    """axes should be an array with the a,b,c axes for the ellipsoid. position is the actual particle position.
    The function returns the nearest surface position as an array [x y z]"""
    return (
        (((position[0] * axes[0]) / (t + axes[0] ** 2.0))) ** 2.0
        + (((position[1] * axes[1]) / (t + axes[1] ** 2.0))) ** 2.0
        + (((position[2] * axes[2]) / (t + axes[2] ** 2.0))) ** 2.0
        - 1
    )


def bisect(l_limit, u_limit, function, axes, position):
    """Provide function and limits. This function returns the root of the function that is within the provided limits.
    It cannot handle the case in which there is more than one solution."""
    tol = 10 ** (-6.0)
    midpoint = (u_limit + l_limit) / 2.0
    # print(midpoint)
    if function(midpoint, axes, position) == 0.0:
        return midpoint
    elif (u_limit - l_limit) < tol:
        return midpoint
    elif function(midpoint, axes, position) * function(l_limit, axes, position) < 0:
        return bisect(l_limit, midpoint, function, axes, position)
    elif function(midpoint, axes, position) * function(u_limit, axes, position) < 0:
        return bisect(midpoint, u_limit, function, axes, position)
    else:
        print("There is no (or more than one) root between the limits.")


@njit
def bisect_numba(l_limit, u_limit, axes, position, tol=1e-6, max_iter=50):
    for _ in range(max_iter):
        midpoint = 0.5 * (l_limit + u_limit)
        f_mid = ellipsoid_potential_numba(midpoint, axes, position)
        f_l = ellipsoid_potential_numba(l_limit, axes, position)
        if abs(f_mid) < tol or abs(u_limit - l_limit) < tol:
            return midpoint
        if f_mid * f_l < 0:
            u_limit = midpoint
        else:
            l_limit = midpoint
    return midpoint


@njit
def get_limits_numba(function, axes, position):
    """This finds the vicinity of the highest root (lower than 100)...."""

    u_limit = 50  # this was 100 before
    l_limit = u_limit - 0.3  # this was 0.03 before
    i = 0

    # Slide down x-vals until root found
    while np.sign(function(u_limit, axes, position)) == np.sign(
        function(l_limit, axes, position)
    ):
        l_limit -= 0.1  # this was 0.01 before
        u_limit -= 0.1  # this was 0.01 before
        i += 1
        if i > 10**4.0:
            break
    return l_limit, u_limit


def get_limits(function, axes, position):
    """This finds the vicinity of the highest root (lower than 100)...."""
    u_limit = 50  # this was 100 before
    l_limit = u_limit - 0.3  # this was 0.03 before
    i = 0

    # Slide down x-vals until root found
    while np.sign(function(u_limit, axes, position)) == np.sign(
        function(l_limit, axes, position)
    ):
        l_limit -= 0.1  # this was 0.01 before
        u_limit -= 0.1  # this was 0.01 before
        i += 1
        if i > 10**4.0:
            # print("No root found")
            break
    return l_limit, u_limit


@njit
def get_surface_position_numba(t, axes, position):
    x_s = position[0] / ((t / axes[0] ** 2.0) + 1)
    y_s = position[1] / ((t / axes[1] ** 2.0) + 1)
    z_s = position[2] / ((t / axes[2] ** 2.0) + 1)
    return np.array([x_s, y_s, z_s])


def get_surface_position(t, axes, position):
    x_s = position[0] / ((t / axes[0] ** 2.0) + 1)
    y_s = position[1] / ((t / axes[1] ** 2.0) + 1)
    z_s = position[2] / ((t / axes[2] ** 2.0) + 1)
    return np.array([x_s, y_s, z_s])


##################################################################
############ end of define the 'potential' function ##############
##################################################################


# Create surface beads uniformly distributed over the surface of the central particle
# I think this is an approximate uniform distribution only, but it equalizes with the
# relaxation so no biggie...

xs = []
for i in range(1, n_col_part + 1):
    x = np.random.randn() * AXES[0]
    y = np.random.randn() * AXES[1]
    z = np.random.randn() * AXES[2]
    d = np.sqrt(
        (x**2.0 / AXES[0] ** 2.0)
        + (y**2.0 / AXES[1] ** 2.0)
        + (z**2.0 / AXES[2] ** 2.0)
    )
    colSurfPos = np.array([x, y, z] / d)
    system.part.add(id=i, pos=colSurfPos, type=1)
print("# Number of colloid beads = {}".format(n_col_part))


# Relax bead positions.
system.force_cap = 1000
system.time_step = eq_tstep


# TODO I've made a mistake in restructuring this I think. What I do now is get a whole set of new pos, based on the old set. Instead of 1 by 1 updating each and the next new pos depending on the last individual step. If that is necessary non of the parallelization is possible because numba needs to be pure python, no espresso interferance. Imho it should be fine. TEST IT!
@njit(parallel=True)
def shift_particle(particle_positions, axes=AXES, mult_axes=None):
    new_poss = np.zeros_like(particle_positions)
    for i in prange(len(particle_positions)):
        if mult_axes is not None:
            axes = mult_axes[i]
        # for i in range(len(particle_positions)):
        pos = particle_positions[i]
        l_limit, u_limit = get_limits_numba(ellipsoid_potential_numba, axes, pos)
        # l_limit, u_limit = get_limits(ellipsoid_potential, axes, pos)
        root = bisect_numba(l_limit, u_limit, axes, pos)
        # root = bisect(l_limit, u_limit, ellipsoid_potential, axes, pos)
        new_poss[i] = get_surface_position_numba(root, axes, pos)
        # new_poss[i] = get_surface_position(root, axes, pos)
    # print("progress: {:3.2f}%".format(j * 100.0 / integ_steps), end="\r")
    return new_poss


# TODO I feel like this can be massively shortened by condition, rather than just doing a fixed 3000 steps
print("Relaxation of the raspberry surface particles")
t0 = time.time()
for i in range(n_cycle):
    print("Integration cycle surface beads: " + str(i + 1) + " of " + str(n_cycle))
    for j in range(integ_steps):
        system.integrator.run(1)
        # now update all surface particle positions and put them back on the surface
        particle_positions = system.part.select(type=1).pos
        new_poss = shift_particle(particle_positions)
        for part, pos in zip(system.part.select(type=1), new_poss):
            part.pos = pos
print("Time: ", time.time() - t0)

# Restore time step
system.time_step = time_step

# region: TODO Disagree. This does nothing the last loop doesn't do.

# this loop moves the surface beads such that they are once again exactly radius_col away from the center
# For the scalar distance, we use system.distance() which considers periodic boundaries
# and the minimum image convention
colPos = system.part.by_id(0).pos
for particle in system.part.select(type=1):
    l_limit, u_limit = get_limits(ellipsoid_potential, AXES, particle.pos)
    root = bisect(l_limit, u_limit, ellipsoid_potential, AXES, particle.pos)
    particle.pos = get_surface_position(root, AXES, particle.pos)

# endregion

# region

# Select the desired implementation for virtual sites
system.virtual_sites = VirtualSitesRelative()
# Setting min_global_cut is necessary when there is no interaction defined with a range larger than
# the colloid such that the virtual particles are able to communicate their forces to the real particle
# at the center of the colloid
system.min_global_cut = radius_col  # not sure what to put here


###############################################
#######   Center of Mass    ###################
#######          &          ###################
#######  Moment of Inertia  ###################
###############################################


# Calculate the center of mass position (com) and the moment of inertia (momI) of the colloid
com = np.average(
    system.part.select(type=1).pos, axis=0
)  # system.part[:].pos returns an n-by-3 array
momIx = 0
momIy = 0
momIz = 0

# for each axis only positions away from that axis are relevant.
# TODO integrating over all, means that the central particle also has a contribution to the moment of Inertia
# up to this point I would've assumed that it was only going to be a refference point, even partially overlapped by the last filling particles or something
# used to all be system.part[i] seems like a mistake tbh considering the section after this
for i in range(n_col_part):
    momIx += np.power(np.linalg.norm(com[1:] - system.part.by_id(i).pos[1:]), 2)
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
    momIz += np.power(np.linalg.norm(com[:2] - system.part.by_id(i).pos[:2]), 2)
    # This shows that particle with id 0 adds a non zero mom, ofc small compared to same mass particles on shell distance to com
    # print(momIx, momIy, momIz)
    # print(system.part.by_id(i).type)
    # exit()

# note that the real particle must be at the center of mass of the colloid because of the integrator
print("\n# moving central particle from {} to {}".format(system.part.by_id(0).pos, com))
system.part.by_id(0).fix = [False, False, False]
system.part.by_id(0).pos = com
system.part.by_id(0).mass = n_col_part

system.part.by_id(0).rinertia = np.array([momIx, momIy, momIz])
print("System rotational moment of inertia: ", system.part.by_id(0).rinertia)

# Convert the surface particles to virtual sites related to the central particle
# The id of the central particles is 0, the ids of the surface particles start at 1.
for p in system.part.select(type=1):
    p.vs_auto_relate_to(0)

system.part.by_id(0).pos = (0, 0, 0)
system.integrator.run(0)


#### HERE THE INNER PARTICLES (Type 2) ARE CREATED ....
## Inner particles live in a second shell, that works the same way as the outer particle - shell
system.non_bonded_inter[2, 2].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss
)


# lets first suppose there is only one inner shell
# as a next step we could divide the shortest semi-axis in the same Way num_of_bins
# is doing it and then dividing the n_fill_part according to the fractions of
# the surface areas of the inner shells....


# the following makes a list of axes for the inner shells
# the number of the inner shells depends on the size of the smallest axis
# of the ellipsoid
number_of_inner_shells = int(np.ceil(min(AXES)))
# TODO so filling by rounding up the nr of necessary shells, seems to only work for weak contrasts in axes? otherwise to few particles?
inner_axes = [AXES - (i + 1) * (sig_ss / 2.0) for i in range(number_of_inner_shells)]
print("Inner axes: ", inner_axes)
print("Number of inner shells: ", number_of_inner_shells)
# The probabilites indicate the probability-distribution in which inner shell
# a particle will be put... (it is proportional to the surface are of the respective
# inner shell.)
probabilities = np.array([ell_surface_area(inner_axis) for inner_axis in inner_axes])
probabilities = probabilities / np.sum(probabilities)
# The following dict is needed to keep track which particle is in which shell
which_shell = []

for i in range(n_col_part + 1, n_col_part + n_fill_part + 1):
    shell_number = np.random.choice(number_of_inner_shells, p=probabilities)
    x = np.random.randn() * inner_axes[shell_number][0]
    y = np.random.randn() * inner_axes[shell_number][1]
    z = np.random.randn() * inner_axes[shell_number][2]
    d = np.sqrt(
        (x**2.0 / inner_axes[shell_number][0] ** 2.0)
        + (y**2.0 / inner_axes[shell_number][1] ** 2.0)
        + (z**2.0 / inner_axes[shell_number][2] ** 2.0)
    )
    colSurfPos = np.array([x, y, z] / d)
    system.part.add(id=i, pos=colSurfPos, type=2)
    which_shell.append(shell_number)

# print("# Number of filling beads = {}".format(n_fill_part))

system.force_cap = 100
system.time_step = eq_tstep
print("Relaxation of the raspberry filling particles")

# endregion

which_axes = np.array([inner_axes[i] for i in which_shell])
for i in range(n_cycle):
    print("Integration cycle filling beads: " + str(i + 1) + " of " + str(n_cycle))
    for j in range(integ_steps):

        system.integrator.run(1)
        # now update all surface particle positions and put them back on the surface
        particle_positions = system.part.select(type=2).pos
        new_poss = shift_particle(particle_positions, mult_axes=which_axes)
        for part, pos in zip(system.part.select(type=1), new_poss):
            part.pos = pos

system.time_step = time_step

for p in system.part.select(type=2):
    p.vs_auto_relate_to(0)

# This checks whether the particles in the outer shell are nicely distributed
# make sure that the mean values bellow correspond to values you expect!!!
# also by tuning the number of particles in the raspberry try to minimise the st_dev!
# you will see once you cant get better, doesnt take much time
# TODO change this product to combinations to avoid double counting
slice_master = system.part.select(type=1).id
lister = [
    np.linalg.norm(system.part.by_id(i).pos - system.part.by_id(j).pos)
    for i, j in product(slice_master, slice_master)
    if i != j
]
lister.sort()
# this works because among nxn distances, picking the smallest will be roughly equal to next neigbour distances only. Small std confirms.
print("mean of shell: ", np.mean(lister[: n_col_part - 10]))
print("std dev of shell:", np.std(lister[: n_col_part - 10]))

# this here shows the total surface on the shell taken up by particle cross-sections. At most should be like 69% or something hexagonal stacking.
# Mean of shell should be > 1 for non overlap. Maybe explicitly check that no distance < 1 even.
# print("surface:", ell_surface_area(axes))
# print("surface of circles on surface: ", n_col_part * np.pi / 4)
# exit()
#######################################################################

if n_fill_part > 0:
    slice_master = system.part.select(type=2).id
    # TODO product -> combinations
    lister = [
        np.linalg.norm(system.part.by_id(i).pos - system.part.by_id(j).pos)
        for i, j in product(slice_master, slice_master)
        if i != j
    ]
    print("mean of core: ", np.mean(lister[: n_col_part - 10]))
    print("std dev of core:", np.std(lister[: n_col_part - 10]))

system.part.by_id(0).pos = (0, 0, 0)
system.integrator.run(0)


custom_data = open(
    "raspberry_ellipsoid_visual" + str(AXES).replace(" ", "_") + ".vtf", mode="w+t"
)
vtf.writevsf(system, custom_data)
vtf.writevcf(system, custom_data)


# write coordinates to textfile
file = open("raspberry_coordinates" + str(AXES).replace(" ", "_") + ".txt", "w+t")
for x in system.part:
    file.write("{}\t{}\t{}\t{}\n".format(x.pos[0], x.pos[1], x.pos[2], x.type))
file.close()

print(
    "raspberry_coordinates"
    + str(AXES).replace(" ", "_")
    + ".txt contains coordinates of raspberry"
)


system.part.by_id(0).pos = center
system.integrator.run(0)


####HIER WIRD X,Y,Z NOCHMAL ALS VARIABLE VERWENDET, DIE ABER VORHER ANDERS VERWENDET WURDE....
####DAS IST VIELLEICHT NICHT DIE SCHLAUESTE VARIANTE.....

x = {k: [] for k in range(system.part.n_part_types)}
y = {k: [] for k in range(system.part.n_part_types)}
z = {k: [] for k in range(system.part.n_part_types)}

for particle in system.part:
    x[particle.type].append(particle.pos_folded[0])
    y[particle.type].append(particle.pos_folded[1])
    z[particle.type].append(particle.pos_folded[2])

fig = plt.figure()
ax = plt.axes(projection="3d")
for type in range(system.part.n_part_types):
    ax.scatter(x[type], y[type], z[type], label="type " + str(type), marker=".")
zoom = 10
plt.title("Close-up (zoom:{})".format(zoom), fontweight="bold")
plt.xlim(43, 47)
plt.ylim(43, 47)
ax.set_zlim(43, 47)
plt.xlabel("x-position", fontsize=10)
plt.ylabel("y-position", fontsize=10)
ax.set_zlabel("z-position", fontsize=10)

plt.legend()
plt.show()


print("We reached the end.")
