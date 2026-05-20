# Setup ellipsoid geometry in desired aspect ratios.
# Uses equillibriation to space particles, first on outer shell, then for the filling particles.
# Pushes excess filling particles outside of the outer shell, which can then be removed.
# It is an attempt to reach nice packing of an ellipsoid, but there is nothing theoretical or optimal about it.

# Only then! correcting mistakes and shortening to necessary bits
# Then expanding to different aspect ratios


#############################################################################################################################
# region: Import

from pathlib import Path
from espressomd.virtual_sites import VirtualSitesRelative
from espressomd.io.writer import vtf
from espressomd.shapes import Ellipsoid
import espressomd
from itertools import combinations
import numpy as np
import time
import numba
from numba import njit, prange
import matplotlib.pyplot as plt

# DATA PATH AND HOW TO SAVE
# plt.savefig("/home/xeranes/DATA/fig.png")

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
# region: System parameters

# output formatting
LINE = "\n___________________________"

# Hex packing 2D
MAX_SURFACE_FRACTION = 0.907
# Hcp packing 3D
MAX_VOLUME_FRACTION = 0.74048

# size of the simulation box, arbitrary, should be larger than the raspberry...
box_l = 90.0
center = [box_l / 2 for _ in range(3)]
number_of_raspberries = 1  # leave this at 1


skin = 3  # Skin parameter for the Verlet lists
time_step = 0.001
eq_tstep = 0.001

# Interaction parameters (Lennard-Jones for each raspberry)
eps_ss = 1  # LJ epsilon
sig_ss = 1  # LJ sigma

# Define the semi-axes of the ellipsoid: (HAS TO BE FLOAT for numba reasons)
a = 4
b = 2
# c = 3.0
AXES = np.array(
    [a, b, b]
)  # Lenth of the axes, the later constraint geometry only allows ellipsoids of rotation
radius_col = max(AXES)

# endregion

#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]

# the LJ potential (WCA potential) between surface beads causes them to be roughly equidistant on the
# colloid surface
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss+0.1
)

# Seed
seed_pass = np.random.randint(0, 2**32 - 1)  # valid 32bit range
print("Seed Pass: ", seed_pass)

# for the warmup we use a Langevin thermostat with an extremely low temperature and high friction coefficient
# such that the trajectories roughly follow the gradient of the potential while not accelerating too much
system.thermostat.set_langevin(kT=0.0001, gamma=40.0, seed=seed_pass)

print(LINE)
print("# Creating raspberry")
center = system.box_l / 2
colPos = (0, 0, 0)

# Place the central particle
system.part.add(id=0, pos=colPos, type=0, fix=(
    True, True, True), rotation=(1, 1, 1))

# endregion

#############################################################################################################################
# region: Functions


def ell_surface_area(axes):
    """
    We approximate the surface area of the ellipsoid with the Knud Thomsens Formula (See wikipedia)
    axes has to be an array in the form [a,b,c] with the semi axes of the ellipsoid.
    """
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


def ellipsoid_volume(axes):
    a, b, c = axes
    return (4.0 / 3.0) * np.pi * a * b * c


def mean_std_smallest(ids):
    "Gives mean, std and smallest distance of next neighbours between particles specified by id lists. Usually in one shell."
    lister = [
        np.linalg.norm(system.part.by_id(i).pos - system.part.by_id(j).pos)
        for i, j in combinations(ids, 2)
        if i != j
    ]

    lister.sort()
    mean = np.mean(lister[: n_col_part - 10])
    std = np.std(lister[: n_col_part - 10])
    smallest = min(lister)
    return mean, std, smallest


# Completely equivalent versions. Just use the non @njit version if you don't use numba

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


@njit
def bisect_numba(l_limit, u_limit, axes, position, tol=1e-6, max_iter=50):
    """
    Splits interval in half, checks which side contains the root and overwrites the interval to that.
    Iterate until root within tolerance distance.
    """
    f_l = ellipsoid_potential_numba(l_limit, axes, position)
    for _ in range(max_iter):
        midpoint = 0.5 * (l_limit + u_limit)
        f_mid = ellipsoid_potential_numba(midpoint, axes, position)
        if abs(f_mid) < tol or abs(u_limit - l_limit) < tol:
            return midpoint
            
        if f_mid * f_l < 0:
            u_limit = midpoint
        else:
            l_limit = midpoint
            f_l = f_mid
    return midpoint

def bisect(l_limit, u_limit, function, axes, position):
    """
    Splits interval in half, checks which side contains the root and overwrites the interval to that.
    Iterate until root within tolerance distance.
    """
    tol = 10 ** (-6.0)
    midpoint = (u_limit + l_limit) / 2.0
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
def get_limits_numba(function, axes, position):
    """
    Slides a bracket down the function and checks if the sign changes.
    So whether there's a root inside.
    """

    u_limit = 50  # this was 100 before
    l_limit = u_limit - 0.3  # this was 0.03 before
    i = 0

    # Slide down x-vals until root found
    # print(midpoint)
    while np.sign(function(u_limit, axes, position)) == np.sign(
        function(l_limit, axes, position)
    ):
        l_limit -= 0.1  # this was 0.01 before
        u_limit -= 0.1  # this was 0.01 before
        i += 1
        if i > 1e4:
            break
    return l_limit, u_limit


def get_limits(function, axes, position):
    """"
    Slides a bracket down the function and checks if the sign changes.
    So whether there's a root inside.
    """
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

# switch commented lines for non numba
@njit(parallel=True)
def shift_particle(particle_positions, axes=AXES):
    new_poss = np.zeros_like(particle_positions)
    for i in prange(len(particle_positions)):
    # for i in range(len(particle_positions)):
        pos = particle_positions[i]

        # Find region for position in which the ellipsoid surface is
        l_limit, u_limit = get_limits_numba(
            ellipsoid_potential_numba, axes, pos)
        # l_limit, u_limit = get_limits(ellipsoid_potential, axes, pos)

        # Find the 
        root = bisect_numba(l_limit, u_limit, axes, pos)
        # root = bisect(l_limit, u_limit, ellipsoid_potential, axes, pos)
        new_poss[i] = get_surface_position_numba(root, axes, pos)
        # new_poss[i] = get_surface_position(root, axes, pos)
    return new_poss


def write():
    """Writes Snapshot to vtf"""
    custom_data = open(
        "raspberry_ellipsoid_visual" + str(AXES).replace(" ", "_") + ".vtf", mode="w+t"
    )
    vtf.writevsf(system, custom_data)
    vtf.writevcf(system, custom_data)
    print("written to:\n raspberry_ellipsoid_visual" +
          str(AXES).replace(" ", "_") + ".vtf")


def fix_vtk(input_file, forces=None):

    # Input and output file
    output_file = input_file[:-4] + "_fixed.vtk"

    lines = []
    # Read the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Find where POINTS start
    points_start = None
    num_points = None
    for i, line in enumerate(lines):
        if line.startswith("POINTS"):
            points_start = i
            num_points = int(line.split()[1])
            break

    if points_start is None:
        raise ValueError("No POINTS section found in the VTK file.")

    # Determine where points end
    points_end = points_start + 1 + num_points

    # Extract header and points
    header_lines = lines[: points_start + 1]  # include POINTS line
    point_lines = lines[points_start + 1: points_end]

    # Create CELLS section: one vertex per point
    cells_lines = [f"CELLS {num_points} {num_points*2}\n"]
    for i in range(num_points):
        cells_lines.append(f"1 {i}\n")

    # Create CELL_TYPES section: VTK_VERTEX
    cell_types_lines = [f"CELL_TYPES {num_points}\n"]
    for _ in range(num_points):
        cell_types_lines.append("1\n")

    if forces is not None:
        # Add VECTORS section if forces are provided
        forces = np.array(forces)  # make sure it's a numpy array

        if forces.shape[0] != num_points or forces.shape[1] != 3:
            raise ValueError(
                f"Forces array must have shape ({num_points},3). Is:{forces.shape[0], forces.shape[1]}."
            )

        vector_lines = [f"POINT_DATA {num_points}\n", "VECTORS forces float\n"]
        for vec in forces:
            vector_lines.append(f"{vec[0]} {vec[1]} {vec[2]}\n")

    # Combine everything
    new_lines = (
        header_lines + point_lines + cells_lines + cell_types_lines + vector_lines
    )

    # Write to output
    with open(output_file, "w") as f:
        f.writelines(new_lines)

    print(f"Fixed VTK written to {output_file}")


# endregion

#############################################################################################################################
# region: Outer Shell

# Nr of particles for shell, is an upper bound for if the ellipsoid surface was flat 2D and hex packaging.
# There is surely a smaller upper bound, but the particles have to spread all the same. I don't expect a high speedup from a better initial guess.
n_col_part = (
    int((ell_surface_area(AXES) * MAX_SURFACE_FRACTION / ((sig_ss / 2) ** 2 * np.pi)))
)
print("Surface Beads: ", n_col_part)


# Create surface beads uniformly distributed over the surface
for i in range(1, n_col_part + 1):
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

# Relax bead positions.
system.force_cap = 1000
system.time_step = eq_tstep

# Empty folder for recording frames for Paraview
folder_path = Path("data/shell_animation")
for item in folder_path.iterdir():
    if item.is_file():
        item.unlink()

################## Iterating ###################

print(LINE)
print("Relaxation of the raspberry surface particles")
t0 = time.time()

k = 0 # itteration variable
check_nth = 100 # nr of itterations before checking progress

high_score_mean = 0
high_score_smallest = 0
smallest_list = [] # list for plotting progress afterwards
removed_at = [] # records time when particles are removed

no_improvement_counter = 0 # goes up when no new high scores
wait_nr = 5 # target value of no_improvement_counter before removing a particle
mean_flag = True # dictates whether new highdscores in mean count as progress

high_t = True
t_counter = 0 # counts itterations with high temperature
thermal_kick_length = 10 # duration of high temperature state

nth_step_recorded = 25 # record every 25th simulation step

while True:

    system.integrator.run(1)

    # now put all particles back on the surface
    particle_positions = system.part.select(type=1).pos
    new_poss = shift_particle(particle_positions)
    for part, pos in zip(system.part.select(type=1), new_poss):
        part.pos = pos

    # Record frame
    if not k % nth_step_recorded:
        system.part.writevtk(
            f"data/shell_animation/frame_{int(k/nth_step_recorded)}.vtk")
        
    # Early exit condition for trials
    # if k == 100:
    #     break

    k += 1
    if not k % check_nth:

        # Check what to do about thermally exited mode
        if high_t:
            # Cooler system after 2/3 of time
            if t_counter == int(thermal_kick_length/3*2):
                system.thermostat.set_langevin(
                    kT=0.01, gamma=1, seed=seed_pass)
            # Reset to low T domain
            if t_counter == thermal_kick_length:
                system.thermostat.set_langevin(
                    kT=0.0001, gamma=40.0, seed=seed_pass)
                high_t = False
                t_counter = 0
                print(LINE)
                print("THERMAL KICK OVER")
            else:
                t_counter += 1

        mean, std, smallest = mean_std_smallest(system.part.select(type=1).id)

        rel_mean = abs(high_score_mean-mean)/(1-high_score_mean)
        rel_smallest = abs(high_score_smallest-smallest) / \
            (1-high_score_smallest)
        
        # Record history for plot
        if not k % 100:
            smallest_list.append(smallest)

        # Set longer thermal kicks after certain point
        if high_score_mean > 0.98:
            thermal_kick_length = 50

        # TODO THIS SEEMS INEFFECTUAL. Actual forces around 30, dont think high forces are the issue for the final stretch
        # Make progress gentler when close to finished
        # if high_score_smallest > 0.99:
        #     system.force_cap = 50
        #     check_nth = 10
        #     wait_nr = 10
        #     mean_flag = False
        # # Make progress gentler when close to finished
        # if high_score_smallest > 0.999:
        #     system.force_cap = 5
        #     check_nth = 1
        #     wait_nr = 20

        ### Particle Removal ###
        # After smallest distance = 0.99 removing another particle is overkill. Just noise to be settled.
        if (no_improvement_counter == wait_nr and high_score_smallest < 0.99 and high_score_mean < 0.999) or no_improvement_counter == 100:

            # Find particle furthest from center t be removed.
            # Avoids disturbing already formed surface lattices.
            furthest = 0
            for part in system.part:
                if np.linalg.norm(part.pos) > furthest:
                    furthest = np.linalg.norm(part.pos)
                    id_of_furthest = part.id
            system.part.by_id(id_of_furthest).remove()

            # Remove particle
            print(LINE)
            print("PARTICLE REMOVED")
            if not k % 100:
                removed_at.append(k)
            no_improvement_counter = 0

            # Initiatet thermal kick to skip long equilibriation after removal
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
                continue
            # Improved smallest dist
            elif smallest > high_score_smallest:
                high_score_smallest = smallest
                no_improvement_counter = 0
            # Or improved mean dist
            elif mean > high_score_mean and mean_flag:
                high_score_mean = mean
                no_improvement_counter = 0
            # If no improvements and no miniscule ones: register lack of progress
            else:
                print(LINE)
                print("NO PROGRESS")
                no_improvement_counter += 1
                continue

        print("Steps: ", k)
        print("Mean: ", mean)
        print("Std: ", std)
        print("Smallest: ", smallest)
        print("Nr of surface particles: ", len(system.part)-1)

        # Exit condition
        if std < 0.001 and mean > 1 and smallest > 1:
            break

print(LINE)
print("Relaxation steps taken: ", k)
print("Time: ", time.time() - t0)

# Restore time step
system.time_step = time_step

# endregion
smallests_plot = smallest_list[5:]
plt.plot(np.arange(len(smallests_plot)), smallests_plot)
for i in removed_at:
    plt.axvline(x=int(i/100), color='red', linestyle='--', linewidth=1.0)
plt.savefig("data/shell_relaxation.png")


#############################################################################################################################
# region: Virtual sites

# Select the desired implementation for virtual sites
system.virtual_sites = VirtualSitesRelative()
# Setting min_global_cut is necessary when there is no interaction defined with a range larger than
# the colloid such that the virtual particles are able to communicate their forces to the real particleat the center of the colloid
# Needs to be at least max(real.pos - virtual.pos)
system.min_global_cut = radius_col


# Calculate the center of mass position (com) and the moment of inertia (momI) of the colloid
com = np.average(system.part.select(type=1).pos, axis=0)
momIx = 0
momIy = 0
momIz = 0


# TODO is com a point of refference for computation? Is particle 0 physically real?
for i in range(1, n_col_part):
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

# Filling the ellipsoid. It might be way better here to start with a hcp configuration and shaving off any particles that violate ellipsoid equation.
# With some padding to the ellipsoid. Rather too much than too few.
# inner_axes = [i - 1 for i in AXES]
# vol_axes = [i - 1 + (1 - np.sqrt(3) / 2) for i in AXES]
# inner_vol = ellipsoid_volume(vol_axes)
# vol_to_fill = inner_vol * MAX_VOLUME_FRACTION
# n_fill_part = int((vol_to_fill / (4 / 3 * (sig_ss / 2) ** 3 * np.pi)))
# print("# Number of filling beads = {}".format(n_fill_part))

# for i in range(n_col_part + 1, n_col_part + n_fill_part + 1):
#     while True:
#         x = np.random.rand() * inner_axes[0] * np.random.choice([-1, 1])
#         y = np.random.rand() * inner_axes[1] * np.random.choice([-1, 1])
#         z = np.random.rand() * inner_axes[2] * np.random.choice([-1, 1])

#         d = np.sqrt(
#             (x**2.0 / (inner_axes[0] + (1 - np.sqrt(3) / 2)) ** 2.0)
#             + (y**2.0 / (inner_axes[1] + (1 - np.sqrt(3) / 2)) ** 2.0)
#             + (z**2.0 / (inner_axes[2] + (1 - np.sqrt(3) / 2)) ** 2.0)
#         )
#         if d < 1:
#             break

#     vol_pos = np.array([x +box_l/2, y+box_l/2 , z +box_l/2])
#     system.part.add(id=i, pos=vol_pos, type=2)


### Blatant AI filing geometry generation ###
def hex_grid_2d(xmin, xmax, ymin, ymax, spacing):
    """
    Generates 2D hexagonal grid points in XY plane.
    """
    dx = spacing
    dy = np.sqrt(3) * spacing / 2

    points = []

    j = 0
    y = ymin
    while y <= ymax:
        offset = (j % 2) * dx / 2
        x = xmin + offset

        while x <= xmax:
            points.append((x, y))
            x += dx

        y += dy
        j += 1

    return np.array(points)


def hex_ellipsoid_points(a, b, c, spacing = 1):
    """
    Generates hex-like packed points inside ellipsoid:
        x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1
    """

    # bounding box
    xmin, xmax = -a, a
    ymin, ymax = -b, b
    zmin, zmax = -c, c

    xy = hex_grid_2d(xmin, xmax, ymin, ymax, spacing)

    points = []

    # z-layers (simple uniform slicing; can also hex-stagger in 3D if desired)
    dz = np.sqrt(2/3) * spacing  # roughly isotropic spacing

    z = zmin
    while z <= zmax:
        for x, y in xy:
            if (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) <= 1.0:
                points.append((x+box_l/2, y+box_l/2, z+box_l/2))
        z += dz

    return np.array(points)


################################
arr_of_points = hex_ellipsoid_points(AXES[0], AXES[1], AXES[1])

for pos in arr_of_points:
    system.part.add(pos = pos, type = 2)

system.force_cap = 100
system.time_step = eq_tstep

system.part.by_id(0).pos = center
system.part.by_id(0).fix = [1, 1, 1]
system.part.by_id(0).rotation = [0, 0, 0]


print("Relaxation of the raspberry filling particles")
t0 = time.time()
k = 0


# Empty folder for recording frames for Paraview
folder_path = Path("data/inside_animation")
for item in folder_path.iterdir():
    if item.is_file():
        item.unlink()

forces = []
second_stage = False
while True:
    system.integrator.run(1)

    # Disable thermal kick
    if not k% 300:
        system.thermostat.set_langevin(kT=0.000001, gamma=40.0, seed=seed_pass)

    if not k % 3000 and second_stage:
        # Find particle furthest from center t be removed.
        # Avoids disturbing already formed surface lattices.
        furthest = 0
        for part in system.part.select(type = 2):
            if np.linalg.norm(part.pos) > furthest:
                furthest = np.linalg.norm(part.pos)
                id_of_furthest = part.id
        # Remove particle
        system.part.by_id(id_of_furthest).remove()
        print(LINE)
        print("PARTICLE REMOVED")
        # Thermal kick
        system.thermostat.set_langevin(kT=0.1, gamma=1, seed=seed_pass)


    if not k % 25:
        system.part.writevtk(f"data/inside_animation/inside{k}.vtk")
        forces.append(max([np.linalg.norm(part.f) for part in system.part.select(type=2)]))

    if k % 100 == 0:
        if max([np.linalg.norm(part.f) for part in system.part.select(type=2)]) <70 and not second_stage:
            second_stage = True
            
            for part in system.part.select(type = 2):
                x, y, z = part.pos
                x, y, z = [i - j for (i,j) in zip(part.pos, center)]
                a, b, c = AXES

                s = (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c)
                if s > 1:
                    part.remove()
                    k+=25
                    system.part.writevtk(f"data/inside_animation/inside{k}.vtk")


        mean, std, smallest = mean_std_smallest(system.part.select(type=2).id)
        print("Steps: ", k)
        print("Mean: ", mean)
        print("Std: ", std)
        print("Smallest: ", smallest)
        if smallest > 1:
            break
    k += 1

plt.plot(np.arange(len(forces)), forces)
plt.savefig("data/max_forces.png")

print("Relaxation steps taken: ", k)
print("Time: ", time.time() - t0)


# Restore time step
system.time_step = time_step

for p in system.part.select(type=2):
    p.vs_auto_relate_to(0)


# this here shows the total surface on the shell taken up by particle cross-sections. At most should be like 69% or something hexagonal stacking.
# Mean of shell should be > 1 for non overlap. Maybe explicitly check that no distance < 1 even.
# print("surface:", ell_surface_area(axes))
# print("surface of circles on surface: ", n_col_part * np.pi / 4)
# exit()
#######################################################################

system.part.by_id(0).pos = center
system.integrator.run(0)

# endregion

# WRITE OUTPUT

# write coordinates to textfile
file = open("raspberry_coordinates" +
            str(AXES).replace(" ", "_") + ".txt", "w+t")
for x in system.part:
    file.write("{}\t{}\t{}\t{}\n".format(x.pos[0], x.pos[1], x.pos[2], x.type))
file.close()

print(
    "raspberry_coordinates"
    + str(AXES).replace(" ", "_")
    + ".txt contains coordinates of raspberry"
)


print("Reached the end.")
