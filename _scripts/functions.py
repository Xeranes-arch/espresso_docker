from numba import njit, prange
import numpy as np
from itertools import combinations

def writevtk(path, system, types=None):
    """Custom writevtk that further handles the file made by espresso"""
    # call original function
    if types:
        system.part.writevtk(path, types)
    else:
        system.part.writevtk(path)

    # extra behavior AFTER writing
    with open(path, "r") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        if line.startswith("SCALARS"):
            parts = line.split()
            out.append(f"VECTORS {parts[1]} float\n")
        elif line.startswith("LOOKUP_TABLE"):
            continue
        else:
            out.append(line)

    out.append("\nSCALARS type float 1")
    out.append("\nLOOKUP_TABLE default")
    if types:
        for part in system.part.select(type=types):
            out.append("\n" + str(part.type))
    else:
        for part in system.part:
            out.append("\n" + str(part.type))

    with open(path, "w") as f:
        f.writelines(out)


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


def mean_std_smallest(ids, system):
    "Gives mean, std and smallest distance of next neighbours between particles specified by id lists. Usually in one shell."
    lister = [
        np.linalg.norm(system.part.by_id(i).pos - system.part.by_id(j).pos)
        for i, j in combinations(ids, 2)
        if i != j
    ]

    lister.sort()
    mean = np.mean(lister[: len(ids) - 10])
    std = np.std(lister[: len(ids) - 10])
    smallest = min(lister)
    return mean, std, smallest


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


@njit
def get_surface_position_numba(t, axes, position):
    x_s = position[0] / ((t / axes[0] ** 2.0) + 1)
    y_s = position[1] / ((t / axes[1] ** 2.0) + 1)
    z_s = position[2] / ((t / axes[2] ** 2.0) + 1)
    return np.array([x_s, y_s, z_s])

# switch commented lines for non numba


@njit(parallel=True)
def shift_particle(particle_positions, axes):
    """Moves particle to nearest surface position"""
    new_poss = np.zeros_like(particle_positions)
    for i in prange(len(particle_positions)):
        pos = particle_positions[i]

        # Find region in which the ellipsoid surface is
        l_limit, u_limit = get_limits_numba(
            ellipsoid_potential_numba, axes, pos)

        # Find the exact point within tolerance
        root = bisect_numba(l_limit, u_limit, axes, pos)

        # Reposition particle there
        new_poss[i] = get_surface_position_numba(root, axes, pos)
    return new_poss


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


def hex_ellipsoid_points(a, b, c, box_l, spacing=1.0):
    """
    Generates HCP (ABAB) packed points inside ellipsoid:
        x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1
    """

    dx = spacing
    dy = np.sqrt(3) * spacing / 2
    dz = np.sqrt(2/3) * spacing

    xmin, xmax = -a, a
    ymin, ymax = -b, b
    zmin, zmax = -c, c

    points = []

    shift = (dx / 2, dy / 3)  # B-layer shift in XY

    jz = 0
    z = zmin

    while z <= zmax:

        # decide layer type
        if jz % 2 == 0:
            ox, oy = 0.0, 0.0   # A layer
        else:
            ox, oy = shift      # B layer

        j = 0
        y = ymin

        while y <= ymax:
            x = xmin + (j % 2) * dx / 2 + ox

            while x <= xmax:
                if (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) <= 1.0:
                    points.append((x+box_l/2, y+box_l/2, z+box_l/2))
                x += dx

            y += dy
            j += 1

        z += dz
        jz += 1

    return np.array(points)
