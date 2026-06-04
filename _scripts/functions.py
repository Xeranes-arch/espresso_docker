from numba import njit, prange
import numpy as np
from itertools import combinations


def rq():
    q = np.random.normal(size=4)
    norm = np.linalg.norm(q, keepdims=True)
    return q / norm


def rv():
    q = rq()
    norm = np.linalg.norm(q[1:])
    return q[1:]/norm


def rpos(box_l):
    return np.random.uniform(0.0, box_l, size=3)


def writevtk(path, system, types=None, mag=False):
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

    if mag:
        out.append("\nVECTORS mag float")
        for part in system.part.all():
            v_str = " ".join(map(str, part.dip))
            out.append(f"\n{v_str}")
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
def shift_particle(particle_positions, axes, if_outside=False):
    """
    Moves particle to nearest surface position.
    Only works centered around 0,0,0
    """
    new_poss = np.zeros_like(particle_positions)
    for i in prange(len(particle_positions)):
        pos = particle_positions[i]

        # Find region in which the ellipsoid surface is
        l_limit, u_limit = get_limits_numba(
            ellipsoid_potential_numba, axes, pos)

        # Find the exact point within tolerance
        root = bisect_numba(l_limit, u_limit, axes, pos)

        new_pos_candidate = get_surface_position_numba(root, axes, pos)

        # Reposition particle there
        if if_outside:
            if np.linalg.norm(new_pos_candidate) < np.linalg.norm(pos):
                new_poss[i] = new_pos_candidate
            else:
                new_poss[i] = pos
        else:
            new_poss[i] = new_pos_candidate

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


def hex_ellipsoid_points(a, b, c, spacing=1, offset_from_000=None):
    dx = spacing
    dy = np.sqrt(3) * spacing / 2
    dz = np.sqrt(2/3) * spacing

    # Define a proper 3D translation vector from your parameter
    # If offset_from_000 is a single number, we treat it as an [X, Y, Z] shift
    if offset_from_000 is not None:
        # Match your original logic of shifting by offset/2
        shift_vector = np.array(
            [offset_from_000/2, offset_from_000/2, offset_from_000/2])
    else:
        shift_vector = np.array([0.0, 0.0, 0.0])

    # To ensure we don't clip the edges, expand search bounds by the shift magnitude
    max_shift = int(np.ceil(np.max(np.abs(shift_vector)) / dz))
    z_max_idx = int(np.ceil(c / dz)) + max_shift
    y_max_idx = int(np.ceil(b / dy)) + max_shift
    x_max_idx = int(np.ceil(a / dx)) + max_shift

    points = []

    for iz in range(-z_max_idx, z_max_idx + 1):
        z_grid = iz * dz
        is_b_layer = (iz % 2 != 0)

        for iy in range(-y_max_idx, y_max_idx + 1):
            y_grid = iy * dy + (dy / 3.0 if is_b_layer else 0.0)

            for ix in range(-x_max_idx - 2, x_max_idx + 3):
                row_toggle = (iy % 2 != 0)
                x_shift = (dx / 2.0 if row_toggle else 0.0) + \
                    (dx / 2.0 if is_b_layer else 0.0)
                x_grid = ix * dx + x_shift

                # Apply the spatial translation to get the actual point position
                x = x_grid + shift_vector[0]
                y = y_grid + shift_vector[1]
                z = z_grid + shift_vector[2]

                # CRITICAL: The boundary check must use the actual final coordinates
                # if the ellipsoid itself is sitting at (0,0,0)
                if (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2) <= 1.0:
                    points.append((x, y, z))

    return np.array(points)

# def hex_ellipsoid_points(a, b, c, offset_from_000=None, spacing=1.0):
#     """
#     Generates HCP (ABAB) packed points inside ellipsoid:
#         x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1
#     """

#     dx = spacing
#     dy = np.sqrt(3) * spacing / 2
#     dz = np.sqrt(2/3) * spacing

#     xmin, xmax = -a, a
#     ymin, ymax = -b, b
#     zmin, zmax = -c, c

#     points = []

#     shift = (dx / 2, dy / 3)  # B-layer shift in XY

#     jz = 0
#     z = zmin

#     while z <= zmax:

#         # decide layer type
#         if jz % 2 == 0:
#             ox, oy = 0.0, 0.0   # A layer
#         else:
#             ox, oy = shift      # B layer

#         j = 0
#         y = ymin

#         while y <= ymax:
#             x = xmin + (j % 2) * dx / 2 + ox

#             while x <= xmax:
#                 if (x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c) <= 1.0:
#                     if offset_from_000:
#                         points.append(
#                             (x+offset_from_000/2, y+offset_from_000/2, z+offset_from_000/2))
#                     else:
#                         points.append((x, y, z))

#                 x += dx

#             y += dy
#             j += 1

#         z += dz
#         jz += 1

#     return np.array(points)
