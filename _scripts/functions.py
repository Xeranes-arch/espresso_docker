# AI. Basically the whole lot of it.

from scipy.signal import find_peaks
from numba import njit, prange
import numpy as np
from itertools import combinations
import espressomd
import os
import json

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random


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


def writevtk(path, system, types=None, mag=False, dir=False, easy_axes=False, half_box_shift=False):
    """Custom writevtk that further handles the file made by espresso"""
    if half_box_shift:
        system.part.all().pos += system.box_l/2
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

    if mag:
        out.append("\nVECTORS mag float")
        for part in system.part.all():
            dir = part.dip
            norm = (dir[0]**2 + dir[1]**2 + dir[2]**2)**0.5
            if norm > 0:  # Avoid division by zero
                dir_normalized = [x / norm for x in dir]
            else:
                dir_normalized = [0, 0, 0]  # Keep zero vector as is
            v_str = " ".join(map(str, dir_normalized))
            out.append(f"\n{v_str}")

    if dir:
        out.append("\nVECTORS dir float")
        for part in system.part.all():
            dir = part.director
            norm = (dir[0]**2 + dir[1]**2 + dir[2]**2)**0.5
            if norm > 0:  # Avoid division by zero
                dir_normalized = [x / norm for x in dir]
            else:
                dir_normalized = [0, 0, 0]  # Keep zero vector as is
            v_str = " ".join(map(str, dir_normalized))
            out.append(f"\n{v_str}")

    if easy_axes:
        out.append("\nVECTORS easy_axes float")
        for part in system.part.all():
            # for part in system.part.all():
            dir = part.director
            norm = (dir[0]**2 + dir[1]**2 + dir[2]**2)**0.5
            if norm > 0:  # Avoid division by zero
                dir_normalized = [x / norm for x in dir]
            else:
                dir_normalized = [0, 0, 0]  # Keep zero vector as is
            v_str = " ".join(map(str, dir_normalized))
            out.append(f"\n{v_str}")

    with open(path, "w") as f:
        f.writelines(out)
    if half_box_shift:
        system.part.all().pos -= system.box_l/2


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


def plot_hist(s, label=""):
    r = random.random()  # Random float between 0 and 1
    g = random.random()
    b = random.random()
    c = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    plt.hist(s, bins=50, color=c, label=label)
    plt.legend()
    plt.savefig("hist.png")
    print("PLOT UPDATED.............................")


def mean_rmsd_smallest(ids, system):
    "Gives mean, rmsd and smallest distance of next neighbours between particles specified by id lists. Usually in one shell."
    lister = [
        np.linalg.norm(system.part.by_id(i).pos - system.part.by_id(j).pos)
        for i, j in combinations(ids, 2)
        if i != j
    ]

    lister.sort()
    mean = np.mean(lister[: len(ids) - 10])
    s = [x for x in lister if (x < (3**(1/2)+1)/2 and x > 1)]
    rmsd = round(np.sqrt(np.mean((np.array(s) - 1) ** 2)), 3)
    smallest = min(lister)

    if smallest > 1:
        plot_hist(s, rmsd)

    return mean, rmsd, smallest


def shuffle_pos(system, AXES):
    n_particles = len(system.part.all())
    spacing = 1
    while True:
        arr_of_points = hex_ellipsoid_points(
            AXES[0], AXES[1], AXES[2], spacing=spacing)

        if (len(arr_of_points) > n_particles):
            break
        spacing -= 0.01
    arr_of_points = arr_of_points[:n_particles]
    # Place particle
    system.part.all().pos = arr_of_points

    if spacing == 1:
        return True
    return False


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


# @njit
# def get_limits_numba(function, axes, position):
#     """
#     Slides a bracket down the function and checks if the sign changes.
#     So whether there's a root inside.
#     """

#     u_limit = 50  # this was 100 before
#     l_limit = u_limit - 0.3  # this was 0.03 before
#     i = 0

#     # Slide down x-vals until root found
#     # print(midpoint)
#     while np.sign(function(u_limit, axes, position)) == np.sign(
#         function(l_limit, axes, position)
#     ):
#         l_limit -= 0.1  # this was 0.01 before
#         u_limit -= 0.1  # this was 0.01 before
#         i += 1
#         if i > 1e4:
#             break
#     return l_limit, u_limit

@njit
def get_limits_numba(axes, position):
    f0 = ellipsoid_potential_numba(0.0, axes, position)

    if f0 > 0.0:
        # Outside ellipsoid.
        l_limit = 0.0
        u_limit = 1.0

        f_u = ellipsoid_potential_numba(u_limit, axes, position)

        while f_u > 0.0:
            u_limit *= 2.0
            f_u = ellipsoid_potential_numba(u_limit, axes, position)

        return l_limit, u_limit

    else:
        # Inside ellipsoid.
        # Root is between -min(a²) and 0.
        min_axis_sq = min(
            axes[0] ** 2,
            axes[1] ** 2,
            axes[2] ** 2,
        )

        l_limit = -min_axis_sq
        u_limit = 0.0

        # Move slightly away from singularity.
        eps = 1e-12 * min_axis_sq
        l_limit += eps

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
        l_limit, u_limit = get_limits_numba(axes, pos)

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


def hex_cube_points(Lx, Ly, Lz, spacing=1.0, offset_from_000=None):
    """
    Generate an HCP lattice by first creating a lattice much larger than
    the requested box, then keeping only points inside:

        0 <= x <= Lx
        0 <= y <= Ly
        0 <= z <= Lz

    This means the requested dimensions are purely a clipping boundary;
    they do not affect the HCP lattice construction itself.
    """

    dx = spacing
    dy = np.sqrt(3.0) * spacing / 2.0
    dz = np.sqrt(2.0 / 3.0) * spacing

    # ------------------------------------------------------------
    # Translation of the lattice
    # ------------------------------------------------------------
    if offset_from_000 is None:
        shift = np.array([0.0, 0.0, 0.0])
    else:
        shift = np.array([
            offset_from_000,
            offset_from_000,
            offset_from_000
        ], dtype=float)

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Generate a LOT more lattice than we actually need.
    #
    # The lattice itself is independent of Lx/Ly/Lz.
    # We simply make a large region around the requested box.
    # ------------------------------------------------------------

    padding = max(Lx, Ly, Lz) + 10.0 * spacing

    x_min = -padding
    x_max = Lx + padding

    y_min = -padding
    y_max = Ly + padding

    z_min = -padding
    z_max = Lz + padding

    # Convert those physical ranges into generous integer ranges.
    ix_min = int(np.floor((x_min - shift[0]) / dx)) - 2
    ix_max = int(np.ceil((x_max - shift[0]) / dx)) + 2

    iy_min = int(np.floor((y_min - shift[1]) / dy)) - 2
    iy_max = int(np.ceil((y_max - shift[1]) / dy)) + 2

    iz_min = int(np.floor((z_min - shift[2]) / dz)) - 2
    iz_max = int(np.ceil((z_max - shift[2]) / dz)) + 2

    points = []

    # ------------------------------------------------------------
    # Build the oversized lattice
    # ------------------------------------------------------------
    for iz in range(iz_min, iz_max + 1):

        is_b_layer = (iz % 2 != 0)

        z_grid = iz * dz

        for iy in range(iy_min, iy_max + 1):

            y_grid = iy * dy

            if is_b_layer:
                y_grid += dy / 3.0

            row_toggle = (iy % 2 != 0)

            x_shift = (
                (dx / 2.0 if row_toggle else 0.0)
                +
                (dx / 2.0 if is_b_layer else 0.0)
            )

            for ix in range(ix_min, ix_max + 1):

                x_grid = ix * dx + x_shift

                # Apply translation
                x = x_grid + shift[0]
                y = y_grid + shift[1]
                z = z_grid + shift[2]

                points.append((x, y, z))

    points = np.asarray(points, dtype=float)

    # ------------------------------------------------------------
    # NOW clip the oversized lattice to the actual box.
    #
    # This is the ONLY place where Lx/Ly/Lz matter.
    # ------------------------------------------------------------
    inside = (
        (points[:, 0] >= 0.0) &
        (points[:, 0] <= Lx) &
        (points[:, 1] >= 0.0) &
        (points[:, 1] <= Ly) &
        (points[:, 2] >= 0.0) &
        (points[:, 2] <= Lz)
    )

    points = points[inside]
    # ------------------------------------------------------------
    # Remove floating-point duplicates, if any.
    # ------------------------------------------------------------
    if len(points) > 0:
        rounded = np.round(points, decimals=12)
        _, unique_idx = np.unique(
            rounded,
            axis=0,
            return_index=True
        )
        points = points[np.sort(unique_idx)]
    # print(points)
    # exit()
    return points


def hex_ellipsoid_points(a, b, c, spacing=1, offset_from_000=None):
    dx = spacing
    dy = np.sqrt(3) * spacing / 2
    dz = np.sqrt(2/3) * spacing

    if offset_from_000 is not None:
        shift_vector = np.array(
            [offset_from_000, offset_from_000, offset_from_000])
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
                x = x_grid
                y = y_grid
                z = z_grid

                if (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2) <= 1.0:
                    points.append(
                        (x + shift_vector[0], y + shift_vector[1], z + shift_vector[2]))

    return np.array(points)


def fcc_box_points(a, b, c, spacing=np.sqrt(2)):
    # FCC lattice vectors (in units of spacing)
    # These define the basis vectors for the FCC lattice
    basis_vectors = np.array([
        [0, 0, 0],    # corner point
        [0.5, 0.5, 0],  # face center
        [0.5, 0, 0.5],  # face center
        [0, 0.5, 0.5]  # face center
    ])

    # Calculate number of unit cells needed in each dimension
    nx = int(np.ceil(a / spacing))
    ny = int(np.ceil(b / spacing))
    nz = int(np.ceil(c / spacing))

    points = []

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Generate all 4 points in the FCC unit cell
                for vector in basis_vectors:
                    x = (ix + vector[0]) * spacing
                    y = (iy + vector[1]) * spacing
                    z = (iz + vector[2]) * spacing

                    # Only include points within the box
                    if (x <= a) and (y <= b) and (z <= c):
                        points.append([x, y, z])

    return np.array(points)


def update_xi(system, xi,  m, field_dir=[1e-7, 1e-7, 1], kT=1):
    # Reset System
    system.constraints.clear()
    # magnetic field strength
    H = xi * kT / m

    H_field = [i * H for i in field_dir]
    H_constraint = espressomd.constraints.HomogeneousMagneticField(
        H=H_field)
    system.constraints.add(H_constraint)
    return H_field


def loop_area(curve1, curve2, set):
    height = np.flip(np.array(curve1)) - np.array(curve2)
    return np.trapezoid(height, set)


def append_to_npz(filepath, **new_data):
    """
    Append new data to an existing .npz file without overwriting.
    If the file doesn't exist, it will be created.
    """
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filepath):
        existing_data = dict(np.load(filepath, allow_pickle=True))

    # Update with new data
    existing_data.update(new_data)

    # Save back to the same file
    np.savez(filepath, **existing_data)
    print(f"Appended data to {filepath}")


def save_to_json(filename, new_entry_data):
    # Load existing data (if file exists)
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    # Check if new entry has an "id" field
    if "id" in new_entry_data:
        # Search for existing entry with matching id
        for key, value in existing_data.items():
            if "id" in value and value["id"] == new_entry_data["id"]:
                # Overwrite existing entry
                existing_data[key] = new_entry_data
                with open(filename, "w") as f:
                    json.dump(existing_data, f, indent=4)
                return key  # Return the existing key

    # If no matching id found, create new entry
    counter = 1
    while f"entry_{counter:03d}" in existing_data:
        counter += 1
    unique_key = f"entry_{counter:03d}"

    # Add the new entry with the unique key
    existing_data[unique_key] = new_entry_data

    # Save back to JSON
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    return unique_key  # Return the key for reference


def dipole_dipole_energy(system, per_particle=False):
    Es = []
    for part in system.part.select(type=1):
        Es.append(- 1/2 * np.dot(part.dip, part.dip_fld))
    if per_particle:
        return Es
    return np.sum(Es)


def anisotropy_energy(m, KV, easy_axis=[0, 0, 1], mu_0=4*np.pi):
    """
    m: magnetic moment vector
    easy_axis: uniaxial easy-axis vector
    K: anisotropy constant (J/m^3)
    V: particle volume (m^3)
    Returns energy in Joules.
    """
    m = np.array(m, dtype=float)
    easy_axis = np.array(easy_axis, dtype=float)

    m_hat = m / np.linalg.norm(m)
    e_hat = easy_axis / np.linalg.norm(easy_axis)

    cos_theta = np.dot(m_hat, e_hat)

    return KV * (1 - cos_theta**2)


def zeeman_energy(system, fld, per_particle=False):
    Es = []
    for part in system.part.select(type=1):
        Es.append(- np.dot(
            np.array(part.dip),
            np.array(fld))
        )
    if per_particle:
        return Es
    return np.sum(Es)


def probability_preditcion(energy_barrier, time_ratio):
    # return 1 - np.exp(-0.97557512 * time_ratio * np.exp(- 0.99757019 * energy_barrier))
    return 1 - np.exp(-0.98791763 * time_ratio * np.exp(- 1.00133747 * energy_barrier))


def stoner_wohlfarth_energy(field, KV, m, easy_axis=np.array([0., 0., 1.]), mu_0=4*np.pi,
                            n_points=360, save=True, title_apendix=""):

    field = np.asarray(field, dtype=float)
    easy_axis = np.asarray(easy_axis, dtype=float)

    # Normalize easy axis
    easy_axis = easy_axis / np.linalg.norm(easy_axis)

    # Construct an orthonormal basis for the plane
    e1 = easy_axis

    field_perp = field - np.dot(field, e1) * e1

    if np.linalg.norm(field_perp) > 1e-12:
        e2 = field_perp / np.linalg.norm(field_perp)
    else:
        reference = np.array([1., 0., 0.])

        if abs(np.dot(reference, e1)) > 0.9:
            reference = np.array([0., 1., 0.])

        e2 = reference - np.dot(reference, e1) * e1
        e2 /= np.linalg.norm(e2)

    # Angular parameterization
    angles = np.linspace(0, 2 * np.pi, n_points)

    magnetization_norm = (
        np.cos(angles)[:, None] * e1
        + np.sin(angles)[:, None] * e2
    )

    # Stoner-Wohlfarth energy
    E_aniso = KV * (
        1 - (magnetization_norm @ easy_axis)**2
    )

    magnetization = m * magnetization_norm
    E_zeeman = -(magnetization @ field)

    energy = E_aniso + E_zeeman

    # Find local maxima and minima
    maxima_idx, _ = find_peaks(energy)
    minima_idx, _ = find_peaks(-energy)

    # Handle periodic boundary conditions (0 and 2π are identical)
    if energy[0] > energy[1] and energy[0] > energy[-2]:
        maxima_idx = np.append(maxima_idx, 0)

    if energy[0] < energy[1] and energy[0] < energy[-2]:
        minima_idx = np.append(minima_idx, 0)

    # Sort minima by energy (lowest first)
    minima_idx = np.unique(minima_idx)
    minima_idx = minima_idx[np.argsort(energy[minima_idx])]

    # Sort maxima by energy (lowest first)
    maxima_idx = np.unique(maxima_idx)
    maxima_idx = maxima_idx[np.argsort(energy[maxima_idx])]

    # Select the lower maximum
    if len(maxima_idx) > 0:
        lower_max_idx = maxima_idx[0]
        E_max = energy[lower_max_idx]
    else:
        lower_max_idx = None
        E_max = None

    # Calculate barriers for each minimum
    barriers = []

    for min_idx in minima_idx[:2]:
        E_min = energy[min_idx]

        if E_max is not None:
            delta_E = E_max - E_min
            barriers.append(delta_E)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(angles, energy, 'b-', linewidth=2, label="total")
    ax.plot(angles, E_aniso, color="orange", linewidth=2, label="Ani")
    ax.plot(angles, E_zeeman, color="green", linewidth=2, label="Local field")

    ax.fill_between(angles, energy, energy.min(),
                    color='blue', alpha=0.1)

    # Plot the lower maximum
    if lower_max_idx is not None:
        ax.axhline(E_max, color='red', linewidth=0.8, linestyle='--')
        ax.plot(angles[lower_max_idx], E_max, 'ro', markersize=4)

    # Plot one or two minima and their barriers
    for i, min_idx in enumerate(minima_idx[:2]):
        E_min = energy[min_idx]

        ax.axhline(E_min, color='red', linewidth=0.8, linestyle=':')
        ax.plot(angles[min_idx], E_min, 'ro', markersize=4)

        if E_max is not None:
            ax.annotate(
                rf'$\Delta E_{i+1}={barriers[i]:.3e}$',
                xy=(0.98, 0.95 - i * 0.07),
                xycoords='axes fraction',
                ha='right',
                color='red'
            )

    ax.set_xlabel('Angle in plane (radians)')
    ax.set_ylabel('Energy')
    ax.set_title('Stoner-Wohlfarth Energy Landscape snapshot')

    ax.set_xticks(
        [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        ['0', 'π/2', 'π', '3π/2', '2π']
    )

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig("SW_Energy_Snapshot" + title_apendix + ".png", dpi=300)

    return angles, energy, magnetization, minima_idx, lower_max_idx, barriers


def stoner_wohlfarth_energy_animation(
    fields, KV, m,
    axes,
    easy_axis=np.array([0., 0., 1.]),
    n_points=360,
    title=""
):

    results = []
    for ax in axes:
        for scatter in ax.collections:
            old_alpha = scatter.get_alpha()
            scatter.set_alpha(
                0.5 if old_alpha is None else old_alpha * 0.5)
    for i, (field, ax) in enumerate(zip(fields, axes)):

        field = np.asarray(field, dtype=float)
        easy_axis = np.asarray(easy_axis, dtype=float)
        easy_axis = easy_axis / np.linalg.norm(easy_axis)

        # Construct orthonormal basis
        e1 = easy_axis

        field_perp = field - np.dot(field, e1) * e1

        if np.linalg.norm(field_perp) > 1e-12:
            e2 = field_perp / np.linalg.norm(field_perp)
        else:
            reference = np.array([1., 0., 0.])

            if abs(np.dot(reference, e1)) > 0.9:
                reference = np.array([0., 1., 0.])

            e2 = reference - np.dot(reference, e1) * e1
            e2 /= np.linalg.norm(e2)

        # Angular parameterization
        angles = np.linspace(0, 2*np.pi, n_points)

        magnetization_norm = (
            np.cos(angles)[:, None] * e1
            + np.sin(angles)[:, None] * e2
        )

        # Energy
        E_aniso = KV * (
            1 - (magnetization_norm @ easy_axis)**2
        )

        magnetization = m * magnetization_norm
        E_zeeman = -(magnetization @ field)

        energy = E_aniso + E_zeeman

        # Find extrema
        maxima_idx, _ = find_peaks(energy)
        minima_idx, _ = find_peaks(-energy)

        # Periodic boundary conditions
        if energy[0] > energy[1] and energy[0] > energy[-2]:
            maxima_idx = np.append(maxima_idx, 0)

        if energy[0] < energy[1] and energy[0] < energy[-2]:
            minima_idx = np.append(minima_idx, 0)

        minima_idx = np.unique(minima_idx)
        maxima_idx = np.unique(maxima_idx)

        minima_idx = minima_idx[np.argsort(energy[minima_idx])]
        maxima_idx = maxima_idx[np.argsort(energy[maxima_idx])]

        lower_max_idx = maxima_idx[0] if len(maxima_idx) else None

        E_max = energy[lower_max_idx] if lower_max_idx is not None else None

        barriers = []

        for min_idx in minima_idx[:2]:
            if E_max is not None:
                barriers.append(E_max - energy[min_idx])

        # Plot
        # Keep scatter plots, remove everything else
        for artist in ax.get_children():
            if artist not in ax.collections:
                try:
                    artist.remove()
                except (AttributeError, NotImplementedError):
                    pass

        ax.plot(angles, energy, 'b-', linewidth=2, label="Total")
        ax.plot(angles, E_aniso, color="orange",
                linewidth=2, label="Anisotropy")
        ax.plot(angles, E_zeeman, color="green",
                linewidth=2, label="Local field")

        ax.fill_between(
            angles, energy, energy.min(),
            color='blue', alpha=0.1
        )

        if lower_max_idx is not None:
            ax.axhline(E_max, color='red', linewidth=0.8, linestyle='--')
            ax.plot(angles[lower_max_idx], E_max, 'ro', markersize=4)

        for j, min_idx in enumerate(minima_idx[:2]):
            E_min = energy[min_idx]

            ax.axhline(E_min, color='red', linewidth=0.8, linestyle=':')
            ax.plot(angles[min_idx], E_min, 'ro', markersize=4)

            ax.annotate(
                rf'$\Delta E_{j+1}={barriers[j]:.3e}$',
                xy=(0.98, 0.95 - j * 0.07),
                xycoords='axes fraction',
                ha='right',
                color='red'
            )

        ax.set_xlabel('Angle in plane (radians)')
        ax.set_ylabel('Energy')
        ax.set_title(f'Particle {i+1}')
        ax.set_xticks(
            [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
            ['0', 'π/2', 'π', '3π/2', '2π']
        )

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        results.append(
            (angles, energy, magnetization, minima_idx,
             lower_max_idx, barriers)
        )

    return results


def plot_dipole_directions(fields, axes, dipole1, dipole2,
                           easy_axis=np.array([0., 0., 1.])):

    easy_axis = np.asarray(easy_axis, dtype=float)
    easy_axis /= np.linalg.norm(easy_axis)

    e1 = easy_axis

    for field, ax in zip(fields, axes):

        field = np.asarray(field, dtype=float)

        # Construct the same in-plane basis as in the energy function
        field_perp = field - np.dot(field, e1) * e1

        if np.linalg.norm(field_perp) > 1e-12:
            e2 = field_perp / np.linalg.norm(field_perp)
        else:
            reference = np.array([1., 0., 0.])

            if abs(np.dot(reference, e1)) > 0.9:
                reference = np.array([0., 1., 0.])

            e2 = reference - np.dot(reference, e1) * e1
            e2 /= np.linalg.norm(e2)

        # Project dipole vectors onto the plotting plane
        for k, dipole in enumerate([dipole1, dipole2]):

            dipole = np.asarray(dipole, dtype=float)

            x = np.dot(dipole, e1)
            y = np.dot(dipole, e2)

            angle = np.arctan2(y, x) % (2 * np.pi)

            ax.plot(
                angle, 0,
                marker='x',
                color='purple',
                markersize=10,
                markeredgewidth=2,
                linestyle='None',
                label=f'Dipole {k+1}' if ax == axes[0] else None
            )


def hysteresis_xis(ulim, n):
    assert (not n % 5)
    xis = []
    xis.extend(np.linspace(0, ulim, int(n/5)))
    xis.extend(np.linspace(
        ulim, -ulim, 2*int(n/5)).tolist())
    xis.extend(
        np.linspace(-ulim, ulim, 2*int(n/5)).tolist())
    return xis


def hysteresis_xis_sin(ulim, n):
    assert not (n-1) % 5
    t = np.linspace(0, 1.25, n)
    xis = ulim * np.sin(2 * np.pi * t)
    return xis


def animate(fields, track_dps, m, KV, easy_axis, Lambda, xi, apdx=""):
    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    H_max = max(
        np.max(np.linalg.norm(fields[0], axis=1)),
        np.max(np.linalg.norm(fields[1], axis=1))
    )

    E_min = -m * H_max
    E_max = KV + m * H_max

    margin = 0.1 * (E_max - E_min)

    for ax in axes:
        ax.set_ylim(E_min - margin, E_max + margin)

    def update(frame):
        global energy_results

        idx = frame // 2

        if frame % 2 == 1:

            energy_results = stoner_wohlfarth_energy_animation(
                fields=[fields[0][idx], fields[1][idx]],
                KV=KV,
                m=m,
                axes=axes,
                easy_axis=easy_axis,
                title="Stoner-Wohlfarth Energy Landscape"
            )

        elif frame % 2 == 0 and not idx == 0:
            # Plot marker at the corresponding minimum
            for i, ax in enumerate(axes):

                angles, energy, magnetization, minima_idx, lower_max_idx, barriers = energy_results[
                    i]

                # Current dipole direction
                dp = np.asarray(track_dps[i][idx])

                # Reconstruct the plotting plane
                e1 = easy_axis / np.linalg.norm(easy_axis)

                field = np.asarray(fields[i][idx])
                field_perp = field - np.dot(field, e1) * e1

                if np.linalg.norm(field_perp) > 1e-12:
                    e2 = field_perp / np.linalg.norm(field_perp)
                else:
                    reference = np.array([1., 0., 0.])
                    if abs(np.dot(reference, e1)) > 0.9:
                        reference = np.array([0., 1., 0.])

                    e2 = reference - np.dot(reference, e1) * e1
                    e2 /= np.linalg.norm(e2)

                # Calculate dipole angle in the plotting plane
                dp_angle = np.arctan2(
                    np.dot(dp, e2),
                    np.dot(dp, e1)
                ) % (2 * np.pi)

                # Find closest energy minimum (circular angular distance)
                min_angles = angles[minima_idx]

                angular_distances = np.abs(
                    np.angle(np.exp(1j * (min_angles - dp_angle)))
                )

                closest_min = minima_idx[np.argmin(angular_distances)]

                ax.scatter(
                    angles[closest_min],
                    energy[closest_min],
                    color='purple',
                    marker='x',
                    s=80,
                    linewidths=2,
                    zorder=10
                )
        fig.suptitle(
            f"KV = {KV}, \u03BB = {Lambda}, \u03BE = {xi}")
        return [artist for ax in axes for artist in ax.get_children() if isinstance(artist, matplotlib.collections.PathCollection)]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(fields[0])*2,
        interval=100,       # milliseconds between frames
        repeat=False,
        blit=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ani.save(
        f"SW_Energy_Animation{apdx}.mp4",
        writer="ffmpeg",
        fps=2,
        dpi=200
    )


def detect_oscillation(v0, v1, v2):

    # Direction reversal 90deg switch
    if np.dot(v1, v0) < 0 and np.dot(v1, v2) < 0:  # 0.x for higher min angle
        return True
    return False


@njit
def get_distance_to_ellipsoid_surface(pos, axes):
    """
    Approximate radial distance from pos to the ellipsoid surface.
    Assumes an ellipsoid centered at (0,0,0).
    """
    x, y, z = pos
    a, b, c = axes

    r = np.linalg.norm(pos)

    if r < 1e-12:
        return min(a, b, c)

    direction = pos / r

    surface_radius = 1.0 / np.sqrt(
        (direction[0] / a)**2 +
        (direction[1] / b)**2 +
        (direction[2] / c)**2
    )

    return surface_radius - r


@njit
def get_particle_closest_to_boundary(
    particle_positions, axes, margin
):
    best_distance = np.inf
    best_id = -1

    for i in range(len(particle_positions)):
        pos = particle_positions[i]

        l_limit, u_limit = get_limits_numba(axes, pos)
        root = bisect_numba(l_limit, u_limit, axes, pos)

        surface_pos = get_surface_position_numba(root, axes, pos)

        distance = np.linalg.norm(surface_pos - pos)

        if distance >= margin and distance < best_distance:
            best_distance = distance
            best_id = i

    return best_id


@njit(parallel=True)
def count_particles_in_shell(particle_positions, axes, margin=0.2):
    """
    Counts particles within `margin` of the ellipsoid surface.

    Only counts particles inside the ellipsoid.
    """
    count = 0

    for i in prange(len(particle_positions)):
        pos = particle_positions[i]

        l_limit, u_limit = get_limits_numba(axes, pos)
        root = bisect_numba(l_limit, u_limit, axes, pos)

        surface_pos = get_surface_position_numba(root, axes, pos)

        distance = np.linalg.norm(surface_pos - pos)

        if 0 <= distance <= margin:
            count += 1

    return count


@njit
def detect_overlap_ids(
    particle_positions, axes,
    min_distance=1.0, margin=0.2
):
    """
    Returns IDs of particles that overlap with another particle
    within the shell region.

    Particles outside the shell region are ignored entirely.
    """
    n = len(particle_positions)
    min_dist_sq = min_distance**2

    # Identify shell particles
    in_shell = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        pos = particle_positions[i]

        l_limit, u_limit = get_limits_numba(axes, pos)
        root = bisect_numba(l_limit, u_limit, axes, pos)
        surface_pos = get_surface_position_numba(root, axes, pos)

        distance = np.linalg.norm(surface_pos - pos)

        if 0 <= distance <= margin:
            in_shell[i] = True

    # To colour all overlaps
    in_shell[:] = True
    # Check overlaps exclusively between shell particles
    overlapping = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if not in_shell[i]:
            continue

        for j in range(i + 1, n):
            if not in_shell[j]:
                continue

            dx = particle_positions[i, 0] - particle_positions[j, 0]
            dy = particle_positions[i, 1] - particle_positions[j, 1]
            dz = particle_positions[i, 2] - particle_positions[j, 2]

            if dx*dx + dy*dy + dz*dz < min_dist_sq:
                overlapping[i] = True
                overlapping[j] = True

    return np.arange(len(particle_positions))[overlapping]


@njit
def particles_not_in_shell(
    particle_positions, ids_to_check,
    axes, margin=0.2
):
    not_in_shell = 0

    for target_id in ids_to_check:
        found = False

        for i in range(len(particle_positions)):
            if i == target_id:
                found = True
                pos = particle_positions[i]
                break

        if not found:
            not_in_shell += 1
            continue

        l_limit, u_limit = get_limits_numba(axes, pos)
        root = bisect_numba(l_limit, u_limit, axes, pos)
        surface_pos = get_surface_position_numba(root, axes, pos)

        distance = np.linalg.norm(surface_pos - pos)

        if distance < 0 or distance > margin:
            not_in_shell += 1

    return not_in_shell
