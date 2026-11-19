from scipy.signal import find_peaks
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


def writevtk(path, system, types=None, mag=False, easy_axes=False, half_box_shift=False):
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
            dip = part.dip
            norm = (dip[0]**2 + dip[1]**2 + dip[2]**2)**0.5
            if norm > 0:  # Avoid division by zero
                dip_normalized = [x / norm for x in dip]
            else:
                dip_normalized = [0, 0, 0]  # Keep zero vector as is
            v_str = " ".join(map(str, dip_normalized))
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


def hysteresis_xis(ulim, n):
    assert (not n % 5)
    xis = []
    xis.extend(np.linspace(0, ulim, int(n/5)))
    xis.extend(np.linspace(
        ulim, -ulim, 2*int(n/5)).tolist())
    xis.extend(
        np.linspace(-ulim, ulim, 2*int(n/5)).tolist())
    return xis
