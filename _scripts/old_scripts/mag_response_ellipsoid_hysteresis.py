from matplotlib.animation import FuncAnimation, PillowWriter
from functions import writevtk
from functions import rv
from functions import upalpha, save_to_json


import os
import sys
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

import espressomd
import espressomd.observables
import espressomd.propagation
Propagation = espressomd.propagation.Propagation

###################### --------Input--------######################
###################### ---------------------######################
RV = False
Z = False

d_mat = 20e-9
kT_mat = 1.3806488e-23 * 298.15
rho_mat = 4.87e3
attempt_time_mat = 1.1e-9

m_mat = rho_mat * d_mat**3*np.pi/6
t_mat = np.sqrt(m_mat*d_mat**2/kT_mat)

# Params via parent process
# Lambda = float(sys.argv[1])
# KV = float(sys.argv[2])
# easy_axes = sys.argv[3]

# ratio = float(ratio)
# Lambda = float(Lambda)
# # KV = float(KV)
# if easy_axes == "Z":
#     Z = True
# else:
#     RV = True

############
Lambda = 4
KV = 5

prefactor_dds = 0.1

# Z = True
RV = True
############
pnr = 512.0
ratio = 6.

# ulim_xis = 60
ulim_xis = 5

assert ((RV or Z) and not (RV and Z))
easy_axis = [0, 0, 1]

id_json = 0
steps_per_measurement = 1
vis = True
# vis = False

###################### ------Sim units------######################
kT = 1
sigma = 1
mass = 1
mu_0 = 4*np.pi
###################### --Calculated params--######################
m = np.sqrt(Lambda * 4*np.pi * sigma ** 3 * kT/mu_0)  # Mag moment
print("mag. moment: ", m)
H_ani_inv = 1/(2*KV/(mu_0 * m))  # Inverse Anisotropy Field
# region: System setup
##################### -------System-------#####################
box_l = 90
time_step = 0.01

system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step  # MD time step in simulation units
system.cell_system.skin = 0.4
system.min_global_cut = 44
system.thermostat.set_langevin(
    kT=kT, gamma=1, gamma_rotation=10, seed=42)

filename = f"_data/coordinates/{pnr}_ratio_{ratio}_1.0.txt"
poss = np.loadtxt(filename)

# For when you want a single central particle (caveat: its easy axis applies to all virt particles (dipms) relating to it)
# Anisotropy axis particle
if Z:
    title_prefix = "Z"
    p1 = system.part.add(pos=[45, 45, 45], fix=(True, True, True), type=0)
    p1.director = easy_axis  # easy axis direction
    p1.rotation = (False, False, False)

# Particle setup
print("kMC time / attempt time ratio: ", time_step * t_mat / attempt_time_mat)
for pos in poss:
    pos = [j + box_l/2 for j in pos]
    # Anisotropy axis particle
    if RV:
        title_prefix = "RV"
        p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
        p1.director = rv()  # easy axis direction
        p1.rotation = (False, False, False)

    p2 = system.part.add(pos=pos, type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = rv()
    # disable rotations of the virtual site tSW handles this (flipping between minima only!)
    p2.rotation = (False, False, False)
    # print(t_mat)
    # print(attempt_time_mat)
    # print(time_step * t_mat / attempt_time_mat)
    p2.magnetodynamics = {
        'is_enabled': True,
        'anisotropy_field_inv': H_ani_inv,  # inverse anisotropy field in reduced units
        'sat_mag': m,  # """saturation magnetisation""" in reduced units
        'anisotropy_energy': KV,  # anisotropy energy K * V in reduced units !!!KV/kT > 5
        # kinetic Monte Carlo time increment [s]
        'sw_dt_incr': time_step * t_mat,
        # inverse attempt time (1/tau_0) [1/s]
        'sw_tau0_inv': 1/attempt_time_mat
    }

    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

# Dipolar Direct Sum for DpDp
dds = espressomd.magnetostatics.DipolarDirectSum(
    prefactor=prefactor_dds, gpu=False)
system.magnetostatics.solver = dds

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)
# endregion

iterations = 100
assert (not iterations % 5)

nr_of_frames = 100
if iterations < nr_of_frames:
    nr_of_frames = iterations

# region: ###################### ------xi values------ ######################
xis = []

set_list = []
set_list.append(np.linspace(0, ulim_xis, int(iterations/5)))
set_list.append(np.linspace(
    ulim_xis, -ulim_xis, 2*int(iterations/5)).tolist())
set_list.append(
    np.linspace(-ulim_xis, ulim_xis, 2*int(iterations/5)).tolist())

for i in set_list:
    xis.extend(i)
# endregion ########################################################################

# MAIN LOOP ###########################
upalpha(system, 0, m)
system.integrator.run(int(1 / time_step))

# region vis folder overwrite
# Empty folder for recording frames for Paraview
if vis:
    os.makedirs(
        f"_data/vtk_frames/mag_response/", exist_ok=True)
    folder_path = Path(
        f"_data/vtk_frames/mag_response/")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
# endregion

t1, t2, t3 = 0, 0, 0
frame = 0
dipms = []
E_ani = []
E_zee = []
E_dip = []
fig, ax = plt.subplots()
c = ["blue", "orange", "green", "red", "pink"]

track_field = []
track_dp = []
track_ext = []
for i in tqdm.tqdm(range(iterations)):
    if not i % 1:
        t0 = time.time()
        dipoles = np.array([
            p.dip for p in system.part if p.type == 1
        ])
        dipoles_normed = [i/np.linalg.norm(i) for i in dipoles]
        dip_field = system.part.select(type=1).dip_fld

        # if i != 0 and not i % 4:
        #     norms_field = np.linalg.norm(dip_field, axis=1)
        #     ax.hist(norms_field, color=c[int(i/4-1)])
        #     ax.axvline(xis[i], color=c[int(i/4-1)])
        # if i == 20:
        #     plt.savefig("hist_of_dip_field_strs.png")
        #     plt.clf()
        #     exit()

        E_ani.append(-KV * np.sum(
                     (dipoles_normed @ np.array(easy_axis))**2
                     ))

        E_zee.append(- xis[i] * np.sum((dipoles_normed @ np.array(easy_axis))))
        # E_zee.append(0)
        E_dip.append(-1/2 * np.sum((dipoles * dip_field)))

        dipms.append(dipm_tot_z.calculate()[2]/m/poss.shape[0])
        t1 += time.time()-t0

        track_field.append(dip_field[351])
        track_dp.append(dipoles[351])
        track_ext.append(xis[i]*np.array(easy_axis))

    t0 = time.time()
    upalpha(system, xis[i], m)
    t2 += time.time()-t0

    t0 = time.time()
    # write animation frame
    if vis and frame/nr_of_frames <= i/iterations:
        writevtk(
            f"_data/vtk_frames/mag_response/mag{frame}.vtk", system, mag=True, easy_axes=True)
        frame += 1
    t3 += time.time()-t0

    # Run
    system.integrator.run(1)

# region: Animation #############


def to_spherical(v):
    x, y = v[0], v[1]
    z = v[2] if len(v) > 2 else 0.0

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)

    return r, theta, phi


def project_vector(v):
    """Project a 3D vector to the clockface, with +z pointing right."""
    x, y = v[0], v[1]
    z = v[2] if len(v) > 2 else 0.0

    # Magnitude and spherical angles
    r, theta, phi = to_spherical(v)

    # Clockface direction: +z right, +x up.
    # The y component is represented through the azimuthal angle.
    if r == 0:
        return 0.0, 0.0

    dx = r * np.cos(phi)
    dy = r * np.sin(phi) * np.cos(theta)

    return dx, dy


# Number of frames shared by all tracks
n_frames = min(len(track_dp), len(track_field), len(track_ext))

# Compute fixed plot limits using all vectors and resultants
all_vectors = []
for i in range(n_frames):
    field_sum = np.asarray(track_field[i]) + np.asarray(track_ext[i])
    all_vectors.extend([
        track_dp[i],
        track_field[i],
        track_ext[i],
        field_sum
    ])

projected = [project_vector(v) for v in all_vectors]
max_r = max(
    max(abs(x), abs(y)) for x, y in projected
)
lim = 2.45 * 1.15

# Figure
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')

ax.axhline(0, color='black', linewidth=0.6)
ax.axvline(0, color='black', linewidth=0.6)
ax.grid(True, linestyle='--', alpha=0.4)
ax.add_patch(plt.Circle(
    (0, 0), lim / 1.15,
    fill=False, linestyle=':', color='gray', alpha=0.5
))

ax.set_xlabel('Projected direction')
ax.set_ylabel('Projected direction')
ax.set_title('Vector Addition Animation')

# --- Stoner-Wohlfarth astroid ---
# Total anisotropy energy [J]
Hk = 2*KV/(mu_0 * m)

# Parametric Stoner-Wohlfarth astroid
t = np.linspace(0, 2 * np.pi, 1000)

H_parallel = Hk * np.cos(t)**3
H_perp = Hk * np.sin(t)**3

# Faint astroid in the background
ax.plot(
    H_parallel,
    H_perp,
    color='gray',
    alpha=0.25,
    linewidth=1.5,
    linestyle='-',
    zorder=0
)


def draw_arrow(start, vector, color, alpha=1.0, label=None):
    dx, dy = project_vector(vector)

    return ax.quiver(
        start[0], start[1], dx, dy,
        angles='xy',
        scale_units='xy',
        scale=1,
        color=color,
        alpha=alpha,
        label=label
    )


def update(i):
    global arrows

    # Remove arrows from previous frame
    for arrow in arrows:
        arrow.remove()

    dp = np.asarray(track_dp[i])
    field = np.asarray(track_field[i])
    ext = np.asarray(track_ext[i])

    # Pad 2D vectors to 3D if needed
    if dp.size == 2:
        dp = np.append(dp, 0.0)
    if field.size == 2:
        field = np.append(field, 0.0)
    if ext.size == 2:
        ext = np.append(ext, 0.0)

    field_sum = field + ext

    # Project the field endpoint to determine where ext should start
    fx, fy = project_vector(field)
    ex, ey = project_vector(ext)

    arrows = [
        # dp from origin
        draw_arrow((0, 0), dp, 'blue', 1.0),

        # field from origin, 50% opacity
        draw_arrow((0, 0), field, 'red', 0.5),

        # ext stacked head-to-tail on field, 50% opacity
        draw_arrow((fx, fy), ext, 'hotpink', 0.5),

        # resultant field + ext from origin
        draw_arrow((0, 0), field_sum, 'purple', 1.0)
    ]

    return arrows


# Initial frame
arrows = []
update(0)

ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=1000,
    blit=False,
    repeat=True
)

ani.save("vecs_clockface.gif", writer=PillowWriter(fps=10))
# endregion ###########################

json = True
# json = False
# region: save to json
if json:
    filename = "./_data/mag_response/hyst_data.json"
    entry = {
        "id": id_json,
        "Lambda": Lambda,
        "KV": KV,
        "pnr": pnr,
        "ratio": ratio,
        "easy_axis": easy_axis,
        "frequency": 1/(time_step * t_mat * iterations),
        "xis": xis,
        "dipms": dipms,
    }
    save_to_json(filename, entry)
# endregion

report = True
# region: general report
if report:
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))

    ax[0].plot(np.arange(len(xis)), xis, label="xi")

    ax[1].plot(np.arange(len(E_ani)), E_ani, label=f"ani {KV}")
    ax[1].plot(np.arange(len(E_dip)), E_dip,
               label=f"dip {Lambda, prefactor_dds}")
    ax[1].plot(np.arange(len(E_zee)), E_zee, label="zee")
    ax[1].plot(np.arange(len(E_zee)), [E_ani[i]+E_zee[i]+E_dip[i]
               for i in range(len(E_zee))], label="total")

    ax[2].plot(np.arange(len(dipms)), dipms, label="dipms")

    ax[0].set_title(f"xi")
    ax[1].set_title(f"Energy")
    ax[2].set_title(f"Dipms")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.savefig(f"plot.png")
    plt.clf()
# endregion

hy = True
# region: plot hysteresis
if hy:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Move the left and bottom spines to the center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust tick positions
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    #######
    nr = len(xis)
    indices = np.linspace(0, len(dipms)-1, nr, dtype=int)
    plot_vals = [dipms[i] for i in indices]

    prev = 0
    for plot_x in set_list:
        ax.plot(plot_x, plot_vals[prev:prev+len(plot_x)])
        prev += len(plot_x)

    plt.xlabel("magnetic field strength")
    plt.ylabel("$\u27E8M_z\u27E9/M_{sat}N$")
    plt.savefig(f"hysteresis.png")

# endregion
