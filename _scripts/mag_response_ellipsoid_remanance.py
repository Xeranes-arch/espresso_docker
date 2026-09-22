# SW docker image

from functions import writevtk
from functions import rv
from functions import upalpha, uplambda, loop_area, append_to_npz, save_to_json


import os
import sys
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ast
import time

import espressomd
import espressomd.observables
import espressomd.propagation
Propagation = espressomd.propagation.Propagation

###################### --------Input--------######################
###################### ---------------------######################

# Params via parent process
ratio = float(sys.argv[1])
Lambda = float(sys.argv[2])
# KV = float(sys.argv[3])
pnr = float(sys.argv[4])
easy_axis = sys.argv[5]
steps_per_measurement = int(sys.argv[6])

ratio = float(ratio)
Lambda = float(Lambda)
# KV = float(KV)
easy_axis = ast.literal_eval(easy_axis)

# region: notes on units
# magnetite
# Energy : 1 = 4,116*10^-21 (room temp)
# Particle size can't get too big otherwise at material 10^4J/m^3, KV would dominate kT. Set KV = 5 -> length scale for single particle emerges
# distance : 1 = 1,578*10^-8
# mass : 1 = 1,06*10^-20kg (density 5170kg/m^3)
# with Ms = 48*10^4 -> lambda = 6,033? and
# Current: 1 = 3,97*10^-3 A
# (time = 1 = 2,53*10^-8s)
# endregion

# Set, don't touch

gyromag_ratio = 18.574
# Lambda = 6.033
# Lambda = 3
# Lambda = 1
# Lambda = 0.5
# Lambda = 0.0001

KV = 5
damping_param = 0.08  # rather arbitrary choice by paper?

# Configurable freely

# pnr = 64.0
# pnr = 512.0

# ratio = 2.
# ratio = 6.

# easy_axis = [0, 0, 1]

time_step = 0.01


# easy_axis = [1, 0, 0]
# region: System setup

###################### ------Constants------######################
###################### ---------------------######################

LINE = "_________________________________\n"
print(LINE)
print("System setup\n")
print(LINE)

# SI
kT = 1
sigma = 1
mass = 1
m = 1
mu_0 = 4 * np.pi

V = np.pi/6 * sigma**3

###################### --Calculated params--######################
###################### ---------------------######################

mu_0 = Lambda/m**2*4*np.pi
M_s = m/V

gamma_rot = 100 * (1+damping_param**2) * m/(damping_param*gyromag_ratio)
# With stokes law friction coefficient
viscosity = gamma_rot / np.pi / sigma**3
gamma = 6 * np.pi * viscosity * sigma / 2 / mass

# Anisotropy
H_ani_inv = 1/(2*KV/(mu_0 * m))

##################### -------System-------#####################
##################### --------------------#####################
box_l = 90
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(
    kT=kT, gamma=gamma, gamma_rotation=gamma_rot, seed=42)

filename = f"_data/coordinates/{pnr}_ratio_{ratio}_1.0.txt"
poss = np.loadtxt(filename)

# For when you want a single central particle (caveat: its easy axis applies to all virt particles (dipms) relating to it)
# Anisotropy axis particle
p1 = system.part.add(pos=[45, 45, 45], fix=(True, True, True), type=0)
# p1.director = rv()  # easy axis direction
p1.director = easy_axis  # easy axis direction
p1.rotation = (False, False, False)

# Particle setup
for i, pos in enumerate(poss):
    pos = [i + box_l/2 for i in pos]

    # # Anisotropy axis particle
    # p1 = system.part.add(pos=pos, fix=(False, False, False), type=0)
    # p1.director = rv()  # easy axis direction
    # p1.rotation = (True, True, True)

    p2 = system.part.add(pos=pos, type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = rv()
    # disable rotations of the virtual site tSW handles this (flipping between minima only!)
    p2.rotation = (False, False, False)
    p2.magnetodynamics = {
        'is_enabled': True,
        # inverse anisotropy field (1/H_k) in reduced units
        'anisotropy_field_inv': H_ani_inv,
        'sat_mag': M_s,  # saturation magnetisation in reduced units
        'anisotropy_energy': KV,  # anisotropy energy K * V in reduced units !!!KV/kT > 3
        'sw_dt_incr': 1.0e-10,  # kinetic Monte Carlo time increment [s]
        'sw_tau0_inv': 1.0e9  # inverse attempt time (1/tau_0) [1/s]
    }
    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

# Dipolar Direct Sum for DpDp
dds = espressomd.magnetostatics.DipolarDirectSum(
    prefactor=Lambda, gpu=False)
system.magnetostatics.solver = dds

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)


# endregion


###################### ------alpha values------ ######################
alphas = []

nr_of_alphas = 100

llim_alphas = 0
ulim_alphas = 150

set_list = []
set_list.append(np.linspace(llim_alphas, ulim_alphas, nr_of_alphas))
set_list.append(np.linspace(
    ulim_alphas, -ulim_alphas, 2*nr_of_alphas).tolist())
set_list.append(
    np.linspace(-ulim_alphas, ulim_alphas, 2*nr_of_alphas).tolist())

for i in set_list:
    alphas.extend(i)

directors = np.array([(1, 0) for i in np.linspace(0, 2*np.pi, len(alphas))])

upalpha(system, 0, m)
system.integrator.run(1)
########################################################################

nr_of_frames = 554

# steps_per_measurement = 2
steps = steps_per_measurement * 1000

iterations = int(steps/steps_per_measurement)
if iterations < nr_of_frames:
    iterations = nr_of_frames
    steps_per_measurement = int(steps/nr_of_frames)

# region: Schedule alpha switches
# Combine all sets into a single list
schedule = []
current_time = 0

x = 0
for frame in range(iterations+1):
    if frame/iterations >= x/(len(alphas)-1):
        schedule.append([frame, alphas[x], directors[x][0], directors[x][1]])
        x += 1
schedule = np.array(schedule)
set_len = schedule[1][0]
# endregion

dipms = []
dirs = []
alg = []
cur_field_dirz = 1
field_zs = []
frame = 0
t1 = 0
t2 = 0
t3 = 0
# MAIN LOOP ###########################

vis = True
vis = False
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

print("With Steps per measurement = ", steps_per_measurement)
for i in tqdm.tqdm(range(iterations)):

    t0 = time.time()
    # Collect data
    dir = system.part.select(type=0).director[0]
    dirs.append(dir[2])
    dipmoments = system.part.select(type=1).dip
    dipmoments = [i / np.linalg.norm(i) for i in dipmoments]
    angles = [np.arccos(np.abs(np.dot(i, dir))) for i in dipmoments]
    alignment = [1 - (angle / (np.pi / 2)) for angle in angles]
    alg.append(np.mean(alignment))
    dipms.append(dipm_tot_z.calculate()[2]/M_s/poss.shape[0])
    field_zs.append(cur_field_dirz)
    t1 += time.time()-t0

    t0 = time.time()
    # Only update alpha according to schedule
    if i in schedule[:, 0]:
        idx = np.where(schedule[:, 0] == i)[0][0]
        cur_field_dirz = schedule[idx, 2]
        upalpha(system, schedule[idx, 1], m,
                field_dir_z=schedule[idx, 2], field_dir_x=schedule[idx, 3])
    t2 += time.time()-t0

    # Run
    system.integrator.run(steps_per_measurement)

    t0 = time.time()
    # write animation frame
    if vis and frame/nr_of_frames <= i/iterations:
        writevtk(
            f"_data/vtk_frames/mag_response/mag{frame}.vtk", system, mag=True, easy_axes=True)
        frame += 1
    t3 += time.time()-t0

print("Timing: ")
print(t1)
print(t2)
print(t3)

json = True
# region: save to json
if json:
    filename = "./_data/mag_response/mag_decay/mag_decay.json"
    entry = {
        "ratio": ratio,
        "Lambda": Lambda,
        "KV": KV,
        "pnr": pnr,
        "easy_axis": easy_axis,
        "time_step": time_step,
        "steps_per_measurement": steps_per_measurement,
        "iterations": iterations,
        "alphas": alphas,
        "dipms": dipms,
    }
    save_to_json(filename, entry)
# endregion

report = True
# region: general report
if report:
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))

    ax[0].plot(np.arange(len(alphas)), alphas, label="alpha")

    ax[1].plot(np.arange(len(dipms)), dipms, label="dipm_z")
    # ax[1].plot(np.arange(len(dirs)), dirs, label="dirs_z")
    ax[1].plot(np.arange(len(alg)), alg, label="alignment")

    ax[1].set_ylim(-1.1, 1.1)

    ax[0].set_title(f"alpha")
    ax[1].set_title(f"dipm orientation vs easy axes direction")

    ax[0].legend()
    ax[1].legend()

    title = f"{pnr} particles, ellipsoid, KV = {KV}, ratio = {ratio}, lambda = {Lambda}, steps = {steps_per_measurement}"
    fig.suptitle(title)

    plt.savefig(f"{title}.png")
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
    # Calculate the time steps orresponding to each alpha value
    time_steps = np.arange(len(dipms)) * steps_per_measurement

    nr = len(alphas)
    indices = np.linspace(0, len(dipms)-1, nr, dtype=int)
    plot_vals = [dipms[i] for i in indices]

    prev = 0
    for plot_x in set_list:
        ax.plot(plot_x, plot_vals[prev:prev+len(plot_x)])
        prev += len(plot_x)

    plt.xlabel("time steps")
    plt.ylabel("$\u27E8M_z\u27E9/M_{sat}N$")
    plt.savefig(f"./hysteresis_plots/hysteresis_{title}.png")

# endregion
