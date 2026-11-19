# SW docker image

from functions import writevtk
from functions import rv
from functions import upalpha, save_to_json, fcc_box_points


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

# # Params via parent process
# ratio = float(sys.argv[1])
# Lambda = float(sys.argv[2])
# # KV = float(sys.argv[3])
# pnr = float(sys.argv[4])
# easy_axis = sys.argv[5]
# steps_per_measurement = int(sys.argv[1])

# ratio = float(ratio)
# Lambda = float(Lambda)
# # KV = float(KV)
# easy_axis = ast.literal_eval(easy_axis)

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
Lambda_calc = 6.033
# Lambda = 3
# Lambda = 1
# Lambda = 0.5
# Lambda = 0.25
Lambda = 0.2

steps_per_measurement = 1
KV = 5
damping_param = 0.08  # rather arbitrary choice by paper?

# Configurable freely

# pnr = 64.0
# pnr = 512.0

# ratio = 2.
# ratio = 6.

easy_axis = [0, 0, 1]

# Doesn't matter for MC tSW steps
time_step = 0.01


# easy_axis = [1, 0, 0]
# region: System setup

###################### ------Constants------######################
# SI
kT = 1
sigma = 1
mass = 1
mu_0 = 1
###################### --Calculated params--######################
m = np.sqrt(Lambda * 4*np.pi * sigma ** 3 * kT/mu_0)
V = np.pi/6 * sigma**3
M_s = m/V
# Anisotropy
H_ani_inv = 1/(2*KV/(mu_0 * m))
# region system setup
##################### -------System-------#####################
nr_per_cell = 2
nr_of_cells = 5
pos_per_cell = 4
nr_of_part = (nr_per_cell * nr_of_cells**3)
nr_of_pos = (pos_per_cell * nr_of_cells**3)
box_l = np.sqrt(2) * nr_of_cells
poss = fcc_box_points(box_l, box_l, box_l)
assert (nr_of_pos == poss.shape[0])
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(
    kT=kT, gamma=1, gamma_rotation=1, seed=42)

V_sys = box_l**3

# For when you want a single central particle (caveat: its easy axis applies to all virt particles (dipms) relating to it)
# Anisotropy axis particle

# Particle setup
for i, pos in enumerate(poss):
    # Anisotropy axis particle
    p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
    p1.director = rv()  # easy axis direction
    p1.rotation = (False, False, False)

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
    prefactor=Lambda, gpu=False, n_replicas=1)
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

###################### ------alpha values------ ######################
alphas = []

llim_alphas = 0
ulim_alphas = 15

set_list = []
set_list.append(np.linspace(llim_alphas, ulim_alphas, int(iterations/5)))
# set_list.append(np.linspace(
#     ulim_alphas, 0, int(iterations/10)).tolist())
# set_list.append(np.linspace(
#     0, 0, 8*int(iterations/10)).tolist())
set_list.append(np.linspace(
    ulim_alphas, -ulim_alphas, 2*int(iterations/5)).tolist())
set_list.append(
    np.linspace(-ulim_alphas, ulim_alphas, 2*int(iterations/5)).tolist())

for i in set_list:
    alphas.extend(i)

upalpha(system, 0, m)
system.integrator.run(10)
########################################################################

frame = 0
dipms = []
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
    upalpha(system, alphas[i], m)
    t2 += time.time()-t0

    # Run
    system.integrator.run(steps_per_measurement)

    t0 = time.time()
    dipms.append(dipm_tot_z.calculate()[2]/M_s/poss.shape[0])
    t1 += time.time()-t0

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

title = f"{nr_of_part} particles, CUBE, KV = {KV}, lambda = {Lambda}, steps = {steps_per_measurement}"
print(title)

json = True
# region: save to json
if json:
    filename = "./_data/mag_response/hyst_data.json"
    entry = {
        "id": 0,
        "Lambda": Lambda,
        "KV": KV,
        "easy_axis": easy_axis,
        "time_step": time_step,
        "steps_per_measurement": steps_per_measurement,
        "iterations": iterations,
        "alphas": alphas,
        "dipms": dipms,
        "ellipsoid": 0,
    }
    save_to_json(filename, entry)
# endregion

report = False
# region: general report
if report:
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))

    ax[0].plot(np.arange(len(alphas)), alphas, label="alpha")

    ax[1].plot(np.arange(len(dipms)), dipms, label="dipm_z")
    # ax[1].plot(np.arange(len(dirs)), dirs, label="dirs_z")
    # ax[1].plot(np.arange(len(alg)), alg, label="alignment")

    ax[1].set_ylim(-1.1, 1.1)

    ax[0].set_title(f"alpha")
    ax[1].set_title(f"dipm orientation vs easy axes direction")

    ax[0].legend()
    ax[1].legend()

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

    plt.xlabel("magnetic field strength")
    plt.ylabel("$\u27E8M_z\u27E9/M_{sat}N$")
    plt.savefig(f"./hysteresis_plots_cube/{title}.png")

# endregion
