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
# ratio = float(sys.argv[1])
# Lambda = float(sys.argv[2])
# KV = float(sys.argv[3])
# pnr = float(sys.argv[4])
# easy_axis = sys.argv[5]

# easy_axis = ast.literal_eval(easy_axis)

# region: notes on units
# magnetite
# Energy = kbT = 1 = 4,116*10^-21 (room temp)
# distance = 1 =  7,44*10^-9 (from K = 10^4J/m^3)
# mass = 1 = 2,13*10^-21kg (density 5170kg/m^3)
# magnetics fixing I_0? I dunno man
# time = 1 = 5,35*10^-9s
# endregion

# Params manually configured
# pnr = 64.0
pnr = 512.0
particle_nr_or = 64
packing_fraction = 0.001

ratio = 2.
Lambda = 0.25
Lambda = 3
KV = 5
# magnetite:
M_s = 48e4

ratio = float(ratio)
Lambda = float(Lambda)
KV = float(KV)

easy_axis = [1, 0, 0]

time_step = 0.001
gamma_rot = 5

vis = True
vis = False

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
mu_0 = 4 * np.pi

V = np.pi/6 * sigma**3

###################### --Calculated params--######################
###################### ---------------------######################

# DpDp
m = round(np.sqrt(Lambda*4*np.pi*sigma**3*kT/mu_0), 2)
M_s = m/V

# Zeeman
alpha = 1
H = alpha * kT / (mu_0 * m)

# Anisotropy
H_ani_inv = 1/(2*KV/(mu_0 * m))

##################### -------System-------#####################
##################### --------------------#####################
box_l = (pnr*np.pi/6 * pnr / 0.001)**(1/3)
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(
    kT=kT, gamma=5, gamma_rotation=gamma_rot, seed=42)

filename = f"_data/coordinates/{pnr}_ratio_{ratio}_1.0.txt"
poss = np.loadtxt(filename)

# # For when you want a single central particle (caveat: its easy axis applies to all virt particles (dipms) relating to it)
# # Anisotropy axis particle
# p1 = system.part.add(pos=[45, 45, 45], fix=(True, True, True), type=0)
# # p1.director = rv()  # easy axis direction
# p1.director = easy_axis  # easy axis direction
# p1.rotation = (True, True, True)
# # p1.rotation = (False, False, False)

# poss = [[0, 0, 0], [0, 0, 1]]
# poss = [[0, 0, 0], [1, 0, 0]]
# poss = np.array(poss)
# Particle setup
for i, pos in enumerate(poss):
    pos = [i + box_l/2 for i in pos]
    # Anisotropy axis particle
    p1 = system.part.add(pos=pos, fix=(False, False, False), type=0)
    p1.director = rv()  # easy axis direction
    p1.rotation = (True, True, True)

    p2 = system.part.add(pos=pos, type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = (m, 0, 0)
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

# system.non_bonded_inter[0, 0].lennard_jones.set_params(
#     epsilon=1, sigma=1, cutoff=1, shift=1)

# Dipolar Direct Sum for DpDp
# dds = espressomd.magnetostatics.DipolarDirectSum(
#     prefactor=Lambda, gpu=False)
# system.magnetostatics.solver = dds

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)
forces = espressomd.observables.ParticleForces(
    ids=system.part.all().id)

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


###################### ------alpha values------ ######################
alphas = []

nr_of_alphas = 13
llim_alphas = 0.1
ulim_alphas = 10
curvature = 4

lin = np.linspace(llim_alphas, ulim_alphas, nr_of_alphas, dtype=float)
unnorm_alphas = [np.exp(i/ulim_alphas*curvature)-1 for i in lin]
maxv = max(unnorm_alphas)
alphas = [i/maxv*ulim_alphas for i in unnorm_alphas]

# visualize distribution of alphas
# plt.scatter(lin, alphas)
# plt.savefig("plot.png")
# plt.clf()
# exit()

nr_of_alphas = len(alphas)
u_lim_alpha = 5

# Hysteresis loop
# lim_alphas = 12
# sets.append(np.linspace(0, lim_alphas, nr_of_alphas).tolist())
# sets.append(np.linspace(lim_alphas, -lim_alphas, 2 * nr_of_alphas).tolist())
# sets.append(np.linspace(-lim_alphas, lim_alphas, 2*nr_of_alphas).tolist())

# alphas.extend(np.linspace(0, u_lim_alpha, nr_of_alphas).tolist())

upalpha(system, alphas[0], m)
########################################################################

nr_of_frames = 600

steps_per_measurement = 10
steps = steps_per_measurement * 10000

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
    if frame/iterations >= x/(nr_of_alphas-1):
        schedule.append([frame, alphas[x]])
        x += 1
schedule = np.array(schedule)
# endregion

set_len = schedule[1][0]
print("Set length: ", set_len)

dipms = []
dipmsx = []
dirs = []
dirsx = []
alg = []
frame = 0
t1 = 0
t2 = 0
t3 = 0
# MAIN LOOP ###########################
for i in tqdm.tqdm(range(iterations)):

    t0 = time.time()
    # Collect data
    # directors = system.part.select(type=0).director
    # dipmoments = system.part.select(type=1).dip
    # directors = [i / np.linalg.norm(i) for i in directors]
    # dipmoments = [i / np.linalg.norm(i) for i in dipmoments]
    # alignment = [np.abs(np.dot(i, j))
    #              for i, j in zip(directors, dipmoments)]
    # alg.append(np.mean(alignment))

    dipms.append(dipm_tot_z.calculate()[2]/M_s/poss.shape[0])

    t1 += time.time()-t0

    t0 = time.time()
    # Only update alpha according to schedule
    if i in schedule[:, 0]:
        print(i)
        idx = np.where(schedule[:, 0] == i)[0][0]
        upalpha(system, schedule[idx, 1], m)
    t2 += time.time()-t0

    # Run
    system.integrator.run(steps_per_measurement)

    # t0 = time.time()
    # # write animation frame
    # if vis and frame/nr_of_frames <= i/iterations:
    #     writevtk(
    #         f"_data/vtk_frames/mag_response/mag{frame}.vtk", system, mag=True, easy_axes=True)
    #     frame += 1
    # t3 += time.time()-t0

print("Timing: ")
print(t1)
print(t2)
print(t3)

fig, ax = plt.subplots(1, 2, figsize=(16, 9))

ax[0].plot(np.arange(len(alphas)), alphas, label="alpha")

ax[1].plot(np.arange(len(dipms)), dipms, label="dipm_z")
ax[1].plot(np.arange(len(dipmsx)), dipmsx, label="dipm_x")
ax[1].plot(np.arange(len(dirs)), dirs, label="dir_z")
ax[1].plot(np.arange(len(dirsx)), dirsx, label="dir_x")
ax[1].plot(np.arange(len(alg)), alg, label="alignment")

ax[1].set_ylim(-1.1, 1.1)

ax[0].set_title(f"alpha")
ax[1].set_title(f"dipm orientation vs easy axes direction")

ax[0].legend()
ax[1].legend()

title = f"{particle_nr_or} particles, ellipsoid, KV = {KV}, lambda = {Lambda}"
fig.suptitle(title)

fig.text(
    # (x, y) position (0.5 = center horizontally, 0.01 = near bottom)
    0.5, 0.01,
    "Notes:\n- lambda = 2 causes demag \n- Now why would the easy axes actively disalign???",
    ha="center",
    fontsize=10,
    bbox=dict(facecolor="lightgray", alpha=0.5)
)
fig.text(
    0.01, 0.01,
    f"gamma_rot = {gamma_rot}",
    fontsize=10,
    bbox=dict(facecolor="lightgray", alpha=0.5)
)

plt.savefig(f"{title}.png")
plt.clf()


prev = 0
x = int(len(dipms)/len(alphas))
for i in range(len(alphas)):
    val = np.average(dipms[prev+100:int(prev+x)])
    plt.scatter(alphas[i], val, c="purple", marker="o")
    prev += int(x)

plt.savefig("mag.png")


exit()

# # Move the left and bottom spines to the center
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')

# # Hide the top and right spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Adjust tick positions
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# prev = 0
# for i in dipms_sets:
#     plt.plot(np.arange(len(i)) + prev, i)
#     prev += len(i) - 1
# plt.xlabel("time steps")
# plt.ylabel("$\u27E8M_z\u27E9/M_{sat}N$")
# plt.savefig("decay.png")

# area = loop_area(dipms_sets[0], dipms_sets[1], sets[1])
# areas.append(area)

# plt.plot(steps, areas)
# plt.savefig("hyst_area_convergence.png")

# filename = "./_data/mag_response/hysteresis_area_convergence/hyst_data.json"

# entry = {
#     "ratio": ratio,
#     "Lambda": Lambda,
#     "KV": KV,
#     "pnr": pnr,
#     "easy_axis": easy_axis,
#     "steps": steps,
#     "areas": areas,
# }
# save_to_json(filename, entry)
