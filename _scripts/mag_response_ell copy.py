from functions import writevtk
from functions import rv
from functions import upalpha, uplambda, loop_area


import os
import sys
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import espressomd
import espressomd.observables
import espressomd.propagation
Propagation = espressomd.propagation.Propagation

###################### --------Input--------######################
###################### ---------------------######################

vis = True
# vis = False

# Params via parent process
# ratio = float(sys.argv[1])
# Lambda = float(sys.argv[2])
# KV = float(sys.argv[3])
# current_filename = sys.argv[4]

# Params manually configured
pnr = 64.0
# pnr = 512

ratio = 2.
Lambda = 0.0001
KV = 5
current_filename = f"_data/mag_response/manual_r{ratio}_l{Lambda}_KV{KV}.npz"

ratio = float(ratio)
Lambda = float(Lambda)
KV = float(KV)

# region: fold
###################### ------Constants------######################
###################### ---------------------######################

LINE = "_________________________________\n"
print(LINE)

# SI defining
sigma = 1
kT = 1
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

system = espressomd.System(box_l=[90.0, 90.0, 90.0])
system.time_step = 0.1  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(kT=kT, gamma=1., gamma_rotation=1., seed=42)

filename = f"_data/coordinates/{pnr}_ratio_{ratio}_1.0.txt"
pos_arr = np.loadtxt(filename)

# Particle setup
for pos in pos_arr:
    # Anisotropy axis particle
    p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
    # p1.director = rv()  # easy axis direction
    p1.director = [0, 0, 1]  # easy axis direction
    # p1.rotation = (True, True, True)
    p1.rotation = (False, False, False)

    p2 = system.part.add(pos=p1.pos, fix=(True, True, True), type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = (m, 0, 0)
    p2.rotation = (False, False, False)
    # p2.rotation = (True, True, True)
    # disable rotations of the virtual site tSW handles this
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

###################### ------Create an exp spread of alpha values------######################


sets = []

nr_of_alphas = 20
nr_of_alphas = 1
lim_alphas = 12
curvature = 1

# sets.append(np.linspace(0, lim_alphas, nr_of_alphas).tolist())
# sets.append(np.linspace(lim_alphas, -lim_alphas, 2 * nr_of_alphas).tolist())
# sets.append(np.linspace(-lim_alphas, lim_alphas, 2*nr_of_alphas).tolist())
sets.append([1000 for _ in range(4)])
dur = 100
sets.append(np.linspace(0, 0, dur).tolist())

# region: shedule
# Combine all sets into a single list
schedule = []
current_time = 0

for values in sets:
    for value in values:
        schedule.append([current_time, value])
        current_time += 1

# Print the schedule
for time_step, value in schedule:
    print(f"Time step {time_step}: {value}")
# endregion
##############################################################################################

fig, ax = plt.subplots(figsize=(8, 6))
# main loop over fields
frame_nr = 0
dipms_sets = []
for set in sets:

    dipms = []
    for i, alpha in tqdm.tqdm(enumerate(set), total=len(set)):
        upalpha(system, alpha, m)

        system.integrator.run(1)
        # write animation frames
        if vis:
            writevtk(
                f"_data/vtk_frames/mag_response/mag{frame_nr}.vtk", system, mag=True)
            frame_nr += 1

        dipms.append(dipm_tot_z.calculate()[2]/M_s/pos_arr.shape[0])

    ax.plot(set, dipms)
    dipms_sets.append(dipms)
# Move the left and bottom spines to the center
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust tick positions
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.savefig("timeplot.png")
plt.clf()

prev = 0
for i in dipms_sets:
    plt.plot(np.arange(len(i)) + prev, i)
    prev += len(i) - 1
plt.xlabel("time steps")
plt.ylabel("$\u27E8M_z\u27E9/M_{sat}N$")
plt.savefig("decay.png")

# area = loop_area(dipms_sets, sets[2])
# print(area)
