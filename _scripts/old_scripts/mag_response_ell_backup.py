from functions import writevtk
from functions import rv
from functions import upalpha, uplambda


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


equil_steps = 1000
equil_steps = 0

sim_steps = 1000
sim_steps = 100

vis = True
# vis = False

# Params via parent process
# ratio = float(sys.argv[1])
# Lambda = float(sys.argv[2])
# KV = float(sys.argv[3])
# current_filename = sys.argv[4]

# Params manually configured
ratio = 2.
Lambda = 0.001
KV = 3
current_filename = f"_data/mag_response/manual_r{ratio}_l{Lambda}_KV{KV}.npz"

ratio = float(ratio)
Lambda = float(Lambda)
KV = float(KV)

###################### ------Create an exp spread of alpha values------######################
nr_of_alphas = 20
llim_alphas = 0
ulim_alphas = 25
curvature = 1

lin = np.linspace(llim_alphas, ulim_alphas, nr_of_alphas, dtype=float)
unnorm_alphas = [np.exp(i/ulim_alphas*curvature)-1 for i in lin]
maxv = max(unnorm_alphas)
alphas = [i/maxv*ulim_alphas for i in unnorm_alphas]

# visualize distribution of alphas
# plt.scatter(lin, alphas)
# plt.savefig("plot.png")
# plt.clf()

reverse = np.linspace(ulim_alphas, llim_alphas, nr_of_alphas)
alphas.extend(reverse)
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
system.time_step = 0.0001  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(kT=kT, gamma=1., gamma_rotation=1., seed=42)

filename = f"_data/coordinates/512_ratio_{ratio}_1.0.txt"
filename = f"_data/coordinates/64.0_ratio_{ratio}_1.0.txt"
pos_arr = np.loadtxt(filename)

# Particle setup
for pos in pos_arr:

    # Anisotropy axis particle
    p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
    p1.director = rv()  # easy axis direction
    p1.rotation = (False, False, False)

    p2 = system.part.add(pos=p1.pos, fix=(True, True, True), type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = (m, 0, 0)
    # disable rotations of the virtual site tSW handles this
    p2.rotation = (True, True, True)
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
    prefactor=Lambda, n_replicas=2, gpu=False)
system.magnetostatics.solver = dds

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)

# main loop over field strengths
d_alpha_means = []
d_alpha_stds = []
for i, alpha in enumerate(alphas):

    # set magnetic field constraint
    H = alpha * kT / (mu_0 * m)
    H_field = [0, 0, H]
    H_constraint = espressomd.constraints.HomogeneousMagneticField(
        H=H_field)
    system.constraints.add(H_constraint)

    # Report
    print(LINE)
    print("Running:\n")
    print("lambda = ", Lambda)
    print(LINE)
    print("Ratio: ", ratio)
    print("KV = ", KV)
    print(LINE)
    print("alpha = ", alpha)

    # Empty folder for recording frames for Paraview
    if vis:
        os.makedirs(
            f"_data/vtk_frames/mag_response/", exist_ok=True)
        folder_path = Path(
            f"_data/vtk_frames/mag_response/")
        for item in folder_path.iterdir():
            if item.is_file():
                item.unlink()

    # Equillibriate
    print(LINE)
    print("Equillibriating")
    for i in tqdm.tqdm(range(equil_steps)):
        system.integrator.run(1)

    # Run
    print("Running system")
    dipms = []
    # for i in tqdm.tqdm(range(sim_steps)):
    for alpha in tqdm.tqdm(alphas):
        system.integrator.run(1)

        alpha = alphas[i]
        upalpha(system, alpha, m)

        # write animation frames
        if vis:
            writevtk(
                f"_data/vtk_frames/mag_response/mag{i}.vtk", system, mag=True)

        dipms.append(dipm_tot_z.calculate()[2]/M_s/pos_arr.shape[0])

    plt.plot(np.arange(len(alphas)), dipms)
    plt.savefig("timeplot.png")
    exit()

    # Gather data
    mean = np.mean(dipms)
    d_alpha_means.append(mean)
    std = np.std(dipms)
    d_alpha_stds.append(std)

    # Write to npz
    n_steps = sim_steps
    np.savez(current_filename,
             ratio=ratio, Lambda=Lambda, KV=KV, kT=kT, n_steps=n_steps, alphas=alphas, means=d_alpha_means, stds=d_alpha_stds)

    # Write the filename string to the 'current' file
    with open("current", "w") as f:
        f.write(current_filename)

    # Reset System
    system.constraints.clear()
