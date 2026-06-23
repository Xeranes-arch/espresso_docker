from functions import writevtk
from functions import rv


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

################

sim_steps = 1000
# sim_steps = 100

equil_steps = 1000
# equil_steps = 100

# vis = True
vis = False

ratio = float(sys.argv[1])
KV = float(sys.argv[2])

# ratio = 2.
# KV = 3

# Creates a nice exp spread of values
# ulim_alphas = 100
ulim_alphas = 50
lin = np.linspace(0, ulim_alphas, 20, dtype=float)
# lin = np.linspace(0, ulim_alphas, 7, dtype=float)

alphas = [np.exp(i/ulim_alphas*5)-1 for i in lin]
maxv = max(alphas)
alphas = [i/maxv*ulim_alphas for i in alphas]

# plt.scatter(lin, alphas)
# plt.savefig("plot.png")
# exit()

# fg
# alphas = np.arange(0, 2, 0.5, dtype=float)
# alphas = np.append(alphas, np.arange(2, 8, dtype=float))
# alphas = np.append(alphas, np.arange(8, 15.1, 1.75, dtype=float))

# fglim
# alphas = np.arange(0, 5.1, 0.25)

# alphas = [100]
# ulim_alphas = max(alphas)

ratio = float(ratio)
KV = float(KV)
################

LINE = "_________________________________"
print(LINE)
print(f"Running ratio:{ratio}, ani:{KV}")

sigma = 1
kT = 1
mass = 1
m = 1

V = np.pi/6 * sigma**3
M_s = m/V

# DPDP
# with
mu_0 = 4 * np.pi
# Lambda = 1 for m = 1
Lambda = mu_0*m**2/(4*np.pi * sigma**3 * kT)

# Zeeman
alpha = 1
H = alpha * kT / (mu_0 * m)

# Anisotropy
# KV = 3
H_ani_inv = 1/(2*KV/(mu_0 * m))

system = espressomd.System(box_l=[90.0, 90.0, 90.0])
system.time_step = 0.001  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(kT=kT, gamma=75., gamma_rotation=25., seed=42)

filename = f"_data/coordinates/512_ratio_{ratio}_1.0.txt"
pos_arr = np.loadtxt(filename)


for pos in pos_arr:

    # One particle with thermal Stoner-Wohlfarth enabled
    p1 = system.part.add(pos=pos, fix=(True, True, True))
    p1.director = rv()  # easy axis direction
    p1.rotation = (False, False, False)  # disallow particle rotation

    p2 = system.part.add(pos=p1.pos, fix=(True, True, True))
    # set dipole moment for the virtual particle in reduced units
    p2.dip = (m, 0, 0)
    # enable rotations of the virtual site
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

# DP3M
accuracy = 5E-4
system.magnetostatics.solver = espressomd.magnetostatics.DipolarP3M(
    accuracy=accuracy, prefactor=Lambda * sigma**3 * kT)

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)

d_alpha_dipms = []
for alpha in alphas:

    # set magnetic field constraint
    H = alpha * kT / (mu_0 * m)
    H_field = [0, 0, H]
    H_constraint = espressomd.constraints.HomogeneousMagneticField(
        H=H_field)
    system.constraints.add(H_constraint)

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
            f"_data/vtk_frames/mag_response/alpha{round(alpha, 2)}", exist_ok=True)
        folder_path = Path(
            f"_data/vtk_frames/mag_response/alpha{round(alpha, 2)}")
        for item in folder_path.iterdir():
            if item.is_file():
                item.unlink()

    # Equillibriate
    print(LINE)
    print("Equillibriating")
    # for i in tqdm.tqdm(range(1000)):
    for i in tqdm.tqdm(range(100)):
        system.integrator.run(1)

    # Run
    print("Running system")
    dipms_list = []
    for i in tqdm.tqdm(range(sim_steps)):
        system.integrator.run(1)
        if vis:
            writevtk(
                f"_data/vtk_frames/mag_response/alpha{round(alpha, 2)}/mag{i}.vtk", system, mag=True)
        dipms_list.append(dipm_tot_z.calculate()[2]/M_s/pos_arr.shape[0])

    # Gather data
    d_alpha_dipms.append(dipms_list)

    # Write to npz
    n_steps = sim_steps
    dipms = np.array(d_alpha_dipms)
    stds = np.std(dipms, axis=1)
    dipm_means = np.mean(dipms, axis=1)
    print("Dipm_mean = ", dipm_means)
    np.savez(f"_data/mag_response/mag_response_data_ratio{ratio}_ani{KV}.npz",
             ratio=ratio, KV=KV, kT=kT, Lambda=Lambda, n_steps=n_steps, alphas=alphas, dipm_means=dipm_means, stds=stds)

    # Reset System
    system.constraints.clear()


plt.errorbar(
    alphas,
    dipm_means,
    yerr=stds,
    fmt="-+",  # '+' means plus marker, no character after means NO line
    # color="black",  # Classic scientific black
    # ecolor="black",  # Error bar color matches
    elinewidth=1,  # Thickness of error bars
    capsize=4,  # Width of the horizontal caps
    capthick=1,  # Thickness of the horizontal caps
    markersize=8,  # Size of the plus marker
    markeredgewidth=1.5,
)

plt.xlim(-1, ulim_alphas + 1)
plt.ylim(-0.1, 1.1)
plt.xlabel(r"Magnetic field strength $\alpha$ in kT")
plt.ylabel(r"$\langle M_z \rangle / M_{sat} N$")
plt.savefig("target_graph.png")
