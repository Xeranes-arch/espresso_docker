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

vis = False

ratio = float(sys.argv[1])
ani = float(sys.argv[2])

# ratio = 2
# ani = 3.

################

print("_________________________________")
print(f"Running ratio:{ratio}, ani:{ani}")

kT = 1
mu_0 = 1
alpha = 1

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
    p1.rotation = (False, False, False)  # allow particle rotation

    p2 = system.part.add(pos=p1.pos, fix=(True, True, True))
    # set dipole moment for the virtual particle in reduced units
    p2.dip = (mu_0, 0, 0)
    # disable rotations of the virtual site
    p2.rotation = (True, True, True)
    p2.magnetodynamics = {
        'is_enabled': True,
        # inverse anisotropy field (1/H_k) in reduced units WAS 1/10 of sat_mag... so maybe that? questionable what this does
        'anisotropy_field_inv': 0.175,
        'sat_mag': mu_0,  # saturation magnetisation in reduced units
        'anisotropy_energy': ani,  # anisotropy energy K * V in reduced units
        'sw_dt_incr': 1.0e-10,  # kinetic Monte Carlo time increment [s]
        'sw_tau0_inv': 1.0e9  # inverse attempt time (1/tau_0) [1/s]
    }
    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

alphas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 12.5, 15]
dipms_list = []
for alpha in alphas:
    # set magnetic field constraint
    H_dipm = alpha * kT / mu_0
    H_field = [0, 0, H_dipm]
    H_constraint = espressomd.constraints.HomogeneousMagneticField(H=H_field)
    system.constraints.add(H_constraint)

    dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
        ids=system.part.all().id)

    if vis:
        # Empty folder for recording frames for Paraview
        os.makedirs(
            f"_data/vtk_frames/mag_response/alpha{alpha}", exist_ok=True)
        folder_path = Path(f"_data/vtk_frames/mag_response/alpha{alpha}")
        for item in folder_path.iterdir():
            if item.is_file():
                item.unlink()

    dipms = []
    system.integrator.run(1000)
    for i in tqdm.tqdm(range(1000)):
        system.integrator.run(10)
        if vis:
            writevtk(
                f"_data/vtk_frames/mag_response/alpha{alpha}/mag{i}.vtk", system, mag=True)
        dipms.append(dipm_tot_z.calculate()[2]/mu_0/pos_arr.shape[0])

    dipms_list.append(dipms)
    system.constraints.clear()


dipms = np.array(dipms_list)
stds = np.std(dipms, axis=1)
dipm_means = np.mean(dipms, axis=1)

if vis:
    plt.errorbar(
        alphas,
        dipm_means,
        yerr=stds,
        fmt="-+",  # '+' means plus marker, no character after means NO line
        color="black",  # Classic scientific black
        ecolor="black",  # Error bar color matches
        elinewidth=1,  # Thickness of error bars
        capsize=4,  # Width of the horizontal caps
        capthick=1,  # Thickness of the horizontal caps
        markersize=8,  # Size of the plus marker
        markeredgewidth=1.5,
    )

    plt.xlim(-0.5, 15.2)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Magnetic field strength")
    plt.ylabel("M_z/(mu * N)")
    plt.savefig("target_graph.png")


np.savez(f"_data/mag_response/mag_response_data_ratio{ratio}_ani{ani}.npz",
         ani=ani, alphas=alphas, dipm_means=dipm_means, stds=stds)

# --- How to load it back ---
# loaded = np.load("my_data.npz")
# print(loaded["array1"])  # Access by the key you assigned
# print(loaded["tags"])
