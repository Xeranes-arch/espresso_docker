# region: imports ################################################################
# from functions import *
from functions import *

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
# endregion

# region: Units ################################################################
np.random.seed(42)
LINE = "\n__________________________"

# Material values
d_mat = 20e-9
kT_mat = 1.3806488e-23 * 298.15
rho_mat = 4.87e3
attempt_time_mat = 1.1e-9
# Sim units
kT = 1
sigma = 1
mass = 1
mu_0 = 4*np.pi
# Scales
length_scale = d_mat / sigma
mass_scale = rho_mat / (mass * np.pi/6*sigma**3) * length_scale**3
energy_scale = kT_mat / kT
time_scale = np.sqrt(mass_scale*length_scale**2/energy_scale)
# endregion

# region: Parameters ################################################################
# Lambda = 6
Lambda = float(sys.argv[1])  # for running from CUE
prefactor_dds = mu_0/(4*np.pi)

# KV = 8
KV = float(sys.argv[2])  # for running from CUE
easy_axis = [0, 0, 1]

ulim_xis = 15
field_dir = [0, 0, 1]

# Sets total iteration steps
frequency = 1e7
# Is ratio for tSW and gives time step
kMC_timestep_ratio = 0.1
# Simulated steps per value of xi
step = 11


vis = True  # records vtk frames, doesn't eat much time, ~1/10
# vis = False  # you need to set folder to save to if you change this

###################### --Calculated params-- ######################

# Time
time_step = kMC_timestep_ratio * (attempt_time_mat/time_scale)
loop_time_sim = (1/frequency)/time_scale
iterations = int(loop_time_sim / (time_step * step))
# iterations = int(loop_time_sim / (time_step * step) / 4 * 5)
iterations += 5-iterations % 5 + 1

# Mag
m = np.sqrt(4*np.pi/mu_0 * Lambda * sigma ** 3 * kT)  # Mag moment
print("Mag. moment: ", m)
H_ani_inv = 1/(2*KV/m)  # Inverse Anisotropy Field

# avoids numerical instabilities that can happen when you give a 0 value in any direction
easy_axis = 1e-7 + np.array(easy_axis)
field_dir = 1e-7 + np.array(field_dir)
# normalize
easy_axis = easy_axis/np.linalg.norm(easy_axis)
field_dir = field_dir/np.linalg.norm(field_dir)

# endregion

# region: System setup ################################################################
box_l = 90

system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step
system.cell_system.skin = 0.4
system.min_global_cut = 20  # has to exceed smallest virt distance
system.thermostat.set_langevin(
    kT=kT, gamma=1, gamma_rotation=1, seed=42)  # doesn't apply really but necessary

# Particle coordinates
filename = f"_data/coordinates/512.0_ratio_6.0_1.0.txt"
poss = np.loadtxt(filename)

# Easy axis Particle - applies to all others
p1 = system.part.add(pos=[box_l/2, box_l/2, box_l/2],
                     fix=(True, True, True), type=0)
p1.director = easy_axis
p1.rotation = (False, False, False)

# Particles carrying magnetic moments
for pos in poss:
    pos = [j + box_l/2 for j in pos]  # centering

    p2 = system.part.add(pos=pos, type=1)
    p2.dip = rv()
    p2.rotation = (False, False, False)
    p2.magnetodynamics = {
        'is_enabled': True,
        'anisotropy_field_inv': H_ani_inv,  # inverse anisotropy field
        'sat_mag': m,  # sets the magnitude of the magnetic moment regardles of p2.dip
        'anisotropy_energy': KV,  # keep >= 5
        # kinetic Monte Carlo time increment [s]
        'sw_dt_incr': time_step * time_scale,
        # inverse attempt time (1/tau_0) [1/s]
        'sw_tau0_inv': 1/attempt_time_mat
    }
    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

# Dipolar Direct Sum
dds = espressomd.magnetostatics.DipolarDirectSum(
    prefactor=prefactor_dds, gpu=False)
system.magnetostatics.solver = dds

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)

# iterations -= 1
xis = hysteresis_xis_sin(ulim_xis, iterations)

if vis:
    os.makedirs(
        f"_data/vtk_frames/mag_response{Lambda}{KV}/", exist_ok=True)
    folder_path = Path(
        f"_data/vtk_frames/mag_response{Lambda}{KV}/")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
# endregion

# region: Main Loop ################################################################
update_xi(system, 0, m, field_dir)
# system.integrator.run(100)

dipms = []

esp_dp0_zee = []
esp_dp1 = []
ani_energies = []

track_field = []

nr_of_frames = 100
frame = 0

for i in tqdm.tqdm(range(iterations)):
    # write animation frame
    if vis and frame/nr_of_frames <= i/iterations:
        writevtk(
            f"_data/vtk_frames/mag_response{Lambda}{KV}/mag{frame}.vtk", system, mag=True, easy_axes=True)
        frame += 1

    # Record Energies, per hand calculated for anisotropy energy
    esp_dp0_zee.append(system.analysis.energy()[("dipolar", 0)]/poss.shape[0])
    esp_dp1.append(system.analysis.energy()[
                   ("dipolar", 1)]/poss.shape[0])  # ARTIFICIAL MINUS
    e = 0
    for part in system.part.select(type=1):
        e += anisotropy_energy(part.dip, KV)/poss.shape[0]
    ani_energies.append(e)

    field_sum = []
    for part in system.part.select(type=1):
        field_sum.append(abs(np.linalg.norm(part.dip_fld)))
    track_field.append(sum(field_sum)/len(field_sum))

    # Draw a SW Energy landscape to see energetic minima
    if i == 10 or i == 11:
        # Track local dipole field to one particle
        field = system.part.by_id(250).dip_fld
        angles, energy, magnetization, minima_idx, lower_max_idx, barriers = stoner_wohlfarth_energy(
            field+xis[i] * kT / m * field_dir, KV, m, title_apendix=f"{i}th_step")

    # Step over
    dipms.append(dipm_tot_z.calculate()[2]/m/poss.shape[0])
    update_xi(system, xis[i], m, field_dir)
    system.integrator.run(step)

# endregion

report = True
# region: report
if report:
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    ax[0, 0].plot(np.arange(len(xis)), xis)
    ax[0, 0].set_title("Magnetic field strength")
    ax[0, 0].set_xlabel("time steps")
    ax[0, 0].set_ylabel("\u03BE")

    ax[0, 1].plot(xis, dipms)
    ax[0, 1].set_title("Hysteresis")
    ax[0, 1].set_xlabel("\u03BE")
    ax[0, 1].set_ylabel(r"$m_z / m_{sat}$")

    ax[1, 0].plot(np.arange(len(ani_energies)),
                  ani_energies, label=f"Anisotropy")
    ax[1, 0].plot(np.arange(len(esp_dp0_zee)), esp_dp0_zee, label="Zeeman")
    ax[1, 0].plot(np.arange(len(esp_dp1)), esp_dp1,
                  label=f"Dipolar")
    ax[1, 0].plot(np.arange(len(esp_dp1)), np.array(ani_energies)+np.array(esp_dp0_zee)+np.array(esp_dp1),
                  label=f"Total")
    ax[1, 0].plot(np.arange(len(track_field)), np.array(track_field),
                  label=f"field_str_avg")
    ax[1, 0].set_title(f"Energies")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("time steps")
    ax[1, 0].set_ylabel(r"Energy in $k_bT$")

    fig.suptitle(
        f"KV = {KV}, \u03BB = {Lambda}, frequency = {frequency:.2e} Hz")

    plt.tight_layout()
    plt.savefig(f"Time_plot_Energies_KV_{KV}_L_{Lambda}.png")
    plt.clf()
# endregion

fifth = int((iterations - 1) / 5)

top_curve = dipms[fifth:fifth*3]
bottom_curve = dipms[fifth*3:]

plt.scatter(xis[fifth:fifth*3], top_curve)
plt.scatter(xis[fifth*3:], bottom_curve)
plt.savefig("hyst_sym.png")
