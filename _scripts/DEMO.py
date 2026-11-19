# TODO: critical field measure. maybe clockface again?

# region: imports ################################################################

from functions import *

import os
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import espressomd
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
Lambda = 8

KV = 5
easy_axis = [0, 0, 1]

xi = 3
field_dir = [1, 0, 0]

time_step = 0.01
iterations = 200

prefactor_dds = 1

vis = True  # records vtk frames
# vis = False

# avoids numerical instabilities that can happen when you give a 0 value in any direction
easy_axis = 1e-7 + np.array(easy_axis)
field_dir = 1e-7 + np.array(field_dir)
# normalize
easy_axis = easy_axis/np.linalg.norm(easy_axis)
field_dir = field_dir/np.linalg.norm(field_dir)

###################### --Calculated params-- ######################
m = np.sqrt(4*np.pi/mu_0 * Lambda * sigma ** 3 * kT)  # Mag moment
print("Mag. moment: ", m)
H_ani_inv = 1/(2*KV/(m))  # Inverse Anisotropy Field
# endregion

# region: System setup ################################################################
box_l = 90

system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step  # MD time step in simulation units
system.cell_system.skin = 0.4
system.min_global_cut = 11
system.thermostat.set_langevin(
    kT=kT, gamma=1, gamma_rotation=1, seed=42)


poss = np.array([[0, 0, 0], [1, 1e-7, 1e-7]])

# Easy axis Particle - applies to all others
p1 = system.part.add(pos=[box_l/2, box_l/2, box_l/2],
                     fix=(True, True, True), type=0)
p1.director = easy_axis
p1.rotation = (False, False, False)

k = -1
# Particles carrying magnetic moments
for pos in poss:
    pos = [j + box_l/2 for j in pos]  # centering

    p2 = system.part.add(pos=pos, type=1)
    p2.dip = np.array([1e-7, 1e-7, 1])
    k = 1
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
# endregion

# region: Main Loop ################################################################
update_xi(system, xi, m, field_dir)
# system.integrator.run(180)

esp_dp0_zee = []
esp_dp1 = []
ani_energies = []
nr_of_switches = 0

nr_of_frames = iterations
frame = 0

fields = [[], []]
track_dps = [[], []]
for i in tqdm.tqdm(range(iterations)):

    # write animation frame
    if vis and frame/nr_of_frames <= i/iterations:
        writevtk(
            f"_data/vtk_frames/mag_response/mag{frame}.vtk", system, mag=True, easy_axes=True)
        frame += 1

    # Record Energies, per hand calculated for anisotropy energy
    esp_dp0_zee.append(system.analysis.energy()[("dipolar", 0)])
    esp_dp1.append(system.analysis.energy()[("dipolar", 1)])
    ani_energies.append(anisotropy_energy(system.part.by_id(1).dip, KV))

    # Detect switches that are not field reverse oscillations
    dps = system.part.select(type=1).dip / m
    if i > 1:
        dots = np.sum(prev_dps * dps, axis=-1)
        if (dots < 0).any() and not (dots < 0).all():
            nr_of_switches += 1
    prev_dps = dps

    # MANUAL INTERFERANCE
    # if i == 100:
    #     dip = system.part.by_id(1).dip
    #     norm = np.linalg.norm(dip)
    #     dip = [system.part.by_id(
    #         1).dip[0]+0.5, dip[1], -1*dip[2]]
    #     rescale = norm / np.linalg.norm(dip)
    #     system.part.by_id(1).dip = [i * rescale for i in dip]

    # Record to Draw a SW Energy landscape to see energetic minima
    for j in [0, 1]:
        fields[j].append(system.part.by_id(j+1).dip_fld + xi*kT/m*field_dir)
        track_dps[j].append(system.part.by_id(j+1).dip)

    # Step over
    system.integrator.run(1)

# Observed average switches
time_ratio = time_step * time_scale / attempt_time_mat
p = nr_of_switches/(iterations-1)/len(poss)
print(LINE)
print("With time step / attempt time: ", time_ratio)
print("And with total nr of detected switches: ", nr_of_switches)
print("Average switching probability per step:", p
      )

# endregion

report = True
# region: Energies over time report
if report:
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.plot(np.arange(len(ani_energies)), ani_energies, label=f"Anisotropy")
    ax.plot(np.arange(len(esp_dp0_zee)), esp_dp0_zee, label="Zeeman")
    ax.plot(np.arange(len(esp_dp1)), esp_dp1,
            label=f"Dipolar")

    ax.plot(np.arange(len(esp_dp1)), np.array(ani_energies)+np.array(esp_dp0_zee)+np.array(esp_dp1),
            label=f"Total")
    ax.set_xlabel("time steps")
    ax.set_ylabel(r"Energy in $k_bT$")
    norm = np.linalg.norm(field_dir)
    field_dir = [[round(float(i / norm), 2) for i in field_dir]]
    ax.set_title(
        f"KV = {KV}, \u03BB = {Lambda}, \u03BE = {xi}, field_dir = {field_dir}")
    ax.legend()
    plt.savefig(f"Time_plot_Energies.png")
    plt.clf()
# endregion


# region: Animation
start = 0
end = 20
partial_field = [fields[0][start:end], fields[1][start:end]]
partial_dps = [track_dps[0][start:end], track_dps[1][start:end]]
animate(partial_field, partial_dps, m, KV, easy_axis, Lambda, xi)
