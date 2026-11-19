# region: imports ################################################################

import importlib
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import espressomd
import espressomd.propagation
Propagation = espressomd.propagation.Propagation

# functinos.py import
sys.path.insert(0, "/home/xeranes/SCRIPTS")
functions = importlib.import_module("functions")
globals().update({
    name: value
    for name, value in vars(functions).items()
    if not name.startswith("_")
})
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
Lambda = 3

# KV = float(sys.argv[1])
KV = 5
easy_axis = [0, 0, 1]

xi = 10
field_dir = [1, 0, 0]

# kMC_timestep_ratio = float(sys.argv[2])
kMC_timestep_ratio = 0.1
time_step = kMC_timestep_ratio * (attempt_time_mat/time_scale)

iterations = 100

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

poss = []
for i in range(10):
    for j in range(10):
        for k in range(10):

            poss.append([i, j, k])

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
# endregion

# region: Main Loop ################################################################
update_xi(system, xi, m, field_dir)

esp_dp0_zee = []
esp_dp1 = []
ani_energies = []
nr_of_switches = 0

nr_of_frames = 100
frame = 0

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
            nr_of_switches += np.sum((dots < 0))
    prev_dps = dps

    # Draw a SW Energy landscape to see energetic minima
    if i == 10:
        for j in [1, 2]:
            # Track local dipole field to one particle
            field = system.part.by_id(j).dip_fld
            angles, energy, magnetization, minima_idx, lower_max_idx, barriers = stoner_wohlfarth_energy(
                field+xi * kT / m * field_dir, KV, m, title_apendix=f"10th_step_part{j}")
            barrier1 = energy[lower_max_idx] - min(energy[minima_idx])
            barrier2 = energy[lower_max_idx] - max(energy[minima_idx])
            pred_prob = probability_preditcion(barrier1, kMC_timestep_ratio)

    # Step over
    system.integrator.run(1)

# Observed average switches
p = nr_of_switches/(iterations-1)/len(poss)
print(LINE)
print("With time step / attempt time: ", kMC_timestep_ratio)
print("And with total nr of detected switches: ", nr_of_switches)
print(LINE)
print("Average switching probability per step:\n", p
      )
print("Given symmetric minima, predicted probability by fit:\n", pred_prob)
print(LINE)

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

write_json = False
if write_json:
    energy_barrier = min(barrier1, barrier2)

    # Write json file
    filename = "data.json"
    # Read JSON
    with open(filename, "r") as f:
        data = json.load(f)
    # Append values
    data["Energy_barrier"].append(energy_barrier)
    data["probability"].append(p)
    data["time_ratio"].append(kMC_timestep_ratio)
    # Overwrite JSON
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
