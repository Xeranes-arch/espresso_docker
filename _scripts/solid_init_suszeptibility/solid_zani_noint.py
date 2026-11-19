# SW docker image
# region: imports
from functions import rv
from functions import upalpha, save_to_json, fcc_box_points

import tqdm
import numpy as np

import espressomd
import espressomd.observables
import espressomd.propagation
Propagation = espressomd.propagation.Propagation
# endregion

json_id = 0
easy_axis = [1e-6, 0., 1.]

nr_of_cells = 10
interacting = False
mirror_images = 1

steps_per_measurement = 10000

# region: System setup
######################### ---Fixed--- #########################
KV = 5
Lambda = 0.15

kT = 1
mass = 1
mu_0 = 1
sigma = 1

################### ---Calculated params--- ###################
m = np.sqrt(Lambda * 4*np.pi * sigma ** 3 * kT/mu_0)
V = np.pi/6 * sigma**3
M_s = m/V

H_ani_inv = 1/(2*KV/(mu_0 * m))

######################### ---System--- #########################
nr_per_cell = 2
pos_per_cell = 4
nr_of_part = (nr_per_cell * nr_of_cells**3)
nr_of_pos = (pos_per_cell * nr_of_cells**3)
box_l = np.sqrt(2) * nr_of_cells
V_sys = box_l**3

poss = fcc_box_points(box_l, box_l, box_l)
assert (nr_of_pos == poss.shape[0])
print("Part nr: ", poss.shape[0])

time_step = 0.01  # Doesn't matter for MC tSW steps
system = espressomd.System(box_l=[box_l, box_l, box_l])
system.time_step = time_step  # MD time step in simulation units
system.cell_system.skin = 0.4
system.thermostat.set_langevin(
    kT=kT, gamma=1, gamma_rotation=1, seed=42)

# For uniform anisotropy axes
if easy_axis:
    p1 = system.part.add(pos=[0, 0, 0], fix=(True, True, True), type=0)
    p1.director = easy_axis  # easy axis direction
    p1.rotation = (False, False, False)

# Particle setup
for i, pos in enumerate(poss):

    # Anisotropy axis
    if not easy_axis:
        p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
        p1.director = rv()  # easy axis direction
        p1.rotation = (False, False, False)

    # Magnetic moment particle
    p2 = system.part.add(pos=pos, type=1)
    p2.dip = rv()
    p2.rotation = (False, False, False)  # flipping between minima only!
    p2.magnetodynamics = {
        'is_enabled': True,
        'anisotropy_field_inv': H_ani_inv,  # inverse anisotropy field (1/H_k)
        'sat_mag': M_s,  # saturation magnetisation
        'anisotropy_energy': KV,  # anisotropy energy K * V !!!KV/kT > 3
        'sw_dt_incr': 1.0e-10,  # kinetic Monte Carlo time increment [s]
        'sw_tau0_inv': 1.0e9  # inverse attempt time (1/tau_0) [1/s]
    }
    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

if interacting:
    # Dipolar Direct Sum for DpDp
    dds = espressomd.magnetostatics.DipolarDirectSum(
        prefactor=Lambda, gpu=False, n_replicas=mirror_images)
    system.magnetostatics.solver = dds

# To be observed
dipm_tot_z = espressomd.observables.MagneticDipoleMoment(
    ids=system.part.all().id)
# endregion

###################### ------alpha values------ ######################
alphas = []

llim_alphas = 0
ulim_alphas = 3

iterations = 10
alphas = np.linspace(llim_alphas, ulim_alphas, int(iterations))
upalpha(system, 0, m)
system.integrator.run(steps_per_measurement)
########################################################################


print("With Steps per measurement = ", steps_per_measurement)
vis = True
vis = False
frame = 0
dipms = []
for i in tqdm.tqdm(range(iterations)):

    upalpha(system, alphas[i], m)
    system.integrator.run(steps_per_measurement)
    collect = []
    for i in range(10):
        system.integrator.run(int(steps_per_measurement/10))
        collect.append(dipm_tot_z.calculate()[2]/M_s/poss.shape[0])
    dipms.append(np.average(collect))
    print("STD: ", round(np.std(collect), 4))

alphas = alphas.tolist()
filename = "./model_fit.json"
entry = {
    "id": json_id,
    "alphas": alphas,
    "dipms": dipms,
}
save_to_json(filename, entry)
