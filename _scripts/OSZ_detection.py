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


def print(*args, override=False, **kwargs):
    if override or not subprocess:
        __builtins__.print(*args, **kwargs)


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

# region: Parameters #################################################################

subprocess = True
if subprocess:
    ratio = int(sys.argv[1])
    Lambda = float(sys.argv[2])
    KV = float(sys.argv[3])
else:
    ratio = 0
    Lambda = 1
    KV = 6

prefactor_dds = mu_0/(4*np.pi)

easy_axis = [0, 0, 1]

field_dir = [0, 0, 1]

# Sets total iteration steps
frequency = 1e7

# Is ratio for tSW and gives time step
# kMC_timestep_ratio = 0.001
kMC_timestep_ratio = 1
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

if ratio == 0:
    poss = fcc_box_points(7, 7, 7)
    pnr = len(poss)
else:
    # Particle coordinates
    filename = f"_data/coordinates/512.0_ratio_{ratio}.0_1.0.txt"
    poss = np.loadtxt(filename)
    pnr = 512
# exit()

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

if vis:
    os.makedirs(
        f"_data/vtk_frames/mag_response/", exist_ok=True)
    folder_path = Path(
        f"_data/vtk_frames/mag_response/")
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()

update_xi(system, 0, m, field_dir)
# endregion

# region: pre catch osz to save compute time
system.integrator.run(100)

r = 3
idxs = range(1, pnr)
buffer = [np.zeros((len(idxs), 3))]*3
buffer_len = len(buffer)
buffer_idx = 0
nr_of_osz_caught = 0
score = 0.0
for i in range(3+r):
    system.integrator.run(1)
    # Oszillation detection
    buffer[buffer_idx] = system.part.all().dip[idxs]
    buffer_idx = (buffer_idx + 1) % buffer_len
    if i >= 3:
        v0, v1, v2 = (
            buffer[(buffer_idx + offset) % buffer_len]
            for offset in range(buffer_len)
        )
        for sample_idx in range(len(idxs)):
            if detect_oscillation(
                v0[sample_idx], v1[sample_idx], v2[sample_idx]
            ):
                nr_of_osz_caught += 1
# endregion

if nr_of_osz_caught/pnr/r > 0.9:
    score = nr_of_osz_caught/pnr/r
else:
    # region: Equilibriation ##############################################################
    print(LINE)
    print("Equillibriating...")

    prev = 0
    count = 0
    i = 0
    for i in tqdm.tqdm(range(100)):

        writevtk(
            f"_data/vtk_frames/mag_response/frame{i}.vtk", system, mag=True, easy_axes=True)

        system.integrator.run(101)

        new = system.analysis.energy()["total"]
        plt.scatter(i, new, c="blue", marker="+")

        if abs(prev - new) <= abs(prev/1000):
            count += 1
            if count > 10:
                print(LINE)
                print("EQUIL")
                break
        else:
            count = 0
        prev = new

    plt.savefig("equil.png")
    plt.clf()
    # endregion

    # region: Osz var #####################################################################
    nr_of_osz_caught = 0
    for i in range(20):
        system.integrator.run(1)
        # Oszillation detection
        buffer[buffer_idx] = system.part.all().dip[idxs]
        buffer_idx = (buffer_idx + 1) % buffer_len

        if i >= 3:
            v0, v1, v2 = (
                buffer[(buffer_idx + offset) % buffer_len]
                for offset in range(buffer_len)
            )
            for sample_idx in range(len(idxs)):
                if detect_oscillation(
                    v0[sample_idx], v1[sample_idx], v2[sample_idx]
                ):
                    nr_of_osz_caught += 1
            print(f"Detected oszillations: {nr_of_osz_caught}")
            score = nr_of_osz_caught / ((i - 1) * len(idxs))
        if i > 5 and score >= 0.9:
            print(LINE)
            print("OSZILLATION")
            break
    # endregion

name = f"t_{time_step}_r_{ratio}_l_{Lambda}_KV_{KV}_"

writevtk(
    f"_data/vtk_frames/osz_end_frames/{name}frame_1.vtk", system, mag=True, easy_axes=True)
system.integrator.run(1)
writevtk(
    f"_data/vtk_frames/osz_end_frames/{name}frame_2.vtk", system, mag=True, easy_axes=True)

save_to_json(
    Path(__file__).resolve().parent.parent /
    f"_data/osz/t_{time_step}_r_{ratio}_osz.json",
    {"Lambda": Lambda, "KV": KV, "score": score},
)

print(score, override=True)
