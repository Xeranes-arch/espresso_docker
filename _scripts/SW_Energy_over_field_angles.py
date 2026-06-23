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

H_angle = 0
phis = np.linspace(0, 2*np.pi, 100)
# phis = np.linspace(0, 2*np.pi, 10)

ratio = 2.
KV = 1

ratio = float(ratio)
KV = float(KV)
################

line_styles = [
    '-',   # Solid (Default)
    '--',  # Dashed
    ':',   # Dotted
    '-.',   # Dash-Dot
    '--.']

LINE = "_________________________________"
print(LINE)
print(f"Running ratio:{ratio}, ani:{KV}")

sigma = 1
kT = 1
mass = 1
m = np.sqrt(4 * np.pi)

V = np.pi/6 * sigma**3
K = KV / V
M_s = m/V

# DPDP
# with
mu_0 = 1
# Lambda = 1 for mu_0 = 1
Lambda = mu_0*m**2/(4*np.pi * sigma**3 * kT)
# Zeeman
alpha = 0
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


for i, pos in enumerate(pos_arr):

    # One particle with thermal Stoner-Wohlfarth enabled
    p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
    p1.director = (0, 0, 1)  # easy axis direction
    p1.rotation = (False, False, False)  # disallow particle rotation

    p2 = system.part.add(pos=p1.pos, fix=(True, True, True), type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = (0, 0, m)
    # enable rotations of the virtual site
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

# DP3M
accuracy = 5E-4
system.magnetostatics.solver = espressomd.magnetostatics.DipolarP3M(
    accuracy=accuracy, prefactor=Lambda * sigma**3 * kT)

E = espressomd.observables.Energy(
    ids=system.part.all().id)

H_crit = 2 * KV / m
alpha_crit = mu_0 * m * H_crit / kT

alpha = 1/3*50
H = kT * alpha / (mu_0 * m)
H = round(H, 2)

thetas = np.linspace(0, np.pi/2, 4)

for i, theta in enumerate(thetas):

    Hz = np.cos(theta) * H
    Hx = -np.sin(theta) * H

    # set magnetic field constraint
    H_field = [Hx, 0, Hz]
    H_constraint = espressomd.constraints.HomogeneousMagneticField(
        H=H_field)
    system.constraints.add(H_constraint)

    e_list = []
    for j, phi in enumerate(tqdm.tqdm(phis)):

        dipx = np.cos(phi) * m
        dipz = np.sin(phi) * m

        system.part.select(type=1).dip = (dipx, 0, dipz)

        system.integrator.run(0)

        e_list.append(E.calculate()/len(pos_arr))

    E_arr = np.array(e_list)
    np.savez(f"_data/mag_response/SW.npz",
             ratio=ratio, KV=KV, kT=kT, Lambda=Lambda, phis=phis, E_arr=E_arr)

    phis_deg = [round(i/(2*np.pi) * 360) for i in phis]
    label = f"{i}/6" + r"$\pi$"
    if not i:
        label = "0"
    if i == 3:
        label = r"$\pi$/2"
    plt.plot(
        phis_deg,
        E_arr,
        linestyle=line_styles[i],
        c="black",
        label=label

    )
    system.constraints.clear()

plt.legend(title=r"$\theta$ = ")
plt.xlabel(r"Angle $\phi$ of $\vec{m}$")
plt.ylabel("Energy")
plt.savefig("target_graph.png")
