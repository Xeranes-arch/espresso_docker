import numpy as np
import tqdm
from functions import *
import espressomd
import espressomd.propagation
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
Propagation = espressomd.propagation.Propagation

KV = 10
Lambda = 1
easy_axis = [1, 0, 1]

kT = 1
mu_0 = 1
sigma = 1
mass = 1
###################### --Calculated params--######################
m = np.sqrt(Lambda * 4*np.pi * sigma ** 3 * kT/mu_0)
V = np.pi/6 * sigma**3
M_s = m/V
# Anisotropy
H_ani_inv = 1/(2*KV/(mu_0 * m))

L = 15
spacing = 1
dx = spacing
dy = np.sqrt(3.0) * spacing / 2.0
dz = np.sqrt(2.0 / 3.0) * spacing
buffer = 0.00001

print("Box dimensions: ", round(L*dx-buffer, 3),
      round(L*dy-buffer, 3), round(L*dz-buffer, 3))
system = espressomd.System(box_l=[L*dx-buffer, L*dy-buffer, L*dz-buffer])

# This is the documented way in newer ESPResSo versions
state = system.cell_system.get_state()

print(
    f"MPI rank count: {state['n_nodes']}"
)
print(
    f"Cell node grid: {system.cell_system.node_grid}"
)

# region positions

poss = hex_cube_points(L*dx-buffer, L*dy-buffer, L*dz-buffer)
print("Particle nr: 2x ", len(poss))
# endregion

system.time_step = 1  # MD time step in simulation units

system.min_global_cut = 1
system.cell_system.skin = 0.4
system.thermostat.set_langevin(
    kT=kT, gamma=1, gamma_rotation=1, seed=42)

# Particle setup
for pos in poss:
    # Anisotropy axis particle
    if True:
        title_prefix = "RV"
        p1 = system.part.add(pos=pos, fix=(True, True, True), type=0)
        p1.director = easy_axis  # easy axis direction
        p1.rotation = (False, False, False)

    p2 = system.part.add(pos=pos, type=1)
    # set dipole moment for the virtual particle in reduced units
    p2.dip = [1, 0, 0]
    # disable rotations of the virtual site tSW handles this (flipping between minima only!)
    p2.rotation = (False, False, False)
    p2.magnetodynamics = {
        'is_enabled': True,
        # inverse anisotropy field (1/H_k) in reduced units
        'anisotropy_field_inv': H_ani_inv,
        'sat_mag': M_s,  # saturation magnetisation in reduced units
        'anisotropy_energy': KV,  # anisotropy energy K * V in reduced units !!!KV/kT > 3
        'sw_dt_incr': 1e-9,  # kinetic Monte Carlo time increment [s]
        'sw_tau0_inv': 1.0e9  # inverse attempt time (1/tau_0) [1/s]
    }
    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

# Dipolar Direct Sum for DpDp
dds = espressomd.magnetostatics.DipolarDirectSum(
    prefactor=Lambda, gpu=False, n_replicas=1)
system.magnetostatics.solver = dds

print("Successfully created x particles")

e = []
frame = 0
for _ in tqdm.tqdm(range(100)):
    # Run
    system.integrator.run(1)
    e.append(system.analysis.energy()["total"])
    # # write animation frame
    if rank == 0:
        writevtk(
            f"_data/vtk_frames/mag_response/mag{frame}.vtk", system, mag=True, easy_axes=True)
        frame += 1

plt.plot(np.arange(len(e)), e)
plt.savefig("plot.png")
