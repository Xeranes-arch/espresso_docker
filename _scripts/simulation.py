import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import espressomd

from functions import writevtk
from functions import rpos
from functions import rq

LINE = "\n___________________________"

AXES_list = [[3, 1.5, 1.5]]

particles = []
for AX in AXES_list:
    end_str = f"{float(AX[0])}_{float(AX[1])}_{float(AX[2])}"
    filename = f"_data/coordinates/512_ratio_2.0_1.0.txt"

    print(LINE)
    print(f"Loading: {end_str}")
    pos_arr = np.loadtxt(filename)
    particles.append(pos_arr)


# size of the simulation box, arbitrary, should be larger than the raspberry...
box_l = 90.0
center = [box_l / 2 for _ in range(3)]

skin = 0.4  # Skin parameter for the Verlet lists
time_step = 0.01
LB_time_step = 0.01

# Interaction parameter
eps_ss = 1  # LJ epsilon
sig_ss = 1  # LJ sigma

# Seed
seed_pass = np.random.randint(0, 2**16 - 1)
print("Seed Pass: ", seed_pass)
#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]

center = system.box_l / 2

# Lattice Boltzmann
lbf = espressomd.lb.LBFluid(
    agrid=1., density=1., kinematic_viscosity=1., tau=0.01, gpu=True)
system.lb = lbf
system.integrator.run(100)


# the LJ potential (WCA potential)
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss+0.1
)


smallest_id = 0
for pos_arr in particles:

    for i, pos in enumerate(pos_arr):
        system.part.add(id=smallest_id + i + 1, pos=pos, type=1,
                        rotation=[True, True, True], dip=rq()[1:])
    highest_current = smallest_id + i + 1

    ########## Calc for virt ##########
    com = np.average(system.part.by_ids(
        np.arange(smallest_id + 1, highest_current + 1)).pos, axis=0)
    momIx = 0
    momIy = 0
    momIz = 0
    for id in np.arange(smallest_id + 1, highest_current + 1):
        momIx += np.power(np.linalg.norm(com[1:] -
                                         system.part.by_id(id).pos[1:]), 2)
        momIy += np.power(
            np.linalg.norm(
                np.array(
                    [
                        com[0] - system.part.by_id(id).pos[0],
                        com[2] - system.part.by_id(id).pos[2],
                    ]
                )
            ),
            2,
        )
        momIz += np.power(np.linalg.norm(com[:2] -
                                         system.part.by_id(id).pos[:2]), 2)
    ################################

    system.part.add(id=smallest_id, pos=com, type=0,
                    rotation=[True, True, True])

    system.part.by_id(smallest_id).mass = highest_current - smallest_id + 1
    system.part.by_id(smallest_id).rinertia = np.array([momIx, momIy, momIz])

    for p in system.part.by_ids(
            np.arange(smallest_id + 1, highest_current + 1)):
        p.vs_auto_relate_to(smallest_id)

    system.part.by_id(smallest_id).pos = rpos(box_l)
    system.part.by_id(smallest_id).quat = rq()
    system.integrator.run(0, recalc_forces=True)

    print(system.part.by_id(smallest_id).pos)
    smallest_id = highest_current + 1

# Remove overlap between particles by means of the steepest descent method
mass = 1.0
f_max = 0.01 * sig_ss * mass / system.time_step**2
system.integrator.set_steepest_descent(
    f_max=f_max,
    gamma=0.1,
    max_displacement=0.05)
system.integrator.run(5000)
assert np.all(np.abs(system.part.all().f) <
              f_max), "Overlap removal did not converge!"

# Switch to velocity Verlet integrator
system.integrator.set_vv()

# system.thermostat.set_lb(LB_fluid=lbf, seed=seed_pass, gamma=1.5)
system.thermostat.set_langevin(kT=1, gamma=1, seed=seed_pass)

p3m = espressomd.magnetostatics.DipolarP3M(prefactor=1, mesh=32, accuracy=1E-4)
system.magnetostatics.solver = p3m


for i in tqdm(range(1000)):
    system.integrator.run(10)
    writevtk(f"_data/vtk_frames/system/system_{i}.vtk", system)
