import numpy as np
import matplotlib.pyplot as plt
import espressomd

from functions import writevtk

LINE = "\n___________________________"

AXES_list = [[3., 1.5, 1.5], [2, 2, 2]]
particles = []
for AX in AXES_list:
    end_str = f"{float(AX[0])}_{float(AX[1])}_{float(AX[2])}"
    filename = f"/workspaces/espresso_docker/data/raspberry_coordinates[{end_str}].txt"

    print(LINE)
    print(f"Loading: {end_str}")
    pos_arr = np.loadtxt(filename)
    particles.append(pos_arr)


# size of the simulation box, arbitrary, should be larger than the raspberry...
box_l = 90.0
center = [box_l / 2 for _ in range(3)]

skin = 3  # Skin parameter for the Verlet lists
time_step = 0.001

# Interaction parameter
eps_ss = 1  # LJ epsilon
sig_ss = 1  # LJ sigma

#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=[box_l] * 3)
system.time_step = time_step
system.cell_system.skin = skin
system.periodicity = [True, True, True]

# the LJ potential (WCA potential)
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=eps_ss, sigma=sig_ss, cutoff=sig_ss, shift=eps_ss+0.1
)

# Seed
seed_pass = np.random.randint(0, 2**32 - 1)
print("Seed Pass: ", seed_pass)

system.thermostat.set_langevin(kT=0.001, gamma=40.0, seed=seed_pass)

center = system.box_l / 2
colPos = (0, 0, 0)

smallest_id = 0
for pos_arr in particles:
    print("smallest")
    print(smallest_id)

    for i, pos in enumerate(pos_arr):
        system.part.add(id=smallest_id + i + 1, pos=pos, type=1)
    highest_current = smallest_id + i + 1
    print("highest")
    print(highest_current)

    ########## Calc for virt ##########
    com = np.average(system.part.all().pos, axis=0)
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
    #################################

    system.part.add(id=smallest_id, pos=com, type=0)

    print("mass")
    print(highest_current - smallest_id)
    system.part.by_id(smallest_id).mass = highest_current - smallest_id + 1
    system.part.by_id(smallest_id).rinertia = np.array([momIx, momIy, momIz])

    for p in system.part.select(type=1):
        p.vs_auto_relate_to(smallest_id)

    smallest_id = highest_current + 1

# TODO spread em out
system.part.by_id(0).pos = center

writevtk(filename[:-4] + ".vtk", system)
