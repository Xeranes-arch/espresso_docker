#############################################################################################################################
#############################################################################################################################
# region: README
# Placeholder
# endregion
#############################################################################################################################
#############################################################################################################################
# region: Import
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import *

import espressomd
espressomd.assert_features(
    [
        "ROTATION",
        "ROTATIONAL_INERTIA",
        "EXTERNAL_FORCES",
        "MASS",
        "VIRTUAL_SITES_RELATIVE",
        "LENNARD_JONES"
    ]
)
LINE = "___________________________\n"
# endregion
#############################################################################################################################
# region: User input
DEBUG = True


def print(*args, debug_override=False, **kwargs):
    """Only obtain barebones output via override = True when DEBUG = FALSE"""
    if DEBUG or debug_override:
        __builtins__.print(*args, **kwargs)


# Write vtk animation frames
save_visualization = True
# save_visualization = False

###

# Placeholder_parameters

# endregion
#############################################################################################################################
# region: System parameters

# size of the simulation box, arbitrary, should be larger than the raspberry...
l = 6
BOX_L = [l, l, l]
KT = 1
TIME_STEP = 0.001
MASS = 1.0

GAMMA = 1

# Interaction parameter
EPS = 1  # LJ epsilon
SIG = 1  # LJ sigma

# endregion
#############################################################################################################################
# region: System setup

system = espressomd.System(box_l=BOX_L)
system.time_step = TIME_STEP
system.cell_system.skin = 0.4  # Skin parameter for the Verlet lists
system.periodicity = [True, True, True]

# the LJ potential
system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=EPS, sigma=SIG, cutoff=2.5*EPS/SIG, shift="auto", offset=0)


# Place particles
for i in range((l-1)**3):
    system.part.add(pos=[random.uniform(0, l) for _ in range(3)])

# endregion
#############################################################################################################################
# region: Running system
frame_nr = 0

# Equillibriation
# Large overlaps with non-bonded interactinos could explode the system. Baby steps.
system.force_cap = 1000
print(LINE, "Equillibriating")
FMAX = 0.00001 * SIG * MASS / system.time_step**2
max_steps = 100
system.integrator.set_steepest_descent(
    f_max=FMAX,
    gamma=10,
    max_displacement=0.01)

print(FMAX)
mins = []
for i in range(max_steps):
    mins.append(system.analysis.min_dist(p1='default', p2='default'))
    system.integrator.run(1)
    print(i)
    # vtk frames
    path = f"_data/vtk_frames/default/frame{frame_nr}.vtk"
    writevtk(path, system)
    frame_nr += 1

    # equil condition
    if np.all(np.abs(system.part.all().f) < FMAX) and mins[-1] > 1:
        print("End equil")
        break

assert np.all(np.abs(system.part.all().f) <
              FMAX), "Overlap removal did not converge!"

###
i = 0
system.integrator.set_vv()
system.thermostat.set_langevin(kT=KT, gamma=GAMMA, seed=42)
while True:
    mins.append(system.analysis.min_dist(
        p1='default', p2='default'))
    system.integrator.run(1)

    # Exit condition
    if i == 100:
        break

    # vtk frames
    path = f"_data/vtk_frames/default/frame{frame_nr}.vtk"
    writevtk(path, system)
    frame_nr += 1

    i += 1

plt.plot(np.arange(len(mins)), mins)
plt.savefig("plot.png")


print(LINE)
print("Reached the end.")
