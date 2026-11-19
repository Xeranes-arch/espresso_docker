import subprocess
import numpy as np

import time

LINE = "_________________________________\n"

t_0 = time.time()

# ratios = ["2.0", "3.0", "4.0", "5.0", "6.0"]
# ratios = ["2.0", "6.0", "10.0"]
ratios = ["6.0"]

# lambdas = [0.25, 0.5, 0.75]
lambdas = [0.5]
# lambdas = [0.1]

# pnrs = [64.0, 128.0, 256.0, 512.0]
pnrs = [512.0]

axes = ["[0, 0, 1]"]

# steps_per_ms = [1, 10, 100]
steps_per_ms = [1]

KVS = [10, 20]

run_nr = 1
for KV in KVS:
    for ratio in ratios:
        for pnr in pnrs:
            for ax in axes:
                for Lambda in lambdas:
                    for steps_per_m in steps_per_ms:
                        print(LINE, LINE, "CURRENT RUN NR: ", run_nr)
                        t = (time.time() - t_0)
                        print("Current time: ", int(t//60),
                              "min ", int(t % 60), "s")
                        run_nr += 1
                        print(str(float(ratio)), str(float(Lambda)),
                              str(float(pnr)), ax, str(int(steps_per_m)))
                        subprocess.run(["/home/xeranes/espresso/build/pypresso",
                                        "/workspace/_scripts/mag_response_ellipsoid_remanance.py", str(float(ratio)), str(float(Lambda)), str(float(pnr)), ax, str(int(steps_per_m)), str(float(KV))], check=True)
                        # subprocess.run(["/home/xeranes/espresso/build/pypresso",
                        #                 "/workspace/_scripts/mag_response_solid.py", str(int(steps_per_m))], check=True)


print("Total time: ", time.time() - t_0)
