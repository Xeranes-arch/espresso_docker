import subprocess
import numpy as np

import time

t_0 = time.time()

ratios = ["2.0", "3.0", "4.0", "5.0", "6.0"]
# ratios = ["2.0", "3.0"]

lambdas = [0.25, 0.5, 1.0]

KVS = [5]

pnrs = [64.0, 128.0, 256.0, 512.0]
# pnrs = [64.0, 128.0]

axes = ["[0, 0, 1]"]

steps_per_ms = [1, 10, 100]

params = []
run_nr = 1
for steps_per_m in steps_per_ms:
    for ratio in ratios:
        for Lambda in lambdas:
            for KV in KVS:
                for pnr in pnrs:
                    for ax in axes:
                        print("CURRENT RUN NR: ", run_nr)
                        run_nr += 1
                        print(str(float(ratio)), str(float(Lambda)), str(
                            float(KV)), str(float(pnr)), ax, str(int(steps_per_m)))
                        subprocess.run(["/home/xeranes/espresso/build/pypresso",
                                        "/workspace/_scripts/mag_response_ellipsoid_remanance.py", str(float(ratio)), str(float(Lambda)), str(float(KV)), str(float(pnr)), ax, str(int(steps_per_m))], check=True)


print("Total time: ", time.time() - t_0)
