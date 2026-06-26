import subprocess
import numpy as np

import time

t_0 = time.time()

ratios = ["2.0", "3.0", "4.0", "5.0", "6.0"]
ratios = ["2.0"]

# lambdas = [1, 2, 5]
lambdas = [0.25]

KVS = [3, 4, 5, 6]
KVS = [4]


ms = [np.sqrt(i) for i in lambdas]

params = []
for ratio in ratios:
    with open("currents", "w") as f:
        pass
    for Lambda in lambdas:
        for KV in KVS:
            filename = f"_data/mag_response/mag_response_data_r{ratio}_l{Lambda}_KV{KV}.npz"

            # Write the filename string to the 'current' file
            with open("currents", "a") as f:
                f.write(f"{filename}\n")

            subprocess.run(["/home/xeranes/espresso/build/pypresso",
                            "/workspace/_scripts/mag_response_ell.py", str(float(ratio)), str(float(Lambda)), str(float(KV)), filename], check=True)

    # subprocess.run(["/home/xeranes/espresso/build/pypresso",
    #                 "/workspace/_scripts/load_mag_data_auto.py"], check=True)

print("Total time: ", time.time() - t_0)
