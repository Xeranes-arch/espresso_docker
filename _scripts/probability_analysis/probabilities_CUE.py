import numpy as np
import subprocess
import json
import tqdm

# Reset json file to empty
filename = "data.json"
# Read JSON
with open(filename, "r") as f:
    data = json.load(f)
# Append values
data["Energy_barrier"] = []
data["probability"] = []
data["time_ratio"] = []
# Overwrite JSON
with open(filename, "w") as f:
    json.dump(data, f, indent=4)


def custom_space(start, stop, num, curvature=1):
    t = np.linspace(0, 1, num)
    t = t**curvature
    return start + (stop - start) * t


steps = 20
time_ratios = custom_space(0.001, 10, steps, curvature=2)
KVs = custom_space(5, 10, steps, curvature=1.5)
for val in tqdm.tqdm(KVs):
    for time_ratio in time_ratios:
        print("\n----------------------------------")
        subprocess.run(["/home/xeranes/espresso/build/pypresso",
                        "/workspace/_scripts/probability_analysis/probabilities.py", str(val), str(time_ratio)], check=True)
        print("\n----------------------------------")
