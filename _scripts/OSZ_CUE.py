import json
import subprocess
from pathlib import Path

import tqdm
import numpy as np

filename = Path(__file__).resolve().parent.parent / "osz.json"
with open(filename, "w") as f:
    json.dump({}, f, indent=4)

ratios = [0]
lams = np.linspace(1, 5, 10)
KVs = np.linspace(3, 13, 10)

with tqdm.tqdm(total=len(ratios), desc="Ratio", position=0, leave=True) as ratio_bar:
    with tqdm.tqdm(total=len(lams), desc="Lambda", position=1, leave=True) as lambda_bar:
        with tqdm.tqdm(total=len(KVs), desc="KV", position=2, leave=True) as kv_bar:
            for ratio_index, ratio in enumerate(ratios):
                if ratio_index:
                    lambda_bar.reset()
                for lam in lams:
                    kv_bar.reset()
                    for KV in KVs:
                        result = subprocess.run(
                            [
                                "/home/xeranes/espresso/build/pypresso",
                                "/workspace/_scripts/OSZ_detection.py",
                                str(ratio), str(lam), str(KV),
                            ],
                            check=True,
                            capture_output=True,
                        )
                        output = float(result.stdout)
                        kv_bar.update(1)
                        if output <= 0.01:
                            break
                    lambda_bar.update(1)
                ratio_bar.update(1)


# for ratio in ratios:
#     subprocess.run(
#         [
#             "/home/xeranes/espresso/build/pypresso",
#             "/workspace/_scripts/plot_OSZ.py",
#             str(ratio)
#         ],
#         check=True,
#     )
