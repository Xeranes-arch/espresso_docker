import json
import subprocess
from pathlib import Path

import tqdm
import numpy as np

filename = Path(__file__).resolve().parent.parent / "osz.json"
with open(filename, "w") as f:
    json.dump({}, f, indent=4)

ratios = [2, 6]
lams = np.arange(1, 3.1, 0.5)
KVs = np.arange(3, 9, 1)

with tqdm.tqdm(total=len(ratios), desc="Ratio", position=0, leave=True) as ratio_bar:
    with tqdm.tqdm(total=len(lams), desc="Lambda", position=1, leave=True) as lambda_bar:
        with tqdm.tqdm(total=len(KVs), desc="KV", position=2, leave=True) as kv_bar:
            for ratio_index, ratio in enumerate(ratios):
                if ratio_index:
                    lambda_bar.reset()
                for lam in lams:
                    kv_bar.reset()
                    for KV in KVs:
                        subprocess.run(
                            [
                                "/home/xeranes/espresso/build/pypresso",
                                "/workspace/_scripts/OSZ_detection.py",
                                str(ratio), str(lam), str(KV)
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL
                        )
                        kv_bar.update(1)
                    lambda_bar.update(1)
                ratio_bar.update(1)
