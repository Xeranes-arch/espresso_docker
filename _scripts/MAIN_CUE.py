import subprocess
import tqdm

lams = [3, 6]
KVs = [5, 8]
for lam in tqdm.tqdm(lams):
    for KV in tqdm.tqdm(KVs):
        print("\n----------------------------------")
        subprocess.run(["/home/xeranes/espresso/build/pypresso",
                        "/workspace/_scripts/MAIN.py", str(lam), str(KV)], check=True)
        print("\n----------------------------------")
