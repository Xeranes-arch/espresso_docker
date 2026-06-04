import subprocess

nrs = [2, 3, 4, 5, 6]
ratios = []
for i, val in enumerate(nrs):
    ratios.append((val, 1))

print(ratios)

for (a, b) in ratios:
    subprocess.run(["/home/xeranes/espresso/build/pypresso",
                    "/workspace/_scripts/one_go.py", str(a), str(b)], check=True)
