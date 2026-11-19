import subprocess

nrs = [10]
ratios = []
for i, val in enumerate(nrs):
    ratios.append((val, 1))

print(ratios)
nr_of_part = [512]

for (a, b) in ratios:
    for nr in nr_of_part:
        subprocess.run(["/home/xeranes/espresso/build/pypresso",
                        "/workspace/_scripts/one_go.py", str(a), str(b), str(nr)], check=True)
