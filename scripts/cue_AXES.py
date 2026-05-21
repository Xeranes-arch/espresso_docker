import subprocess
nrs = [2,3,4,5]
ratios = []
for i, val in enumerate(nrs):
    for val2 in nrs[i:]:
        ratios.append([val,val2])

print(ratios)

for (a,b) in ratios:
    subprocess.run(["/home/xeranes/espresso/build/pypresso", "/workspaces/espresso_docker/scripts/ellipsoid_4.2.py", str(a), str(b)], check=True)