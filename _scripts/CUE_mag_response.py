import subprocess

anis = [2, 3]
nrs = [2, 3, 4, 5, 6]
params = []
for ani in anis:
    for val in nrs:
        params.append([val, ani])

# print(params)

for (a, ani) in params:
    subprocess.run(["/home/xeranes/espresso/build/pypresso",
                    "/workspace/_scripts/mag_response_ell.py", str(a), str(ani)], check=True)
