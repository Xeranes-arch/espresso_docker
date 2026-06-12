import subprocess

anis = [0.0, 0.5, 1]
ratios = [2., 6.]

params = []
for ani in anis:
    for ratio in ratios:
        params.append([ratio, ani])

# print(params)

for (a, ani) in params:
    subprocess.run(["/home/xeranes/espresso/build/pypresso",
                    "/workspace/_scripts/mag_response_ell.py", str(a), str(ani)], check=True)
