import subprocess


ratios = ["2.0", "3.0", "4.0", "5.0", "6.0"]
anis = [100]

params = []
for ani in anis:
    for ratio in ratios:
        params.append([ratio, ani])

# print(params)

for (ratio, ani) in params:
    subprocess.run(["/home/xeranes/espresso/build/pypresso",
                    "/workspace/_scripts/mag_response_ell.py", str(float(ratio)), str(float(ani))], check=True)
