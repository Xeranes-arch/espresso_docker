import subprocess

# nrs = [2, 3, 4, 5]
# ratios = []
# for i, val in enumerate(nrs):
#     if val > 2.5:
#         ratios.append([val, 2.5])
#     if val < 2.5:
#         ratios.append([2.5, val])
#     if val > 3.5:
#         ratios.append([val, 3.5])
#     if val < 3.5:
#         ratios.append([3.5, val])

# print(ratios)

# for (a, b) in ratios:
#     subprocess.run(["/home/xeranes/espresso/build/pypresso",
#                    "/workspaces/espresso_docker/scripts/ellipsoid_4.2.py", str(a), str(b)], check=True)

a = 7
b = 1
subprocess.run(["/home/xeranes/espresso/build/pypresso",
               "/workspaces/espresso_docker/scripts/ellipsoid_4.2.py", str(a), str(b)], check=True)
