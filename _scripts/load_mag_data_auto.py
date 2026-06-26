import numpy as np
import matplotlib.pyplot as plt
import os

line_styles = [
    '-',   # Solid (Default)
    '--',  # Dashed
    ':',   # Dotted
    '-.',   # Dash-Dot
]

if os.path.exists("currents"):
    with open("currents", "r") as f:
        # Read all lines, strip the trailing '\n', and filter out empty lines
        nms = [line.strip() for line in f if line.strip()]

nms = []
KVS = [3, 4]
for KV in KVS:
    nms.append(
        f"/workspace/_data/mag_response/mag_response_data_r2.0_l0.1_KV{KV}.npz")
    nms.append(
        f"/workspace/_data/mag_response/mag_response_data_r2.0_l0.25_KV{KV}.npz")
    nms.append(
        f"/workspace/_data/mag_response/mag_response_data_r2.0_l0.5_KV{KV}.npz")

# nms = ["/workspace/_data/mag_response/mag_response_data_r2.0_l1_KV3.npz"]

cs = ["black", "blue"]
for i, a in enumerate(nms):
    loaded = np.load(a)

    print("\n", a)

    dipm_means = np.array(loaded["means"])
    stds = np.array(loaded["stds"])
    alphas = np.array(loaded["alphas"])
    ratio = np.array(loaded["ratio"])
    Lambda = np.array(loaded["Lambda"])
    KV = np.array(loaded["KV"])

    alphas = alphas[:len(dipm_means)]
    plt.plot(
        alphas,
        dipm_means,
        marker="+",
        color=cs[i//3],
        linestyle=line_styles[i % 3],
        markersize=8,  # Size of the plus marker
        markeredgewidth=1,
        label=f"r:{ratio}, " + r"$\lambda$"+f": {Lambda}, KV: {KV}"
    )


plt.xlim(-0.5, max(alphas) + 1)
plt.ylim(-0.1, 1.1)

plt.legend()
plt.xlabel(r"Magnetic field strength $\alpha$ in kT")
plt.ylabel(r"$\langle M_z \rangle / M_{sat} N$")
plt.savefig(
    f"_data/mag_response/graphs/r_{ratio}_L_{Lambda}_KV_{KV}.png")
plt.clf()

with open("currents", "w") as f:
    pass
