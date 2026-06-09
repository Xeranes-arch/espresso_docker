import numpy as np
import matplotlib.pyplot as plt


ratios = ["2.0", "3.0", "4.0", "5.0", "6.0"]
anis = ["3.0"]

ratios = ["2.0"]
anis = ["2.0", "3.0"]

fig, ax = plt.subplots(figsize=(10, 6))

for ratio in ratios:
    for c, ani in zip(["black", "red"], anis):
        loaded = np.load(
            f"/workspace/_data/mag_response/fg_mag_response_data_ratio{ratio}_ani{ani}.npz")

        dipm_means = np.array(loaded["dipm_means"])
        stds = np.array(loaded["stds"])

        alphas1 = np.arange(0, 8, 0.25)
        alphas2 = np.arange(8, 15.1, 0.5)
        alphas = np.concatenate((alphas1, alphas2))

        ax.errorbar(
            alphas,
            dipm_means,
            yerr=stds,
            fmt="-+",
            # color=c,
            # ecolor=c,
            elinewidth=1,  # Thickness of error bars
            capsize=4,  # Width of the horizontal caps
            capthick=1,  # Thickness of the horizontal caps
            markersize=8,  # Size of the plus marker
            markeredgewidth=1.5,
            label=f"ratio: {ratio}, ani: {ani}"
        )

# plt.xlim(0.9, 5)
# plt.ylim(0.4, 0.9)

plt.xlim(-0.5, 15.2)
plt.ylim(-0.1, 1.1)

plt.legend()
plt.xlabel("Magnetic field strength")
plt.ylabel("M_z/(mu * N)")
plt.savefig("target_graph.png")
