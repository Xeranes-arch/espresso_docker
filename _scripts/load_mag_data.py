import numpy as np
import matplotlib.pyplot as plt


ratios = ["2.0", "3.0", "4.0", "5.0", "6.0"]
# ratios = ["2.0"]
anis = [3, 4, 5, 6]
# anis = [6]
# anis = [3]

fig, ax = plt.subplots(figsize=(10, 6))

for ratio in ratios:
    for ani in anis:
        filename = f"/workspace/_data/mag_response/mag_response_data_ratio{ratio}_ani{float(ani)}.npz"
        loaded = np.load(filename)

        # keys = list(loaded.keys())
        # for key in keys:
        #     print(key, loaded[key])
        # exit()

        print("\n", filename)

        dipm_means = np.array(loaded["dipm_means"])
        stds = np.array(loaded["stds"])
        alphas = np.array(loaded["alphas"])
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

plt.xlim(-0.5, max(alphas) + 1)
plt.ylim(-0.1, 1.1)

plt.legend()
plt.xlabel(r"Magnetic field strength $\alpha$ in kT")
plt.ylabel(r"$\langle M_z \rangle / M_{sat} N$")
plt.savefig("target_graph.png")
