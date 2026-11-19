import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
with open('_data/mag_response/hyst_data.json', 'r') as file:
    data = json.load(file)

fig, ax = plt.subplots()

alphass = []
dipmss = []
for key, entry in data.items():
    alphass.append(entry["alphas"])
    dipmss.append(entry["dipms"])

alphas0, alphas1, alphas2 = alphass
dipms0, dipms1, dipms2 = dipmss

plt.plot(alphas0, dipms0, c="black", linewidth=1, label="solid")
plt.plot(alphas1, dipms1, c="black", linewidth=1,
         linestyle="-.", label="rea-ell")
plt.plot(alphas2, dipms2, c="black", linewidth=1,
         linestyle=":", label="ali-ell")

x_lim = 15
plt.legend(title="geometry:")


# region plot config
# Move the left and bottom spines to the center
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust tick positions
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Places label at the top end
plt.ylabel("m", rotation=0)
plt.xlabel(r"$\xi$")  # Places label at the right end

ax.xaxis.set_label_coords(1.05, 0.525)
ax.yaxis.set_label_coords(0.5, 1.05)

plt.xticks(np.linspace(-x_lim, x_lim, 4))  # Custom x-axis ticks
plt.yticks([-1, -0.5, 0.5, 1])  # Custom y-axis ticks

plt.savefig("plot.png")
# endregion
