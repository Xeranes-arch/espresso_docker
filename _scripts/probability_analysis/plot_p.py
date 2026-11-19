import json
import numpy as np
import matplotlib.pyplot as plt

# Load your JSON file
with open("data.json", "r") as f:
    data = json.load(f)

# Extract data
energy = np.array(data["Energy_barrier"])
time_ratio = np.array(data["time_ratio"])
probability = np.array(data["probability"])

# Create 3D scatter plot
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# Reshape data into a 10 x 10 grid
X = energy.reshape(10, 10)
Y = time_ratio.reshape(10, 10)
Z = probability.reshape(10, 10)

surf = ax.plot_surface(
    X, Y, Z,
    cmap="viridis",
    edgecolor="none"
)

ax.set_xlabel("Energy Barrier")
ax.set_ylabel("Time Ratio")
ax.set_zlabel("Probability")
ax.set_title("3D Probability Distribution")

fig.colorbar(surf, ax=ax, label="Probability", shrink=0.7)

ax.view_init(elev=25, azim=-75)
plt.tight_layout()
plt.savefig("p.png")
