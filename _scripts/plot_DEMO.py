import json
import matplotlib.pyplot as plt

json_file = "data.json"

# Load data
with open(json_file, "r") as f:
    data = json.load(f)

distance, probability = list(data.values())[:2]

# Plot
plt.figure(figsize=(7, 5))
plt.plot(distance, probability, "o-", markersize=4)

plt.xlabel("Energy barrier")
plt.ylabel("Probability")
plt.title("Probability vs Energy barrier")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_DEMO.png")
