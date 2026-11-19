
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. Load your data
# ==========================================

with open("data.json", "r") as f:
    data = json.load(f)

E = np.array(data["Energy_barrier"])
r = np.array(data["time_ratio"])
p = np.array(data["probability"])

# ==========================================
# 2. Physical fitting function
# ==========================================


def thermal_model(X, A, B):
    energy, time = X
    return 1 - np.exp(-A * time * np.exp(-B * energy))


# ==========================================
# 3. Fit all data points
# ==========================================


popt, pcov = curve_fit(
    thermal_model,
    (E, r),
    p,
    p0=[1.0, 1.0],
    maxfev=20000
)

A, B = popt
errors = np.sqrt(np.diag(pcov))

p_fit = thermal_model((E, r), A, B)

mse = np.mean((p - p_fit)**2)
ss_res = np.sum((p - p_fit)**2)
ss_tot = np.sum((p - np.mean(p))**2)
r_squared = 1 - ss_res / ss_tot

print("FITTED PARAMETERS")
print("-------------------------")
print(f"A = {A:.8f} +/- {errors[0]:.8f}")
print(f"B = {B:.8f} +/- {errors[1]:.8f}")
print(f"MSE = {mse:.8e}")
print(f"R² = {r_squared:.8f}")

print("\nFitted function:")
print(f"p(E,r) = 1 - exp(-{A:.6f} * r * exp(-{B:.6f} * E))")

# ==========================================
# 4. Generate fitted surface
# ==========================================

E_grid = np.linspace(E.min(), E.max(), 100)
r_grid = np.linspace(r.min(), r.max(), 100)

EE, RR = np.meshgrid(E_grid, r_grid)

PP = thermal_model((EE, RR), A, B)

# ==========================================
# 5. Plot observed data and fitted surface
# ==========================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    EE, RR, PP,
    cmap="viridis",
    alpha=0.7,
    edgecolor="none"
)

ax.scatter(
    E, r, p,
    color="red",
    s=20,
    label="Simulation data"
)

ax.set_xlabel("Energy Barrier")
ax.set_ylabel("Time Ratio")
ax.set_zlabel("Probability")

ax.set_title("Thermal Néel Model: Nonlinear Fit")

ax.view_init(elev=25, azim=-45)

fig.colorbar(surface, ax=ax, shrink=0.65,
             label="Fitted probability")

ax.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 6. Plot residuals
# ==========================================

residuals = p - p_fit

plt.figure(figsize=(9, 5))
plt.scatter(E, residuals, c=r, cmap="viridis")
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Energy Barrier")
plt.ylabel("Residual (observed - fitted)")
plt.title("Fitting Residuals")
plt.colorbar(label="Time Ratio")
plt.tight_layout()
plt.savefig("fit.png")
