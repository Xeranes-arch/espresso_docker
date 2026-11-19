import numpy as np
import matplotlib.pyplot as plt


def lj_potential(r, epsilon, sigma, cutoff, shift=0.0, off=0):
    """
    ESPResSo Lennard-Jones potential with an explicit shift.

    For r < cutoff:

        V(r) = 4 * epsilon * (
            (sigma / r)**12
            - (sigma / r)**6
            + shift
        )

    For r >= cutoff:

        V(r) = 0
    """

    V = np.zeros_like(r, dtype=float)

    mask = r < cutoff

    V[mask] = epsilon * (
        (sigma / (r[mask]-off))**12
        - (sigma / (r[mask]-off))**6
        + shift
    )

    return V


def lj_zero_crossings(sigma, shift, r_off):
    discriminant = 1.0 - 4.0 * shift

    if discriminant < 0:
        return np.array([])

    x1 = (1.0 + np.sqrt(discriminant)) / 2.0
    x2 = (1.0 - np.sqrt(discriminant)) / 2.0

    return r_off + sigma / np.array([x1, x2])**(1.0 / 6.0)


# ============================================================
# ESPResSo parameters
# ============================================================

epsilon = 1
sigma = 1
cutoff = 4 * sigma
shift = -1

off = 0.077
# ============================================================
# Radial coordinate
# ============================================================

r = np.linspace(
    0.8 * sigma,
    1.05 * cutoff,
    3000,
)


# ============================================================
# Potential
# ============================================================

V = lj_potential(
    r,
    epsilon=epsilon,
    sigma=sigma,
    cutoff=cutoff,
    shift=shift,
    off=off
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    r,
    V,
    linewidth=2,
    label=(
        rf"$\epsilon={epsilon:g}$, "
        rf"$\sigma={sigma:g}$, "
        rf"$r_c={cutoff:g}$, "
        rf"$shift={shift:g}$"
    ),
)

# Cutoff
ax.axvline(
    cutoff,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=rf"$r_\mathrm{{cut}}={cutoff:g}$",
)

# Zero-energy line
ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    alpha=0.5,
)

# ============================================================
# LJ minimum
# ============================================================

r_minimum = 2.0 ** (1.0 / 6.0) * sigma

V_minimum = lj_potential(
    np.array([r_minimum]),
    epsilon,
    sigma,
    cutoff,
    shift,
)[0]

ax.scatter(
    r_minimum,
    V_minimum,
    color="black",
    zorder=5,
)

ax.annotate(
    rf"$r_\mathrm{{min}}={r_minimum:.3g}$"
    "\n"
    rf"$V_\mathrm{{min}}={V_minimum:.3g}$",
    xy=(r_minimum, V_minimum),
    xytext=(1.4 * sigma, 0.5 * V_minimum),
    arrowprops=dict(arrowstyle="->"),
)


# ============================================================
# Formatting
# ============================================================

ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$V(r)$")
ax.set_ylim(-5, 10)

ax.set_title("ESPResSo Lennard-Jones potential")

ax.grid(alpha=0.25)
ax.legend()

plt.tight_layout()
plt.savefig("lj.png")

x = lj_zero_crossings(sigma, shift, off)
print(x)
