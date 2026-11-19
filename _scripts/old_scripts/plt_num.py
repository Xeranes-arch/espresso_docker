import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import expit
from scipy.integrate import simpson


# ============================================================
# Thermally averaged Stoner-Wohlfarth model
#
# Randomly oriented uniaxial particles:
#
#   E(phi) = K V sin^2(phi)
#          - mu0 * mu * H * cos(phi - theta)
#
# The two lowest stable minima are thermally populated according
# to their Boltzmann weights, and the result is averaged over
# random easy-axis orientations p(theta) = sin(theta).
# ============================================================


# ------------------------------------------------------------
# 1. Physical parameters
# ------------------------------------------------------------
kB = 1            # Boltzmann constant [J/K]
mu0 = 1          # vacuum permeability [H/m]

Lambda = 0.1

KV = 5                     # uniaxial anisotropy [J/m^3]
V = np.pi/6                  # particle volume [m^3]
K = KV / V
T = 1              # temperature [K]

mu = np.sqrt(Lambda * 4*np.pi * kB*T/mu0)


Ms = mu/V                 # magnetic moment [A m^2]

# Anisotropy field
Hk = 2.0 * K / (mu0 * Ms)     # [A/m]

# Dimensionless anisotropy parameter
sigma = K * V / (kB * T)

print(f"Magnetic moment mu = {mu:.3e} A m^2")
print(f"Anisotropy field Hk = {Hk:.3e} A/m")
print(f"Reduced anisotropy sigma = KV/(kBT) = {sigma:.3f}")


# ------------------------------------------------------------
# 2. Stoner-Wohlfarth energy
# ------------------------------------------------------------

def energy(phi, theta, h):
    """
    Dimensionless SW energy E/(KV).

    phi   : magnetization angle relative to easy axis
    theta : easy-axis angle relative to applied field
    h     : H/Hk
    """

    return np.sin(phi)**2 - 2.0 * h * np.cos(phi - theta)


def denergy(phi, theta, h):
    """
    First derivative of dimensionless SW energy.
    """

    return (
        np.sin(2.0 * phi)
        + 2.0 * h * np.sin(phi - theta)
    )


def ddenergy(phi, theta, h):
    """
    Second derivative of dimensionless SW energy.
    """

    return (
        2.0 * np.cos(2.0 * phi)
        + 2.0 * h * np.cos(phi - theta)
    )


# ------------------------------------------------------------
# 3. Find all stationary points and retain minima
# ------------------------------------------------------------

def find_minima(theta, h, ngrid=1000):
    """
    Find all local minima of the SW energy in 0 <= phi < 2*pi.

    Returns
    -------
    minima : list of phi values
    """

    phi_grid = np.linspace(0.0, 2.0 * np.pi, ngrid + 1)
    f_grid = denergy(phi_grid, theta, h)

    roots = []

    # Locate sign changes in dE/dphi
    for i in range(ngrid):
        f1 = f_grid[i]
        f2 = f_grid[i + 1]

        if f1 == 0.0:
            roots.append(phi_grid[i])

        elif f1 * f2 < 0.0:
            root = brentq(
                lambda phi: denergy(phi, theta, h),
                phi_grid[i],
                phi_grid[i + 1]
            )
            roots.append(root)

    # Remove duplicate roots
    roots = np.array(roots)

    if len(roots) == 0:
        return []

    roots = np.mod(roots, 2.0 * np.pi)
    roots = np.unique(np.round(roots, decimals=10))

    # Keep only local minima: d2E/dphi2 > 0
    minima = [
        phi for phi in roots
        if ddenergy(phi, theta, h) > 0.0
    ]

    # Sort by energy
    minima.sort(key=lambda phi: energy(phi, theta, h))

    return minima


# ------------------------------------------------------------
# 4. Thermal population of SW minima
# ------------------------------------------------------------

def thermal_SW_magnetization(theta, h):
    """
    Equilibrium magnetization projected onto the applied field
    for one easy-axis orientation theta.

    The two lowest SW minima are thermally populated.
    """

    minima = find_minima(theta, h)

    if len(minima) == 0:
        return 0.0

    # Take at most the two lowest minima
    minima = minima[:2]

    # Dimensionless energies
    eps = np.array([
        energy(phi, theta, h)
        for phi in minima
    ])

    # Projected magnetic moments, normalized by mu
    projections = np.array([
        np.cos(phi - theta)
        for phi in minima
    ])

    # Boltzmann weights:
    #
    # w_j ~ exp[-sigma * epsilon_j]
    #
    # Subtract the minimum energy for numerical stability.
    x = -sigma * (eps - np.min(eps))

    weights = np.exp(x)
    weights /= np.sum(weights)

    return np.sum(weights * projections)


# ------------------------------------------------------------
# 5. Average over random easy-axis orientations
# ------------------------------------------------------------

def SW_ensemble(h, ntheta=250):
    """
    Randomly oriented ensemble.

    p(theta) = sin(theta), 0 <= theta <= pi/2.

    Returns M/Ms.
    """

    theta = np.linspace(0.0, np.pi / 2.0, ntheta)

    m_theta = np.array([
        thermal_SW_magnetization(th, h)
        for th in theta
    ])

    # Normalize p(theta) = sin(theta).
    #
    # Integral_0^(pi/2) sin(theta) dtheta = 1,
    # so no additional normalization is required.
    return simpson(
        m_theta * np.sin(theta),
        x=theta
    )


# ------------------------------------------------------------
# 6. Langevin and tanh reference functions
# ------------------------------------------------------------

def langevin(x):
    """
    Langevin function:
        L(x) = coth(x) - 1/x

    A small-x expansion is used to avoid numerical problems.
    """

    x = np.asarray(x)

    result = np.empty_like(x, dtype=float)

    small = np.abs(x) < 1e-4

    # L(x) = x/3 - x^3/45 + 2x^5/945 - ...
    result[small] = (
        x[small] / 3.0
        - x[small]**3 / 45.0
        + 2.0 * x[small]**5 / 945.0
    )

    result[~small] = (
        1.0 / np.tanh(x[~small])
        - 1.0 / x[~small]
    )

    return result


# ------------------------------------------------------------
# 7. Field range
# ------------------------------------------------------------

# Reduced field h = H/Hk
h_values = np.linspace(-3.0, 3.0, 301)

# Calculate SW ensemble
print("\nCalculating thermally averaged SW curve...")

M_SW = np.array([
    SW_ensemble(h)
    for h in h_values
])


# ------------------------------------------------------------
# 8. Compare using the same dimensionless field
# ------------------------------------------------------------
#
# The natural thermal field variable is
#
#     x = mu0 * mu * H / (kB*T)
#
# Since
#
#     h = H/Hk
#
# and
#
#     Hk = 2K/(mu0*Ms),
#
# we have
#
#     x = 2 sigma h
#
# where sigma = KV/(kBT).
# ------------------------------------------------------------

x_values = 2.0 * sigma * h_values

M_Langevin = langevin(x_values)
M_tanh = np.tanh(x_values)


# ------------------------------------------------------------
# 9. Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    h_values,
    M_SW,
    linewidth=2.5,
    label="Thermal SW, random orientations"
)

plt.plot(
    h_values,
    M_Langevin,
    "--",
    linewidth=2,
    label="Langevin"
)

plt.plot(
    h_values,
    M_tanh,
    ":",
    linewidth=2.5,
    label="tanh"
)

plt.axhline(0.0, color="black", linewidth=0.7)
plt.axvline(0.0, color="black", linewidth=0.7)

plt.xlabel(r"Reduced field $h=H/H_K$")
plt.ylabel(r"Normalized magnetization $M/M_s$")

plt.title(
    rf"Thermally averaged SW ensemble, "
    rf"$KV/k_BT={sigma:.2f}$"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("bitch.png")
