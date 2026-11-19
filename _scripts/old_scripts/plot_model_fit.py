
import json
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import brentq

m = 1
kT = 1
kB = 1
T = 1
mu_0 = 1
sigma = 1
Lambda = 0.1

V = np.pi/6 * sigma**3
m = np.sqrt(Lambda * 4*np.pi * sigma ** 3 * kT/mu_0)
M_s = m/V

KV = 5
K = KV / V
# region fn


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
    x = -KV * (eps - np.min(eps))

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


def langevin(xi):
    return np.cosh(xi) / np.sinh(xi) - 1 / xi


def mmft_langevin(M, M_s, mu0, m, H, lambda_, kT):
    xi_eff = mu0 * m * (H + lambda_ * M) / (kT)
    return M_s * langevin(xi_eff) - M


def solve_mmft(M_s, mu0, m, H, lambda_, kT, initial_guess=0.0):
    M, = fsolve(mmft_langevin, initial_guess,
                args=(M_s, mu0, m, H, lambda_, kT))
    return M

# endregion fn


with open('model_fit.json', 'r') as f:
    data = json.load(f)

alphas0 = data["entry_001"]['alphas']
alphas1 = data["entry_002"]['alphas']
alphas2 = data["entry_003"]['alphas']
alphas3 = data["entry_004"]['alphas']

dipms0 = data["entry_001"]['dipms']
dipms1 = data["entry_002"]['dipms']
dipms2 = data["entry_003"]['dipms']
dipms3 = data["entry_004"]['dipms']

plt.scatter(alphas0, dipms0, marker="s", c="none",
            edgecolor="red", s=50, label="ali, no int")
plt.scatter(alphas1, dipms1, marker="o", c="none",
            edgecolor="red", s=50, label=r"ali, $\lambda$ = 0.15")
plt.scatter(alphas2, dipms2, marker="s", c="none",
            edgecolor="purple", s=50, label="rea, no int")
plt.scatter(alphas3, dipms3, marker="o", c="none",
            edgecolor="purple", s=50, label=r"rea, $\lambda$ = 0.15")

# region plot initial suszeptibility

tanh = []
maxv = max(alphas1)
alphas = np.linspace(0.001, maxv, 100)
for alpha in alphas:
    H = alpha * kT / (mu_0 * m)
    xi = m * H

    tanh.append(np.tanh(xi))

hs = [alpha/(2*KV) for alpha in alphas]
M_SW = np.array([SW_ensemble(h) for h in hs])

plt.plot(alphas, tanh, label="tanh",
         linestyle="--", color="red")
plt.plot(alphas, M_SW, label="rea model",
         linestyle="--", color="purple")

plt.xlabel(r"$\xi$")
plt.ylabel("m")
plt.legend()
plt.grid(alpha=0.5)
plt.savefig(f"Model_fit.png")

# endregion
