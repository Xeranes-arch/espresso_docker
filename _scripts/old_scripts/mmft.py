import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def langevin(xi):
    return np.cosh(xi) / np.sinh(xi) - 1 / xi


def mmft_langevin(M, M_s, mu0, m, H, lambda_, kT):
    xi_eff = mu0 * m * (H + lambda_ * M) / (kT)
    return M_s * langevin(xi_eff) - M


def solve_mmft(M_s, mu0, m, H, lambda_, kT, initial_guess=0.0):
    M, = fsolve(mmft_langevin, initial_guess,
                args=(M_s, mu0, m, H, lambda_, kT))
    return M


lambda_ = 0.5
# lambda_ = 1

# Example usage
sigma = 1
V = np.pi/6 + sigma**3
mu0 = 1  # T·m/A
m = 1  # A·m² (magnetic moment)
kT = 1

m = np.sqrt(lambda_ * 4*np.pi * sigma ** 3 * kT/mu0)
M_s = m/V

### alt#########
cells = 3**3
nr_of_part = 2*cells
V_sys = cells * np.sqrt(2)**3
chi_L = 2 * m**2 * nr_of_part / (3 * V_sys)

M_list = []
m_nonint = []
m_int_paper = []
alphas = np.linspace(0.001, 15, 50)
for alpha in alphas:
    H = alpha * kT / (mu0 * m)
    M = solve_mmft(M_s, mu0, m, H, lambda_, kT)
    M_list.append(M/M_s)

    xi = m * H
    m_nonint.append((np.cosh(xi)/np.sinh(xi) - 1/xi))

    xi_eff = xi + chi_L * langevin(xi)
    m_int_paper.append(langevin(xi_eff))

plt.plot(alphas, m_nonint, label="Langevin")
plt.plot(alphas, M_list, label="Weiß")
plt.plot(alphas, m_int_paper, label="Paper")
plt.legend()
plt.savefig("mmft.png")
