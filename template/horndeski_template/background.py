"""Analytic w0wa background (matter + dark energy, radiation neglected).

Time variable N = ln(a). Densities are normalised so that H(N=0) = 1,
i.e. every H is in units of H0.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .constants import H_LCDM, OMEGA_B_H2, OMEGA_CDM_H2, OMEGA_R_H2, Z_REC

W_M = 0  # matter equation of state
OMEGA_M_H2 = OMEGA_B_H2 + OMEGA_CDM_H2


def _distance_to_recombination(h, w0, wa):
    """Comoving distance to Z_REC in units of c / (100 km/s/Mpc)."""
    omega_de_h2 = h**2 - OMEGA_M_H2 - OMEGA_R_H2

    def integrand(x):        # x = ln(1 + z)
        one_plus_z = np.exp(x)
        de = np.exp(3 * (1 + w0 + wa) * x - 3 * wa * (1 - 1 / one_plus_z))
        return one_plus_z / np.sqrt(OMEGA_M_H2 * one_plus_z**3 + OMEGA_R_H2 * one_plus_z**4 + omega_de_h2 * de)

    return quad(integrand, 0, np.log(1 + Z_REC), epsabs=1e-10, epsrel=1e-10, limit=200)[0]


@dataclass(frozen=True)
class Background:
    w0: float
    wa: float
    Omega_m0: float

    @classmethod
    def from_theta_s(cls, w0, wa):
        """Build the background without CLASS: with the early universe fixed, theta_s fixed means the
        distance to recombination of the LCDM calibration, which gives h and Omega_m0 = omega_m / h^2."""
        target = _distance_to_recombination(H_LCDM, -1.0, 0.0)
        h = brentq(lambda h: _distance_to_recombination(h, w0, wa) - target, 0.4, 1.1)
        return cls(w0=w0, wa=wa, Omega_m0=OMEGA_M_H2 / h**2)

    @property
    def Omega_DE0(self):
        return 1 - self.Omega_m0

    # Densities (in units of H0^2 / (8 pi G))
    def rho_m(self, N):
        return 3 * self.Omega_m0 * np.exp(-3 * N)

    def wDE(self, N):
        return self.w0 + self.wa * (1 - np.exp(N))

    def rho_DE(self, N):
        return 3 * self.Omega_DE0 * np.exp(-3 * (1 + self.w0 + self.wa) * N - 3 * self.wa * (1 - np.exp(N)))

    # Dependent variables
    def H(self, N):
        return np.sqrt((self.rho_m(N) + self.rho_DE(N)) / 3)

    def Omega_m(self, N):
        return self.rho_m(N) / (3 * self.H(N)**2)

    def Omega_DE(self, N):
        return self.rho_DE(N) / (3 * self.H(N)**2)

    def w_eff(self, N):
        return (W_M * self.rho_m(N) + self.wDE(N) * self.rho_DE(N)) / (self.rho_m(N) + self.rho_DE(N))
