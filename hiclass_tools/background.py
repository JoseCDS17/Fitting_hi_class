"""Analytic w0wa background (matter + dark energy, radiation neglected).

Time variable N = ln(a). Densities are normalised so that H(N=0) = 1,
i.e. every H is in units of H0.
"""
from dataclasses import dataclass

import numpy as np

W_M = 0  # matter equation of state


@dataclass(frozen=True)
class Background:
    w0: float
    wa: float
    Omega_m0: float

    @classmethod
    def from_class(cls, w0, wa, cosmo=None):
        """Build the background with Omega_m0 derived by CLASS from `cosmo`."""
        from classy import Class
        from .spectra import class_params_gr

        params = class_params_gr(w0, wa, cosmo)
        params['output'] = ''  # background + thermodynamics only
        params.pop('P_k_max_h/Mpc', None)

        c = Class()
        c.set(params)
        try:
            c.compute()
            return cls(w0=w0, wa=wa, Omega_m0=c.Omega0_m())
        finally:
            c.struct_cleanup()
            c.empty()

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
