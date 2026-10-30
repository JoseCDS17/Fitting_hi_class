"""Horndeski alpha-functions and characteristic scales (Jeans and braiding).

Both alpha_K and alpha_B share the time dependence fixed by the parametrization:
    'propto_omega': alpha_X = c_X * Omega_DE(N)
    'dust'        : alpha_X = c_X * a^3
"""
from dataclasses import dataclass

import numpy as np

from .background import Background, W_M
from .config import H0_OVER_C, PARAMETRIZATIONS


def gravity_params(ck, cb, parametrization='propto_omega'):
    """hi_class 'gravity_model' and 'parameters_smg' entries for (ck, cb)."""
    if parametrization == 'propto_omega':
        return {'gravity_model': 'propto_omega',
                'parameters_smg': f'{ck}, {cb}, 0., 0., 1.'}
    if parametrization == 'dust':
        return {'gravity_model': 'eft_alphas_power_law',
                'parameters_smg': f'0., {ck}, {cb}, 0., 0., 3., 3., 0.'}
    raise ValueError(f"Unknown parametrization '{parametrization}'. Options: {PARAMETRIZATIONS}")


@dataclass(frozen=True)
class Horndeski:
    ck: float
    cb: float
    background: Background
    parametrization: str = 'propto_omega'

    def __post_init__(self):
        if self.parametrization not in PARAMETRIZATIONS:
            raise ValueError(f"Unknown parametrization '{self.parametrization}'. Options: {PARAMETRIZATIONS}")

    def class_params(self):
        return gravity_params(self.ck, self.cb, self.parametrization)

    # Horndeski variables
    def _time_dependence(self, N):
        if self.parametrization == 'propto_omega':
            return self.background.Omega_DE(N)
        return np.exp(3 * N)

    def _dlog_time_dependence(self, N):
        """d ln(time dependence) / dN."""
        if self.parametrization == 'propto_omega':
            return 3 * (self.background.w_eff(N) - self.background.wDE(N))
        return 3

    def alpha_b(self, N=0.0):
        return self.cb * self._time_dependence(N)

    def dalpha_b(self, N=0.0):
        """d alpha_b / dN."""
        return self.alpha_b(N) * self._dlog_time_dependence(N)

    def alpha_k(self, N=0.0):
        return self.ck * self._time_dependence(N)

    # Complex variables
    def D(self, N=0.0):
        return 3/2 * self.alpha_b(N)**2 + self.alpha_k(N)

    def cs2(self, N=0.0):
        bg = self.background
        ab = self.alpha_b(N)
        w_eff = bg.w_eff(N)
        return 1 / self.D(N) * ((2 - ab) * (3/2 * (1 + w_eff) + ab / 2) - (1 + W_M) / bg.H(N)**2 * bg.rho_m(N)
                                + self.dalpha_b(N))

    # Characteristic scales [h/Mpc]
    def kJ(self, N=0.0):
        """Jeans scale."""
        aH = np.exp(N) * self.background.H(N)
        return aH / np.sqrt(self.cs2(N)) * H0_OVER_C

    def kB(self, N=0.0):
        """Braiding scale."""
        bg = self.background
        aH = np.exp(N) * bg.H(N)
        return np.sqrt(self.D(N) / self.alpha_b(N)**2 * bg.Omega_DE(N) * (1 - bg.wDE(N))
                       + 9/2 * bg.Omega_m(N)) * aH * H0_OVER_C
