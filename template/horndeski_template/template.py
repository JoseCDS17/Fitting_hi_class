"""Analytic template for the MG deviation P_MG/P_GR - 1."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .background import Background
from .horndeski import Horndeski

COEFFICIENTS_PATH = Path(__file__).parent / 'coefficients.json'   # distributed with the package


def tanh_kB_L(k, kB, kJ, A, L, s):
    """tanh_kB written as large-scale level L (k << kB) plus a step of height 2A (a = A, b = L + A)."""
    return L + A * (np.tanh(2 * np.log10(s * k / kB)) + 1)


@dataclass
class GlobalTemplate:
    """y = L + A [tanh(2 log10(s k / kB)) + 1] with 6 coefficients valid for any (w0, wa):
        A = m_A cB + y0_A,   s = m_s cB + y0_s,   L = l1 (1 + w0) + l2 wa  (L = 0 in LCDM)
    """
    A: dict                   # {'m': (value, error), 'y0': (value, error)}
    s: dict                   # {'m': (value, error), 'y0': (value, error)}
    L: dict                   # {'l1': (value, error), 'l2': (value, error)}
    parametrization: str
    model_name: str = 'tanh_kB_L'

    @classmethod
    def default(cls):
        """Template with the coefficients distributed with the package."""
        return cls(**json.loads(COEFFICIENTS_PATH.read_text()))

    def export(self):
        """Make this template the one returned by `default` (rewrites the versioned coefficients file)."""
        COEFFICIENTS_PATH.write_text(json.dumps(asdict(self), indent=2) + '\n')

    def params(self, cb, background):
        """Template parameters (A, L, s) at cb for a background."""
        return {'A': self.A['m'][0] * cb + self.A['y0'][0],
                'L': self.L['l1'][0] * (1 + background.w0) + self.L['l2'][0] * background.wa,
                's': self.s['m'][0] * cb + self.s['y0'][0]}

    def __call__(self, k, ck, cb, w0, wa):
        """Predicted P_MG/P_GR - 1 at k [h/Mpc] for (ck, cb) on (w0, wa), with the base cosmology of the fit."""
        return self.evaluate(k, ck, cb, Background.from_theta_s(w0, wa))

    def evaluate(self, k, ck, cb, background):
        """Predicted P_MG/P_GR - 1 at k [h/Mpc] for (ck, cb) on a given background."""
        params = self.params(cb, background)
        if params['s'] <= 0:
            raise ValueError(f"s must be positive, but s={params['s']:.4g} for cb={cb}")
        kB = Horndeski(ck, cb, background, self.parametrization).kB()
        return tanh_kB_L(np.asarray(k), kB, None, **params)
