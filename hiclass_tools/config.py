"""Paths and the base cosmology passed to CLASS.

Units convention for the whole package:
    k, kB, kJ  -> h/Mpc
    P(k)       -> (Mpc/h)^3
"""
from pathlib import Path

from horndeski_template.constants import OMEGA_B_H2, OMEGA_CDM_H2

''' Paths '''
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'

''' Base cosmology (single source of truth for every CLASS run) '''
BASE_COSMO = {
    'output': 'mPk',
    'P_k_max_h/Mpc': 20,

    '100*theta_s': 1.04109,
    'omega_b': OMEGA_B_H2,
    'omega_cdm': OMEGA_CDM_H2,

    'A_s': 2.100e-9,
    'n_s': 0.9649,
}
