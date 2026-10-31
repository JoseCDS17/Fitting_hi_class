"""Project-wide constants, paths and the base cosmology passed to CLASS.

Units convention for the whole package:
    k, kB, kJ  -> h/Mpc
    P(k)       -> (Mpc/h)^3
"""
from pathlib import Path

''' Paths '''
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'

''' Physical constants '''
C_KM_S = 299792.458        # speed of light [km/s]
H0_OVER_C = 100 / C_KM_S   # H0/c [h/Mpc]

''' Base cosmology (single source of truth for every CLASS run) '''
BASE_COSMO = {
    'output': 'mPk',
    'P_k_max_h/Mpc': 20,

    '100*theta_s': 1.04109,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,

    'A_s': 2.100e-9,
    'n_s': 0.9649,
}

''' Calibration of Background.from_theta_s. Valid only for BASE_COSMO: update H_LCDM (h from CLASS with
    w0 = -1, wa = 0) if it changes '''
H_LCDM = 0.6759982
OMEGA_R_H2 = 2.473e-5 * (1 + 0.2271 * 3.044)   # photons + massless neutrinos (T_cmb = 2.7255 K, Neff = 3.044)
Z_REC = 1088.78                                # redshift of recombination

''' Horndeski parametrizations available '''
PARAMETRIZATIONS = ('propto_omega', 'dust')
