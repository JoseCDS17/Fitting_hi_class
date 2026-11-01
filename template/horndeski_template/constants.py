"""Physical constants and the base cosmology of the template.

Units convention for the whole package:
    k, kB, kJ  -> h/Mpc
"""

''' Physical constants '''
C_KM_S = 299792.458        # speed of light [km/s]
H0_OVER_C = 100 / C_KM_S   # H0/c [h/Mpc]

''' Base cosmology (physical densities used for the template and for every CLASS run) '''
OMEGA_B_H2 = 0.02237
OMEGA_CDM_H2 = 0.1200

''' Calibration of Background.from_theta_s. Valid only for the base cosmology: update H_LCDM (h from CLASS
    with w0 = -1, wa = 0) if it changes '''
H_LCDM = 0.6759982
OMEGA_R_H2 = 2.473e-5 * (1 + 0.2271 * 3.044)   # photons + massless neutrinos (T_cmb = 2.7255 K, Neff = 3.044)
Z_REC = 1088.78                                # redshift of recombination

''' Horndeski parametrizations available '''
PARAMETRIZATIONS = ('propto_omega', 'dust')
