"""Compute GR + MG power spectra with hi_class and save them for the notebooks.

Edit the configuration block and run from the project root (hiclass_env activated):
    python scripts/compute_spectra.py

One background  -> data/spectra/<NAME>.pkl
Several         -> data/spectra/<NAME>/<background>.pkl, one file each. Backgrounds
                   already saved are skipped, so an interrupted grid resumes by
                   relaunching the script.

Every file stores all the parameters used (k, cK, cB, w0, wa, parametrization,
cosmology), so nothing needs to be remembered here.
"""
import time

import numpy as np

from hiclass_tools.spectra import SpectraSet, spectra_path

# ===================== Configuration =====================
NAME = 'fit_DE_grid'
PARAMETRIZATION = 'propto_omega'      # 'propto_omega' | 'dust'

# Grid of (w0, wa), keeping w0 + wa <= 0
BACKGROUNDS = [(w0, wa) for w0 in np.linspace(-1, -0.7, 10) for wa in np.linspace(-1, 0, 10)
               if w0 + wa <= 0]

k = np.logspace(-3, 1, 50)            # h/Mpc
cK = np.logspace(3, 6, 10)
cB = np.linspace(0.5, 1.5, 10)

WORKERS = 4                        # parallel processes (None: all cores)
# =========================================================


def file_name(w0, wa):
    if len(BACKGROUNDS) == 1:
        return NAME
    return f'{NAME}/w0_{w0:+.3f}_wa_{wa:+.3f}'


if __name__ == '__main__':
    for w0, wa in BACKGROUNDS:
        name = file_name(w0, wa)
        if spectra_path(name).exists():
            print(f'[skip] {name} already exists (delete the file to recompute)')
            continue

        t0 = time.time()
        spectra = SpectraSet.compute(k, cK, cB, w0, wa, PARAMETRIZATION, workers=WORKERS)
        print(f'[saved] {spectra.save(name)} ({time.time() - t0:.0f} s)\n')
