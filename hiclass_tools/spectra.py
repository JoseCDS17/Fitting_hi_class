"""Running hi_class: GR (w0wa fluid) and modified-gravity power spectra.

The central object is `SpectraSet`: for one background (w0, wa) it stores the GR
spectrum and the MG spectra on a (cK, cB) grid, and knows how to compute, save
and load itself.
"""
import concurrent.futures
import itertools
import pickle
from dataclasses import dataclass, field

import numpy as np

from horndeski_template import Background, Horndeski
from horndeski_template.constants import PARAMETRIZATIONS

from .config import BASE_COSMO, DATA_DIR


''' CLASS parameters '''
def gravity_params(ck, cb, parametrization='propto_omega'):
    """hi_class 'gravity_model' and 'parameters_smg' entries for (ck, cb)."""
    if parametrization == 'propto_omega':
        return {'gravity_model': 'propto_omega',
                'parameters_smg': f'{ck}, {cb}, 0., 0., 1.'}
    if parametrization == 'dust':
        return {'gravity_model': 'eft_alphas_power_law',
                'parameters_smg': f'0., {ck}, {cb}, 0., 0., 3., 3., 0.'}
    raise ValueError(f"Unknown parametrization '{parametrization}'. Options: {PARAMETRIZATIONS}")


def class_params_gr(w0=-1.0, wa=0.0, cosmo=None):
    """GR run with a w0wa fluid as dark energy."""
    return {**(cosmo or BASE_COSMO),
            'Omega_Lambda': 0,
            'w0_fld': w0,
            'wa_fld': wa}


def class_params_mg(ck, cb, w0=-1.0, wa=0.0, parametrization='propto_omega', cosmo=None):
    """hi_class run with w0wa expansion and Horndeski perturbations."""
    return {**(cosmo or BASE_COSMO),
            'Omega_Lambda': 0,
            'Omega_fld': 0,
            'Omega_smg': -1,
            'expansion_model': 'wowa',
            'expansion_smg': f'0.5, {w0}, {wa}',
            **gravity_params(ck, cb, parametrization)}


''' Running CLASS '''
def compute_pk(params, k, z=0.0):
    """Linear P(k) [(Mpc/h)^3] at k [h/Mpc]. Returns (pk, h, Omega_m0)."""
    from classy import Class

    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
        h = cosmo.h()
        pk = np.array([cosmo.pk(ki * h, z) for ki in k]) * h**3
        return pk, h, cosmo.Omega0_m()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def _error_message(e):
    msg = str(e).strip()
    return msg.splitlines()[-1] if msg else type(e).__name__


def _mg_worker(ck, cb, w0, wa, parametrization, cosmo, k, z):
    try:
        pk, _, _ = compute_pk(class_params_mg(ck, cb, w0, wa, parametrization, cosmo), k, z)
        return ck, cb, pk, None
    except Exception as e:
        return ck, cb, None, _error_message(e)


''' Spectra container '''
@dataclass
class SpectraSet:
    k: np.ndarray
    pk_gr: np.ndarray
    pk_mg: dict                      # (ck, cb) -> P(k)
    w0: float = -1.0
    wa: float = 0.0
    parametrization: str = 'propto_omega'
    z: float = 0.0
    h: float = None
    Omega_m0: float = None
    cosmo: dict = field(default_factory=lambda: dict(BASE_COSMO))
    failed: dict = field(default_factory=dict)  # (ck, cb) -> error message

    @classmethod
    def compute(cls, k, cK, cB, w0=-1.0, wa=0.0, parametrization='propto_omega',
                cosmo=None, z=0.0, workers=None, verbose=True):
        """Run GR + every (ck, cb) MG model in parallel (`workers=None` uses all cores)."""
        cosmo = dict(cosmo or BASE_COSMO)
        k = np.asarray(k, dtype=float)

        pk_gr, h, Omega_m0 = compute_pk(class_params_gr(w0, wa, cosmo), k, z)
        if verbose:
            print(f"[w0={w0:.3g}, wa={wa:.3g}] GR done. Launching {len(cK) * len(cB)} MG models...")

        pk_mg, failed = {}, {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_mg_worker, ck, cb, w0, wa, parametrization, cosmo, k, z)
                       for ck, cb in itertools.product(cK, cB)]
            for future in concurrent.futures.as_completed(futures):
                ck, cb, pk, error = future.result()
                if error is None:
                    pk_mg[(ck, cb)] = pk
                else:
                    failed[(ck, cb)] = error
                    if verbose:
                        print(f"  Failed ck={ck:.3g}, cb={cb:.3g}: {error}")

        spectra = cls(k=k, pk_gr=pk_gr, pk_mg=dict(sorted(pk_mg.items())), w0=w0, wa=wa,
                      parametrization=parametrization, z=z, h=h, Omega_m0=Omega_m0,
                      cosmo=cosmo, failed=failed)
        if verbose:
            print(f"  {len(pk_mg)} MG models computed, {len(failed)} failed.")
            spectra.report_scales()
        return spectra

    # Grid
    @property
    def cK(self):
        return np.array(sorted({ck for ck, _ in self.pk_mg}))

    @property
    def cB(self):
        return np.array(sorted({cb for _, cb in self.pk_mg}))

    def cB_for(self, ck):
        """cB values successfully computed for a given ck."""
        return np.array(sorted(cb for c, cb in self.pk_mg if c == ck))

    def __contains__(self, key):
        return key in self.pk_mg

    # Physics
    @property
    def background(self):
        return Background(w0=self.w0, wa=self.wa, Omega_m0=self.Omega_m0)

    def model(self, ck, cb):
        return Horndeski(ck, cb, self.background, self.parametrization)

    def scales(self, ck, cb, N=0.0):
        """(kB, kJ) in h/Mpc."""
        m = self.model(ck, cb)
        return m.kB(N), m.kJ(N)

    def deviation(self, ck, cb):
        """P_MG / P_GR - 1."""
        return self.pk_mg[(ck, cb)] / self.pk_gr - 1

    def report_scales(self, k_min=None, k_max=None):
        """Print the models whose Jeans scale falls outside [k_min, k_max] (default: k range)."""
        k_min = self.k[0] if k_min is None else k_min
        k_max = self.k[-1] if k_max is None else k_max
        outside = [(ck, cb, self.scales(ck, cb)[1]) for ck, cb in self.pk_mg]
        outside = [(ck, cb, kJ) for ck, cb, kJ in outside if not (k_min <= kJ <= k_max)]
        for ck, cb, kJ in outside:
            print(f"  Jeans scale outside [{k_min:.1e}, {k_max:.1e}]: kJ={kJ:.2e} for ck={ck:.3g}, cb={cb:.3g}")
        return outside

    # I/O (stored as plain dicts so the files don't depend on this package)
    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def save(self, name):
        path = spectra_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        return path

    @classmethod
    def load(cls, name):
        with open(spectra_path(name), 'rb') as f:
            return cls.from_dict(pickle.load(f))


def spectra_path(name):
    return DATA_DIR / 'spectra' / f'{name}.pkl'


def load_spectra(name):
    return SpectraSet.load(name)


def load_spectra_group(group):
    """Load every SpectraSet saved under data/spectra/<group>/ as {(w0, wa): SpectraSet}."""
    sets = {}
    for path in sorted((DATA_DIR / 'spectra' / group).glob('*.pkl')):
        s = SpectraSet.load(f'{group}/{path.stem}')
        sets[(s.w0, s.wa)] = s
    return dict(sorted(sets.items()))
