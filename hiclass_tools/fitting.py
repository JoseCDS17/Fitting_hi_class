"""Fitting pipeline for the MG deviation P_MG/P_GR - 1.

Three nested steps:
  1. Fit every spectrum (ck, cb) with a `TemplateModel`            -> params a, b, s
  2. For each ck, fit every param linearly in cB: p = m cB + y0     -> m(ck), y0(ck)
  3. Average m and y0 over cK (simple mean)                         -> final `Template`
"""
import pickle
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
from lmfit import Model

from .config import DATA_DIR
from .horndeski import Horndeski
from .background import Background


''' Template models '''
def tanh_kBkJ(k, kB, kJ, a, b, s):
    """Transition centred at sqrt(kB kJ), width log10(kB/kJ)."""
    width = np.maximum(np.log10(kB / kJ), 1e-10)
    x = np.log10(np.maximum(k, 1e-10) / np.sqrt(kB * kJ)) / width
    return a * np.tanh(s * x) + b


def tanh_kB(k, kB, kJ, a, b, s):
    """Transition centred at kB / s (kJ unused)."""
    return a * np.tanh(2 * np.log10(s * k / kB)) + b


@dataclass(frozen=True)
class TemplateModel:
    name: str
    func: Callable                 # f(k, kB, kJ, *params)
    params: tuple
    guess: Callable                # (y, cb) -> dict of initial values
    latex: str
    positive: tuple = ()           # params that must be > 0 when evaluating the template
    needs_kJ: bool = True          # template requires kB > kJ
    bounds: dict = field(default_factory=dict)   # param -> (min, max) used when fitting

    def __call__(self, k, kB, kJ, **params):
        return self.func(k, kB, kJ, **params)


MODELS = {m.name: m for m in [
    TemplateModel('tanh_kBkJ', tanh_kBkJ, ('a', 'b', 's'),
                  lambda y, cb: dict(a=y.max() / 2, b=y.max() / 2, s=1),
                  r'a \tanh\left[s\,\frac{\log_{10}(k/\sqrt{k_B k_J})}{\log_{10}(k_B/k_J)}\right] + b',
                  positive=('s',)),
    TemplateModel('tanh_kB', tanh_kB, ('a', 'b', 's'),
                  lambda y, cb: dict(a=y.max() / 2, b=y.max() / 2, s=1 / cb),
                  r'a \tanh\left[2\log_{10}(s\,k/k_B)\right] + b',
                  positive=('s',), needs_kJ=False, bounds={'s': (1e-10, np.inf)}),
]}


def get_model(model):
    return MODELS[model] if isinstance(model, str) else model


def relative_error(y, y_fit):
    """mean|y - y_fit| / mean|y|."""
    denom = np.mean(np.abs(y))
    return np.mean(np.abs(y - y_fit)) / denom if denom != 0 else np.nan


''' Weighted fits '''
def _clean(x, values, sigma):
    x, values = np.asarray(x, float), np.asarray(values, float)
    sigma = np.array([np.inf if s is None else s for s in sigma], dtype=float)
    ok = np.isfinite(x) & np.isfinite(values) & np.isfinite(sigma) & (sigma > 0)
    return x[ok], values[ok], sigma[ok]


@dataclass
class LinearFit:
    """Weighted linear fit p = m x + y0 (covariance scaled by chi2_red, as np.polyfit)."""
    m: float
    m_err: float
    y0: float
    y0_err: float
    x: np.ndarray             # data used in the fit
    values: np.ndarray
    sigma: np.ndarray
    chi2_red: float

    def __call__(self, x):
        return self.m * x + self.y0


def linear_fit(x, values, sigma):
    x, values, sigma = _clean(x, values, sigma)
    if len(x) < 2:
        raise ValueError(f"Need at least 2 valid points for a linear fit, got {len(x)}")

    (m, y0), cov = np.polyfit(x, values, 1, w=1 / sigma, cov=True if len(x) > 3 else 'unscaled')
    dof = len(x) - 2
    chi2_red = np.sum(((values - (m * x + y0)) / sigma)**2) / dof if dof > 0 else np.nan
    return LinearFit(m, np.sqrt(abs(cov[0, 0])), y0, np.sqrt(abs(cov[1, 1])), x, values, sigma, chi2_red)


@dataclass
class MeanFit:
    """Simple mean, with the standard error of the mean (scatter / sqrt(N)) as error."""
    value: float
    error: float
    x: np.ndarray             # data used in the mean
    values: np.ndarray
    sigma: np.ndarray         # error bars of each point (only for plotting and chi2_red)
    chi2_red: float           # >> 1 means the points are not compatible with a constant


def mean_fit(x, values, sigma):
    x, values, sigma = _clean(x, values, sigma)
    if len(x) == 0:
        raise ValueError("No valid points for the mean")

    mean = np.mean(values)
    dof = len(x) - 1
    error = np.std(values, ddof=1) / np.sqrt(len(x)) if dof > 0 else np.nan
    chi2_red = np.sum(((values - mean) / sigma)**2) / dof if dof > 0 else np.nan
    return MeanFit(mean, error, x, values, sigma, chi2_red)


''' Step 1: single spectrum '''
@dataclass
class SpectrumFit:
    ck: float
    cb: float
    k: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    params: dict              # name -> (value, stderr)
    err: float                # relative error of the fit


def fit_spectrum(spectra, ck, cb, model, k_min=None, k_max=None, method='leastsq',
                 guess=None, bounds=None):
    """Fit P_MG/P_GR - 1 of one (ck, cb) model. `guess`/`bounds` override the model defaults."""
    model = get_model(model)
    k = spectra.k
    mask = (k >= (k_min if k_min is not None else k[0])) & (k <= (k_max if k_max is not None else k[-1]))
    k, y = k[mask], spectra.deviation(ck, cb)[mask]
    kB, kJ = spectra.scales(ck, cb)

    lm = Model(model.func, independent_vars=['k', 'kB', 'kJ'])
    params = lm.make_params(**{**model.guess(y, cb), **(guess or {})})
    for name, (lo, hi) in {**model.bounds, **(bounds or {})}.items():
        params[name].set(min=lo, max=hi)

    result = lm.fit(y, params, k=k, kB=kB, kJ=kJ, method=method, nan_policy='omit')
    return SpectrumFit(ck, cb, k, y, result.best_fit,
                       {p: (result.params[p].value, result.params[p].stderr) for p in model.params},
                       relative_error(y, result.best_fit))


''' Final template '''
@dataclass
class Template:
    model_name: str
    parametrization: str
    coefficients: dict        # param -> {'m': (value, error), 'y0': (value, error)}
    background: dict          # Background fields of the spectra used in the fit

    @property
    def model(self):
        return MODELS[self.model_name]

    @property
    def fit_background(self):
        return Background(**self.background)

    def params(self, cb):
        """Template parameters (a, b, s) at cb: p = m cb + y0."""
        return {p: c['m'][0] * cb + c['y0'][0] for p, c in self.coefficients.items()}

    def __call__(self, k, ck, cb, background=None):
        """Predicted P_MG/P_GR - 1 at k [h/Mpc]. `background` defaults to the one used in the fit."""
        model = self.model
        params = self.params(cb)
        hd = Horndeski(ck, cb, background or self.fit_background, self.parametrization)
        kB, kJ = hd.kB(), hd.kJ()

        for p in model.positive:
            if params[p] <= 0:
                raise ValueError(f"{p} must be positive, but {p}={params[p]:.4g} for ck={ck}, cb={cb}")
        if model.needs_kJ and kB <= kJ:
            raise ValueError(f"kB must be greater than kJ, but kB={kB:.4g}, kJ={kJ:.4g} for ck={ck}, cb={cb}")

        return model(np.asarray(k), kB, kJ, **params)

    def errors(self, spectra):
        """{(ck, cb): relative error} of the template against a SpectraSet (full k range)."""
        out = {}
        for ck, cb in spectra.pk_mg:
            try:
                out[(ck, cb)] = relative_error(spectra.deviation(ck, cb), self(spectra.k, ck, cb, spectra.background))
            except ValueError:
                out[(ck, cb)] = np.nan
        return out

    def save(self, name):
        (DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / 'templates' / f'{name}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, name):
        with open(DATA_DIR / 'templates' / f'{name}.pkl', 'rb') as f:
            return cls(**pickle.load(f))


''' Full pipeline '''
@dataclass
class TemplateFit:
    model: TemplateModel
    spectra: object
    spectrum_fits: dict = field(default_factory=dict)      # (ck, cb) -> SpectrumFit
    failed: dict = field(default_factory=dict)             # (ck, cb) -> message
    cb_fits: dict = field(default_factory=dict)            # ck -> {param: LinearFit in cb}
    ck_fits: dict = field(default_factory=dict)            # param -> {'m': MeanFit, 'y0': MeanFit}

    @property
    def cK(self):
        return np.array(sorted(self.cb_fits))

    @property
    def template(self):
        return Template(self.model.name, self.spectra.parametrization,
                        {p: {q: (f.value, f.error) for q, f in fits.items()} for p, fits in self.ck_fits.items()},
                        asdict(self.spectra.background))

    @property
    def mean_error(self):
        """Mean relative error of the individual spectrum fits."""
        return np.nanmean([f.err for f in self.spectrum_fits.values()])

    def mean_chi2_red(self, param):
        """Mean chi2_red over ck of the linear cB fit of `param`."""
        return np.nanmean([fits[param].chi2_red for fits in self.cb_fits.values() if param in fits])


def fit_template(spectra, model='tanh_kBkJ', k_min=None, k_max=None, method='leastsq',
                 guess=None, bounds=None, err_floor=0.0, verbose=True):
    """Run the three fitting steps on a SpectraSet.

    err_floor: minimum relative stderr of each spectrum parameter (e.g. 0.005 in
               the old DE-models notebook) to avoid over-weighting single points.
    """
    model = get_model(model)
    tf = TemplateFit(model, spectra)

    def stderr(value, err):
        if err is None or not np.isfinite(err) or err == 0:
            return np.inf
        return max(err, 1e-6 if err_floor else 0, abs(value) * err_floor)

    # 1. Each spectrum
    for ck, cb in spectra.pk_mg:
        try:
            tf.spectrum_fits[(ck, cb)] = fit_spectrum(spectra, ck, cb, model, k_min, k_max, method, guess, bounds)
        except Exception as e:
            tf.failed[(ck, cb)] = str(e)
            if verbose:
                print(f"Spectrum fit failed for ck={ck:.3g}, cb={cb:.3g}: {e}")

    # 2. Linear fit in cB for each ck
    for ck in spectra.cK:
        fits = [tf.spectrum_fits[(ck, cb)] for cb in spectra.cB_for(ck) if (ck, cb) in tf.spectrum_fits]
        cbs = [f.cb for f in fits]
        tf.cb_fits[ck] = {}
        for p in model.params:
            values = [f.params[p][0] for f in fits]
            errors = [stderr(*f.params[p]) for f in fits]
            try:
                tf.cb_fits[ck][p] = linear_fit(cbs, values, errors)
            except ValueError as e:
                if verbose:
                    print(f"cB fit failed for ck={ck:.3g}, param={p}: {e}")
    tf.cb_fits = {ck: fits for ck, fits in tf.cb_fits.items() if fits}

    # 3. Mean of m and y0 over cK
    for p in model.params:
        cks = [ck for ck in tf.cb_fits if p in tf.cb_fits[ck]]
        lines = [tf.cb_fits[ck][p] for ck in cks]
        tf.ck_fits[p] = {'m': mean_fit(cks, [l.m for l in lines], [l.m_err for l in lines]),
                         'y0': mean_fit(cks, [l.y0 for l in lines], [l.y0_err for l in lines])}

    return tf


def fit_backgrounds(sets, verbose=False, **fit_kwargs):
    """fit_template on every background of {(w0, wa): SpectraSet}. Failures are reported and skipped."""
    fits = {}
    for (w0, wa), spectra in sets.items():
        try:
            fits[(w0, wa)] = fit_template(spectra, verbose=verbose, **fit_kwargs)
        except Exception as e:
            print(f"[w0={w0:.2f}, wa={wa:.2f}] fit failed: {type(e).__name__}: {e}")
    return fits
