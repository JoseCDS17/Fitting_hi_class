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

from horndeski_template import Background, Horndeski
from horndeski_template import GlobalTemplate as AnalyticGlobalTemplate
from horndeski_template.template import tanh_kB_L

from .config import DATA_DIR


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
    TemplateModel('tanh_kB_L', tanh_kB_L, ('A', 'L', 's'),
                  lambda y, cb: dict(A=(y[-1] - y[0]) / 2, L=y[0], s=1 / cb),
                  r'L + A \left(\tanh\left[2\log_{10}(s\,k/k_B)\right] + 1\right)',
                  positive=('s',), needs_kJ=False, bounds={'s': (1e-10, np.inf)}),
]}


def get_model(model):
    return MODELS[model] if isinstance(model, str) else model


def r_squared(y, y_fit):
    """Fraction of the variance of y explained by y_fit."""
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y - y_fit)**2) / ss_tot if ss_tot > 0 else np.nan


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
    """Weighted linear fit p = m x + y0 (parameter errors as np.polyfit, scaled by the residuals)."""
    m: float
    m_err: float
    y0: float
    y0_err: float
    x: np.ndarray             # data used in the fit
    values: np.ndarray
    sigma: np.ndarray
    r2: float                 # fraction of the variance of `values` explained by the line

    def __call__(self, x):
        return self.m * x + self.y0


def linear_fit(x, values, sigma):
    x, values, sigma = _clean(x, values, sigma)
    if len(x) < 2:
        raise ValueError(f"Need at least 2 valid points for a linear fit, got {len(x)}")

    (m, y0), cov = np.polyfit(x, values, 1, w=1 / sigma, cov=True if len(x) > 3 else 'unscaled')
    return LinearFit(m, np.sqrt(abs(cov[0, 0])), y0, np.sqrt(abs(cov[1, 1])), x, values, sigma,
                     r_squared(values, m * x + y0))


@dataclass
class MeanFit:
    """Simple mean, with the standard error of the mean (scatter / sqrt(N)) as error."""
    value: float
    error: float
    x: np.ndarray             # data used in the mean
    values: np.ndarray
    sigma: np.ndarray         # error bars of each point (only for plotting)
    rel_scatter: float        # std / |mean| of the points: how far they are from a constant


def mean_fit(x, values, sigma):
    x, values, sigma = _clean(x, values, sigma)
    if len(x) == 0:
        raise ValueError("No valid points for the mean")

    mean = np.mean(values)
    dof = len(x) - 1
    std = np.std(values, ddof=1) if dof > 0 else np.nan
    return MeanFit(mean, std / np.sqrt(len(x)), x, values, sigma, std / abs(mean))


def weighted_mean_fit(x, values, sigma):
    """Inverse-variance weighted mean: points with huge errors (degenerate fits) barely weigh."""
    x, values, sigma = _clean(x, values, sigma)
    if len(x) == 0:
        raise ValueError("No valid points for the mean")

    w = 1 / sigma**2
    mean = np.average(values, weights=w)
    std = np.sqrt(np.average((values - mean)**2, weights=w))
    return MeanFit(mean, np.sqrt(1 / np.sum(w)), x, values, sigma, std / abs(mean))


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
def template_errors(template, spectra, k_min=None, k_max=None):
    """{(ck, cb): relative error} of template(k, ck, cb, background) against a SpectraSet in [k_min, k_max]."""
    k = spectra.k
    mask = (k >= (k_min if k_min is not None else k[0])) & (k <= (k_max if k_max is not None else k[-1]))
    out = {}
    for ck, cb in spectra.pk_mg:
        try:
            y = spectra.deviation(ck, cb)[mask]
            out[(ck, cb)] = relative_error(y, template(k[mask], ck, cb, spectra.background))
        except ValueError:
            out[(ck, cb)] = np.nan
    return out


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

    def errors(self, spectra, k_min=None, k_max=None):
        """{(ck, cb): relative error} of the template against a SpectraSet in [k_min, k_max] (default: full k range)."""
        return template_errors(self, spectra, k_min, k_max)

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

    def mean_r2(self, param):
        """Mean R^2 over ck of the linear cB fit of `param`."""
        return np.nanmean([fits[param].r2 for fits in self.cb_fits.values() if param in fits])


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


''' Global template over backgrounds (model tanh_kB_L) '''
class GlobalTemplate(AnalyticGlobalTemplate):
    """horndeski_template.GlobalTemplate plus the comparison against spectra and save/load."""

    @property
    def model(self):
        return MODELS[self.model_name]

    def errors(self, spectra, k_min=None, k_max=None):
        """{(ck, cb): relative error} against a SpectraSet in [k_min, k_max] (default: full k range)."""
        return template_errors(self.evaluate, spectra, k_min, k_max)

    def save(self, name):
        (DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / 'templates' / f'{name}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, name):
        with open(DATA_DIR / 'templates' / f'{name}.pkl', 'rb') as f:
            return cls(**pickle.load(f))


@dataclass
class GlobalFit:
    sets: dict                                             # {(w0, wa): SpectraSet}
    spectrum_fits: dict = field(default_factory=dict)      # (w0, wa) -> {(ck, cb): SpectrumFit}
    failed: dict = field(default_factory=dict)             # (w0, wa, ck, cb) -> message
    A_fit: LinearFit = None                                # A vs cB, all spectra of all backgrounds
    s_fit: LinearFit = None                                # s vs cB, all spectra of all backgrounds
    L_means: dict = field(default_factory=dict)            # (w0, wa) -> MeanFit of L over its spectra
    L_coeffs: dict = field(default_factory=dict)           # {'l1': (value, error), 'l2': (value, error)}
    L_r2: float = np.nan                                   # R^2 of the L(w0, wa) fit

    @property
    def template(self):
        return GlobalTemplate({'m': (self.A_fit.m, self.A_fit.m_err), 'y0': (self.A_fit.y0, self.A_fit.y0_err)},
                              {'m': (self.s_fit.m, self.s_fit.m_err),
                               'y0': (self.s_fit.y0, self.s_fit.y0_err)},
                              self.L_coeffs, next(iter(self.sets.values())).parametrization)

    def L_model(self, w0, wa):
        return self.L_coeffs['l1'][0] * (1 + w0) + self.L_coeffs['l2'][0] * wa

    def spectrum_errors(self):
        """{(w0, wa): mean relative error of the individual spectrum fits}."""
        return {key: np.nanmean([f.err for f in fits.values()]) for key, fits in self.spectrum_fits.items()}

    def template_errors(self, k_min=None, k_max=None):
        """{(w0, wa): mean relative error of the global template}."""
        template = self.template
        return {key: np.nanmean(list(template.errors(spectra, k_min, k_max).values()))
                for key, spectra in self.sets.items()}


def fit_global_template(sets, k_min=None, k_max=None, method='leastsq', verbose=True):
    """Fit the tanh_kB_L model on every spectrum of every background and build a GlobalTemplate.

    A and s: weighted linear fits in cB using the spectra of all backgrounds at once (degenerate
    spectrum fits have huge errors, so they barely weigh). L: weighted mean over the spectra of each
    background, then L = l1 (1 + w0) + l2 wa by least squares.
    """
    gf = GlobalFit(sets)
    for key, spectra in sets.items():
        gf.spectrum_fits[key] = {}
        for ck, cb in spectra.pk_mg:
            try:
                gf.spectrum_fits[key][(ck, cb)] = fit_spectrum(spectra, ck, cb, 'tanh_kB_L', k_min, k_max, method)
            except Exception as e:
                gf.failed[(*key, ck, cb)] = str(e)
                if verbose:
                    print(f"Spectrum fit failed for w0={key[0]:.3g}, wa={key[1]:.3g}, ck={ck:.3g}, cb={cb:.3g}: {e}")

    fits = [f for bg_fits in gf.spectrum_fits.values() for f in bg_fits.values()]
    cbs = [f.cb for f in fits]
    gf.A_fit = linear_fit(cbs, [f.params['A'][0] for f in fits], [f.params['A'][1] for f in fits])
    gf.s_fit = linear_fit(cbs, [f.params['s'][0] for f in fits], [f.params['s'][1] for f in fits])

    for key, bg_fits in gf.spectrum_fits.items():
        Ls = [f.params['L'] for f in bg_fits.values()]
        gf.L_means[key] = weighted_mean_fit([f.cb for f in bg_fits.values()], [v for v, _ in Ls], [e for _, e in Ls])

    keys = list(gf.L_means)
    X = np.array([[1 + w0, wa] for w0, wa in keys])
    y = np.array([gf.L_means[key].value for key in keys])
    lam, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ lam
    dof = len(y) - 2
    cov = np.sum(res**2) / dof * np.linalg.inv(X.T @ X) if dof > 0 else np.full((2, 2), np.nan)
    gf.L_coeffs = {'l1': (lam[0], np.sqrt(cov[0, 0])), 'l2': (lam[1], np.sqrt(cov[1, 1]))}
    gf.L_r2 = r_squared(y, X @ lam)
    return gf
