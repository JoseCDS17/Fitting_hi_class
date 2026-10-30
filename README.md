# Fitting_hi_class

Analytic templates for the linear matter power-spectrum deviation $P_{MG}(k)/P_{GR}(k) - 1$ of Horndeski
models. The goal is to replace a full [hi_class](https://github.com/miguelzuma/hi_class_public) run with a
formula that has as few coefficients as possible, each with a physical reason to be there.

The repository holds the `hiclass_tools` Python package, the script that computes the spectra, and the
notebooks with the analysis and figures. Work in progress: the numbers below are the current state of the fit.

Units everywhere: `k`, `kB`, `kJ` in **h/Mpc**, `P(k)` in **(Mpc/h)³**.

## Current template

The deviation at $z=0$ is a step in $\log_{10} k$ located at the braiding scale (model `tanh_kB_L`):

$$\frac{P_{MG}}{P_{GR}} - 1 = L + A\left(\tanh\left[2\log_{10}\frac{s \, k}{k_B}\right] + 1\right)$$
 
- $L$ is the deviation at large scales ($k \ll k_B$). It depends only on the background and vanishes in ΛCDM.
- $2A$ is the height of the step and $k_B/s$ its position. Both depend on $c_B$.
- $c_K$ enters only through $k_B$.

Six coefficients cover every background:

$$A = m_A \, c_B + y_{0,A}, \qquad s = m_s \, c_B + y_{0,s}, \qquad L = \ell_1 (1 + w_0) + \ell_2 \, w_a$$

| Coefficient | $m_A$ | $y_{0,A}$ | $m_s$ | $y_{0,s}$ | $\ell_1$ | $\ell_2$ |
|---|---|---|---|---|---|---|
| Value | 0.02526 | 0.00020 | −0.42524 | 1.39114 | 0.04142 | 0.02154 |
| Error | 0.00009 | 0.00009 | 0.00364 | 0.00453 | 0.00216 | 0.00065 |

### Accuracy

Measured on the `fit_DE_grid` data used for the fit (100 backgrounds, 9930 spectra), for $k \geq 3\times10^{-3}$ h/Mpc.
The relative error is $\langle|y - y_{\rm fit}|\rangle / \langle|y|\rangle$ for each spectrum, so it is a fraction
of the deviation, not of $P(k)$.

| Mean relative error per background | Median | Maximum |
|---|---|---|
| One free fit per spectrum (3 parameters each) | 3.92 % | 4.64 % |
| Global template (6 coefficients) | 12.31 % | 33.79 % |

- 83 of the 100 backgrounds have a template error below 20 %.
- In units of $P_{GR}$, the $|{\rm template} - {\rm hi\_class}|$ of a spectrum has a median of 0.25 % averaged
  over $k$ and 0.56 % at its worst $k$. The largest value over all spectra is 3.0 %. For scale, the median peak
  deviation is 4.4 %.
- $L(w_0, w_a)$ is the weakest ingredient: its fit has $R^2 = 0.899$.

### Range of validity

The template has only been tested inside the fitted ranges:

| Quantity | Range |
|---|---|
| $w_0$ | $[-1, -0.7]$ |
| $w_a$ | $[-1, 0]$ |
| $c_K$ | $[10^3, 10^6]$ |
| $c_B$ | $[0.5, 1.5]$ |
| $k$ | $[3\times10^{-3}, 10]$ h/Mpc |
| Redshift | $z = 0$ only |
| Spectrum | linear $P(k)$ |
| Parametrization | `propto_omega` |

70 of the 10000 requested models were rejected by hi_class for a gradient instability. All of them have
$c_B = 0.5$ and belong to 7 backgrounds near the corner $w_0 + w_a \lesssim -1.86$.

## Physical setup

Every hi_class run has $\alpha_M = \alpha_T = 0$ and $M_*^2 = 1$, leaving two free functions with the same time
dependence:

| `parametrization` | $\alpha_K$ | $\alpha_B$ |
|---|---|---|
| `propto_omega` | $c_K \, \Omega_{DE}(a)$ | $c_B \, \Omega_{DE}(a)$ |
| `dust` | $c_K \, a^3$ | $c_B \, a^3$ |

The background is $w_0w_a$. $P_{GR}$ comes from a GR run with a $w_0w_a$ fluid of the same $(w_0, w_a)$, so the
deviation isolates the modified perturbations. The base cosmology is in `hiclass_tools/config.py`
($100 \, \theta_s = 1.04109$, $\omega_b = 0.02237$, $\omega_{cdm} = 0.1200$, $A_s = 2.1\times10^{-9}$, $n_s = 0.9649$).
With $\theta_s$ fixed, $\Omega_{m,0}$ changes with the background and is derived by CLASS.

The characteristic scales are computed analytically in `hiclass_tools/horndeski.py`, with
$D = \alpha_K + \tfrac{3}{2}\alpha_B^2$:

$$k_B = \sqrt{\frac{D}{\alpha_B^2} \, \Omega_{DE} \, (1 - w_{DE}) + \frac{9}{2} \, \Omega_m}\;\frac{aH}{c},
\qquad k_J = \frac{aH}{c \, c_s}$$

For $c_K \gg 1$ both scales grow as $\sqrt{c_K}$, so $k_B/\sqrt{c_K}$ and $k_J/\sqrt{c_K}$ depend only on $c_B$
and on the background (see `notebooks/range_application/scales_w0wa.ipynb`).

## Installation

Requires Python ≥ 3.10, a C compiler, and LaTeX for the figures (`use_style()` enables `text.usetex`; on
Debian/Ubuntu, `sudo apt install texlive-latex-extra cm-super dvipng`).

```bash
git clone --recurse-submodules https://github.com/JoseCDS17/Fitting_hi_class.git
cd Fitting_hi_class

python3 -m venv hiclass_env
source hiclass_env/bin/activate
pip install "numpy<2" scipy matplotlib lmfit "cython<3" setuptools jupyter

# Compile hi_class and install its Python wrapper (classy) in the environment
cd hi_class_public
make clean && make -j
cd ..

# Install this package (editable)
pip install -e .
```

The working environment uses Python 3.11, NumPy 1.26.4, Cython 0.29.36 and lmfit 1.3.4. See the hi_class/CLASS
README for compilation issues.

## Usage

### Evaluate the template

`data/` is not versioned, so a fresh clone builds the template from the coefficients above. No hi_class run is
needed if $\Omega_{m,0}$ is known:

```python
import numpy as np
from hiclass_tools import Background, GlobalTemplate

template = GlobalTemplate(
    A={'m': (0.02526, 0.00009), 'y0': (0.00020, 0.00009)},
    s={'m': (-0.42524, 0.00364), 'y0': (1.39114, 0.00453)},
    L={'l1': (0.04142, 0.00216), 'l2': (0.02154, 0.00065)},
    parametrization='propto_omega',
)

background = Background(w0=-0.85, wa=-0.56, Omega_m0=0.3124)
k = np.logspace(-3, 1, 200)                                    # h/Mpc
deviation = template(k, ck=1e4, cb=1.0, background=background)  # P_MG/P_GR - 1
```

`Background.from_class(w0, wa)` gets $\Omega_{m,0}$ from CLASS with $\theta_s$ fixed. After running
`fit_global.ipynb`, `GlobalTemplate.load('fit_DE_grid_global')` loads the template with full precision.

### Compute spectra (slow, once)

Edit the configuration block of `scripts/compute_spectra.py` (name, backgrounds, cK, cB, k) and run it from the
project root:

```bash
source hiclass_env/bin/activate
python scripts/compute_spectra.py
```

The result goes to `data/spectra/<NAME>.pkl`, or to one file per background in `data/spectra/<NAME>/` when several
backgrounds are given. Backgrounds already saved are skipped, so an interrupted grid resumes by relaunching the
script. Each file stores every parameter used.

The notebooks expect these datasets. The script is currently configured for `fit_DE_grid`.

| `NAME` | Backgrounds $(w_0, w_a)$ | $c_K$ | $c_B$ | $k$ [h/Mpc] |
|---|---|---|---|---|
| `fit_LCDM` | $(-1, 0)$ | 10 in $[10^3, 10^6]$ | 10 in $[0.5, 1.5]$ | 100 in $[10^{-3}, 10]$ |
| `fit_w0wa` | $(-0.85, -0.56)$ | 10 in $[10^3, 10^6]$ | 10 in $[0.5, 1.5]$ | 100 in $[10^{-3}, 10]$ |
| `fit_DE_grid` | 10×10 grid, $w_0 \in [-1, -0.7]$, $w_a \in [-1, 0]$ | 10 in $[10^3, 10^6]$ | 10 in $[0.5, 1.5]$ | 50 in $[10^{-3}, 10]$ |
| `range_DE_grid` | 5×5 grid, $w_0 \in [-1, -0.8]$, $w_a \in [-1, 0]$ | 10 in $[10^3, 10^6]$ | 10 in $[0.5, 1.5]$ | 50 in $[10^{-3}, 10]$ |
| `range_LCDM_ckhigh`, `range_w0wa_ckhigh` | $(-1, 0)$, $(-0.85, -0.56)$ | 6 in $[10^2, 10^7]$ | 5 in $[0.5, 1.5]$ | 100 in $[3\times10^{-4}, 10]$ |
| `range_LCDM_cklow`, `range_w0wa_cklow` | $(-1, 0)$, $(-0.85, -0.56)$ | 6 in $[0.1, 20]$ | 5 in $[0.5, 1.5]$ | 100 in $[3\times10^{-5}, 10^{-2}]$ |

### Analyse and fit (fast, as often as needed)

```python
from hiclass_tools import load_spectra, load_spectra_group, fit_template, fit_global_template

# One background
spectra = load_spectra('fit_LCDM')
deviation = spectra.deviation(ck=1e3, cb=0.5)       # P_MG/P_GR - 1
kB, kJ = spectra.scales(ck=1e3, cb=0.5)
tfit = fit_template(spectra, 'tanh_kB')

# Grid of backgrounds: {(w0, wa): SpectraSet}
sets = load_spectra_group('fit_DE_grid')
gfit = fit_global_template(sets, k_min=3e-3)
gfit.template.save('fit_DE_grid_global')            # data/templates/fit_DE_grid_global.pkl
```

## Fitting pipelines

**Per background**, `fit_template`, for one `SpectraSet`:

1. Fit each $(c_K, c_B)$ spectrum with a template model, giving its three parameters.
2. For each $c_K$, fit each parameter linearly in $c_B$: $p = m(c_K) \, c_B + y_0(c_K)$.
3. Average $m$ and $y_0$ over $c_K$ (simple mean, error = scatter$/\sqrt{N}$).

**Global**, `fit_global_template`, for many backgrounds (model `tanh_kB_L` only):

1. Fit every spectrum of every background.
2. Fit $A$ and $s$ linearly in $c_B$ with all the spectra at once, weighted by their errors.
3. Take the weighted mean of $L$ for each background, then fit $L = \ell_1(1 + w_0) + \ell_2 w_a$ by least squares.

| Model | Form | Status |
|---|---|---|
| `tanh_kBkJ` | $a \tanh\left[s \, \frac{\log_{10}(k/\sqrt{k_B k_J})}{\log_{10}(k_B/k_J)}\right] + b$ | First version, needs $k_B > k_J$ |
| `tanh_kB` | $a \tanh\left[2\log_{10}(s \, k/k_B)\right] + b$ | Per-background fits |
| `tanh_kB_L` | $L + A\left(\tanh\left[2\log_{10}(s \, k/k_B)\right] + 1\right)$ | Current. Same shape as `tanh_kB`, with $a = A$ and $b = L + A$ |

To add a model, write its function and add a `TemplateModel` entry to `MODELS` in `hiclass_tools/fitting.py`.

## Notebooks

| Notebook | Content |
|---|---|
| `fitting/fit_LCDM`, `fitting/fit_w0wa` | The three fitting steps on a single background |
| `fitting/fit_DE_grid` | Per-background fits as maps over $(w_0, w_a)$, dependence on $w_0 + w_a$, reference template |
| `fitting/fit_global` | The 6-coefficient global template, its error maps and the region where it works |
| `range_application/range_{LCDM,w0wa}_{ckhigh,cklow}` | Deviations with $k_B$ and $k_J$ for large and small $c_K$ |
| `range_application/range_DE_grid` | Deviations across a coarse grid of backgrounds |
| `range_application/scales_w0wa` | Maps of $k_B$, $k_J$ and $c_s^2$ over $(w_0, w_a)$, and their scaling with $c_K$ |

Figures are written out in the notebooks. `hiclass_tools/plotting.py` only provides the style and colour scales.

## Structure

```
hiclass_tools/      Python package
  config.py         data path, constants, base cosmology for every CLASS run
  background.py     Background(w0, wa, Omega_m0): H, Omega_m, Omega_DE, w_eff...
  horndeski.py      Horndeski(ck, cb, background, parametrization): alphas, cs2, kJ, kB
  spectra.py        SpectraSet: runs hi_class (GR + MG grid in parallel), save/load
  fitting.py        template models, fitting pipelines, Template and GlobalTemplate
  plotting.py       matplotlib style and colour scales
scripts/
  compute_spectra.py
notebooks/
  fitting/             fit_LCDM, fit_w0wa, fit_DE_grid, fit_global
  range_application/   range_{LCDM,w0wa}_{ckhigh,cklow}, range_DE_grid, scales_w0wa
hi_class_public/       hi_class (git submodule)
data/                  generated locally, not versioned
  spectra/   computed spectra
  templates/ saved templates
```

## hi_class

All the power spectra are computed with **hi_class**, the Horndeski extension of the Boltzmann code CLASS, taken
from [miguelzuma/hi_class_public](https://github.com/miguelzuma/hi_class_public) (branch `hi_class`). It is included
as a git submodule in `hi_class_public/`, pinned to the exact commit used (tag `v3.0`), and is not modified. Please
cite the hi_class and CLASS papers listed in its README when using this code.
