# Fitting_hi_class

Power-spectrum deviations $P_{MG}/P_{GR}-1$ of Horndeski models (hi_class) and their fitting templates.

Units everywhere: `k`, `kB`, `kJ` in **h/Mpc**, `P(k)` in **(Mpc/h)³**.

## hi_class

All the power spectra are computed with **hi_class**, the Horndeski extension of the
Boltzmann code CLASS, taken from [miguelzuma/hi_class_public](https://github.com/miguelzuma/hi_class_public)
(branch `hi_class`). It is included in this repository as a git submodule (`hi_class_public/`),
pinned to the exact commit used, and is not modified. Please cite the hi_class and CLASS
papers listed in its README when using this code.

## Installation

```bash
git clone --recurse-submodules https://github.com/JoseCDS17/Fitting_hi_class.git
cd Fitting_hi_class

python3 -m venv hiclass_env
source hiclass_env/bin/activate
pip install numpy scipy matplotlib lmfit cython jupyter

# Compile hi_class and install its Python wrapper (classy) in the environment
cd hi_class_public
make clean && make -j
cd python && pip install . && cd ../..

# Install this package (editable)
pip install -e .
```

See the hi_class/CLASS README for compilation issues.

## Workflow

1. **Compute** (slow, once): edit the configuration block of `scripts/compute_spectra.py`
   (name, backgrounds, cK, cB, k) and run it from the project root:
   ```bash
   source hiclass_env/bin/activate
   python scripts/compute_spectra.py
   ```
   The result goes to `data/spectra/<NAME>.pkl` (or one file per background in
   `data/spectra/<NAME>/` when several backgrounds are given). Each file stores every
   parameter used.
2. **Analyse** (fast, as often as needed): in a notebook, `load_spectra('<NAME>')`
   (or `load_spectra_group('<NAME>')` for several backgrounds), then plot and fit.

## Structure

```
hiclass_tools/      Python package (installed in hiclass_env with `pip install -e .`)
  config.py         data path, constants, base cosmology for every CLASS run
  background.py     Background(w0, wa, Omega_m0): H, Omega_m, Omega_DE, w_eff...
  horndeski.py      Horndeski(ck, cb, background, parametrization): alphas, cs2, kJ, kB
  spectra.py        SpectraSet: runs hi_class (GR + MG grid in parallel), save/load
  fitting.py        template models ('tanh_kBkJ', 'tanh_kB') and fitting pipeline -> Template
  plotting.py       matplotlib style and colour scales (figures live in the notebooks)
scripts/
  compute_spectra.py
notebooks/
  range_application/   range_{LCDM,w0wa}_{ckhigh,cklow}, range_DE_grid
  fitting/             fit_LCDM, fit_w0wa
hi_class_public/       hi_class (git submodule)
data/                  generated locally, not versioned
  spectra/   computed spectra
  templates/ saved templates
```

New template models: add a function and a `TemplateModel` entry to `MODELS` in `hiclass_tools/fitting.py`.
