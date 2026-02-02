# Acoustic Wave Propagation in Sandwich Structures

Study notes + code for modeling and analyzing acoustic/elastic wave propagation in sandwich structures (facesheets + core).

## Structure
- `docs/`    – writeups (theory, derivations, reports)
- `notes/`   – scratch notes / meeting notes
- `src/`     – code (simulation, post-processing)
- `data/`    – input data
- `results/` – generated outputs (keep large files out of git)
- `figs/`    – figures for docs
- `refs/`    – papers / bibtex

## Python wavenumber demo (ported from MATLAB)

Install deps:

```bash
python3 -m pip install -r requirements.txt
```

Run demo:

```bash
python3 src/demo_sandwich_wavenumber.py
```

Main implementation:
- `src/sandwich_wavenumber.py`

## Getting started
Add your method choice (FEM/COMSOL, transfer matrix, spectral element, etc.) in `docs/`.
