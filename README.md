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

## Python wavenumber (ported from MATLAB)

Install deps:

```bash
python3 -m pip install -r requirements.txt
```

### Single-branch demo (tracking one mode)

```bash
python3 src/demo_sandwich_wavenumber.py
```

Main implementation:
- `src/sandwich_wavenumber.py`

### Multi-mode extraction (multiple roots per frequency + clustering + tracking)

Config-driven run (writes CSV + plots):

```bash
python3 src/run_dispersion.py --config examples/config.example.yml
```

Outputs (default):
- `results/dispersion.csv` (freq, mode, Re{k}, Im{k})
- `figs/re_k.png`, `figs/im_k.png`

Multi-mode solver:
- `src/multimode.py`
- `src/config_io.py`

## Getting started
Add your method choice (FEM/COMSOL, transfer matrix, spectral element, etc.) in `docs/`.
