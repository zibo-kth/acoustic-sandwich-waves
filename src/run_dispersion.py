"""run_dispersion.py

CLI-ish script to compute dispersion (Re{k}) + attenuation (Im{k}) and save CSV.

Usage:
  python3 -m pip install -r requirements.txt
  python3 src/run_dispersion.py --config examples/config.example.yml

Outputs:
- CSV: one row per (frequency, mode)
- Plots: Re{k}(f), Im{k}(f) per mode (optional if matplotlib installed)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config_io import load_config
from multimode import MultiModeOpts, multimode_wavenumber
from sandwich_wavenumber import Core, Params, Skin, SolverOpts


def _build_params(cfg: Dict[str, Any]) -> Params:
    panel = cfg.get("panel", {})

    s1 = panel.get("skin1", {})
    skin1 = Skin(
        E=float(s1.get("E", 70e9)),
        nu=float(s1.get("nu", 0.33)),
        rho=float(s1.get("rho", 2700.0)),
        t=float(s1.get("t", 1e-3)),
        eta=float(s1.get("eta", 0.0)),
    )

    s3 = panel.get("skin3", "same_as_skin1")
    if isinstance(s3, str) and s3 == "same_as_skin1":
        skin3 = Skin(**skin1.__dict__)
    else:
        skin3 = Skin(
            E=float(s3.get("E", skin1.E)),
            nu=float(s3.get("nu", skin1.nu)),
            rho=float(s3.get("rho", skin1.rho)),
            t=float(s3.get("t", skin1.t)),
            eta=float(s3.get("eta", skin1.eta)),
        )

    c = panel.get("core", {})
    core = Core(
        E=float(c.get("E", 200e6)),
        nu=float(c.get("nu", 0.25)),
        rho=float(c.get("rho", 120.0)),
        h=float(c.get("h", 20e-3)),
        eta=float(c.get("eta", 0.0)),
    )

    return Params(skin1=skin1, skin3=skin3, core=core)


def _build_freq(cfg: Dict[str, Any]) -> np.ndarray:
    fcfg = cfg.get("freq", {})
    start = float(fcfg.get("start_hz", 20.0))
    stop = float(fcfg.get("stop_hz", 2000.0))
    n = int(fcfg.get("n", 200))
    return np.linspace(start, stop, n)


def _build_solver_opts(cfg: Dict[str, Any]) -> tuple[SolverOpts, MultiModeOpts]:
    scfg = cfg.get("solver", {})

    solver = SolverOpts(
        use_root=bool(scfg.get("use_root", True)),
        root_maxiter=int(scfg.get("root_maxiter", 200)),
        nm_maxiter=int(scfg.get("nm_maxiter", 2000)),
        nm_xatol=float(scfg.get("nm_xatol", 1e-10)),
        nm_fatol=float(scfg.get("nm_fatol", 1e-14)),
        do_newton_polish=bool(scfg.get("do_newton_polish", True)),
        newton_iters=int(scfg.get("newton_iters", 6)),
        dk_fd=float(scfg.get("dk_fd", 1e-6)),
        k0_imag=float(scfg.get("k0_imag", 0.0)),
    )

    # seeds sub-blocks
    re_cfg = scfg.get("k_seed_re", {})
    im_cfg = scfg.get("k_seed_im", {})

    mm = MultiModeOpts(
        max_modes=int(scfg.get("max_modes", 6)),
        n_seeds=int(scfg.get("n_seeds", 18)),
        span_factor=float(re_cfg.get("span_factor", 8.0)),
        min_factor=float(re_cfg.get("min_factor", 0.25)),
        im_values=tuple(float(x) for x in im_cfg.get("values", [0.0, 1.0, 5.0])),
        cluster_tol_rel=float(scfg.get("cluster_tol_rel", 1e-3)),
        match_tol_rel=float(scfg.get("match_tol_rel", 5e-2)),
    )

    return solver, mm


def save_csv(path: str | Path, freq: np.ndarray, k_modes: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, f in enumerate(freq):
        for m in range(k_modes.shape[1]):
            k = k_modes[i, m]
            if np.isfinite(k.real) and np.isfinite(k.imag):
                rows.append((float(f), int(m), float(k.real), float(k.imag)))

    header = "freq_hz,mode,k_real_rad_per_m,k_imag_per_m"
    data = np.asarray(rows, dtype=float)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_plots(paths: Dict[str, Any], freq: np.ndarray, k_modes: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    # Re{k}
    re_path = paths.get("re_k")
    if re_path:
        plt.figure()
        for m in range(k_modes.shape[1]):
            plt.plot(freq, np.real(k_modes[:, m]), linewidth=1.2, label=f"mode {m}")
        plt.grid(True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Re{k} (rad/m)")
        plt.title("Dispersion (real part)")
        plt.legend()
        Path(re_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(re_path, dpi=200)

    # Im{k}
    im_path = paths.get("im_k")
    if im_path:
        plt.figure()
        for m in range(k_modes.shape[1]):
            plt.plot(freq, np.imag(k_modes[:, m]), linewidth=1.2, label=f"mode {m}")
        plt.grid(True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Im{k} (1/m)")
        plt.title("Attenuation (imag part)")
        plt.legend()
        Path(im_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(im_path, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to .yml/.yaml/.json config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = _build_params(cfg)
    freq = _build_freq(cfg)
    solver, mm = _build_solver_opts(cfg)

    k_modes, info = multimode_wavenumber(freq, p, solver_opts=solver, mm=mm)
    print(info)

    out = cfg.get("output", {})
    csv_path = out.get("csv", "results/dispersion.csv")
    save_csv(csv_path, freq, k_modes)

    plots = out.get("plots", {})
    save_plots(plots, freq, k_modes)


if __name__ == "__main__":
    main()
