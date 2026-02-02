"""sandwich_wavenumber.py

Compute complex in-plane wavenumber k(ω) for a sandwich panel.

Ported from the provided MATLAB demo ("global matrix + interface impedances").
The solver tracks a single dispersion branch by using the previous frequency
solution as the initial guess.

Model (as in the MATLAB script):
- Core: 2D isotropic elasticity via longitudinal + transverse potentials.
- Skins: Kirchhoff thin plate bending + in-plane membrane, enforced via
         impedance-style traction/displacement coupling at y=0 and y=h.

Numerics:
- Primary: scipy.optimize.root on [Re(detM), Im(detM)] = 0 for variables
  [Re(k), Im(k)].
- Fallback: scipy.optimize.minimize (Nelder-Mead) on |detM|^2.
- Optional Newton polish using complex finite-difference derivative.

Notes:
- Requires numpy. Root solving requires scipy.
- The determinant equation has multiple branches; initial guesses determine
  which mode you follow. To find multiple modes, run multiple guesses per
  frequency and cluster distinct roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class Skin:
    E: float = 70e9
    nu: float = 0.33
    rho: float = 2700.0
    t: float = 1.0e-3
    eta: float = 0.0  # loss factor (E -> E*(1+i*eta))


@dataclass
class Core:
    E: float = 200e6
    nu: float = 0.25
    rho: float = 120.0
    h: float = 20e-3
    eta: float = 0.0


@dataclass
class Params:
    skin1: Skin
    skin3: Skin
    core: Core


@dataclass
class SolverOpts:
    # Root solver
    use_root: bool = True
    root_maxiter: int = 200
    # Minimizer fallback
    nm_maxiter: int = 2000
    nm_xatol: float = 1e-10
    nm_fatol: float = 1e-14
    # Newton polish
    do_newton_polish: bool = True
    newton_iters: int = 6
    dk_fd: float = 1e-6  # relative finite-diff step for df/dk
    # Tracking
    k0_imag: float = 0.0


def complex_modulus(E0: float, eta: float | None = 0.0) -> complex:
    """Loss-factor model: E = E0 * (1 + i*eta)."""
    eta = 0.0 if eta is None else float(eta)
    return complex(E0) * (1.0 + 1j * eta)


def approx_static_bending_stiffness(p: Params) -> complex:
    """Very rough bending stiffness for initial guess only."""
    E1 = complex_modulus(p.skin1.E, p.skin1.eta)
    E3 = complex_modulus(p.skin3.E, p.skin3.eta)

    nu1, nu3 = p.skin1.nu, p.skin3.nu
    t1, t3 = p.skin1.t, p.skin3.t
    h = p.core.h

    E1p = E1 / (1 - nu1**2)
    E3p = E3 / (1 - nu3**2)

    D1 = E1p * t1**3 / 12.0
    D3 = E3p * t3**3 / 12.0

    # Distance between skin mid-planes (approx)
    H = h + 0.5 * (t1 + t3)

    # Separation contribution (dominant for a sandwich)
    Dsep = E1p * t1 * (H / 2) ** 2 + E3p * t3 * (H / 2) ** 2

    return D1 + D3 + Dsep


def _safe_sqrt(z: complex) -> complex:
    """Principal square root, with numpy complex handling."""
    return np.sqrt(z + 0j)


def M_sandwich(k: complex, omega: float, p: Params) -> np.ndarray:
    """Build the 4x4 global matrix whose determinant is the dispersion equation."""
    # Complex moduli
    E1 = complex_modulus(p.skin1.E, p.skin1.eta)
    E3 = complex_modulus(p.skin3.E, p.skin3.eta)
    Ec = complex_modulus(p.core.E, p.core.eta)

    nu1, nu3, nuc = p.skin1.nu, p.skin3.nu, p.core.nu
    rho1, rho3, rhoc = p.skin1.rho, p.skin3.rho, p.core.rho
    t1, t3, h = p.skin1.t, p.skin3.t, p.core.h

    # Skin stiffness/impedances
    E1p = E1 / (1 - nu1**2)  # plane-stress effective modulus
    E3p = E3 / (1 - nu3**2)

    D1 = E1p * t1**3 / 12.0
    D3 = E3p * t3**3 / 12.0

    mu1 = rho1 * t1  # mass/area
    mu3 = rho3 * t3

    Zw1 = D1 * k**4 - mu1 * omega**2
    Zw3 = D3 * k**4 - mu3 * omega**2

    Zu1 = E1p * t1 * k**2 - mu1 * omega**2
    Zu3 = E3p * t3 * k**2 - mu3 * omega**2

    # Core wave numbers
    # 3D isotropic Lamé parameters (as in MATLAB script)
    mu = Ec / (2 * (1 + nuc))
    lam = Ec * nuc / ((1 + nuc) * (1 - 2 * nuc))

    cL = _safe_sqrt((lam + 2 * mu) / rhoc)
    cT = _safe_sqrt(mu / rhoc)

    kL = omega / cL
    kT = omega / cT

    qL = _safe_sqrt(kL**2 - k**2)
    qT = _safe_sqrt(kT**2 - k**2)

    ELp, ELm = np.exp(1j * qL * h), np.exp(-1j * qL * h)
    ETp, ETm = np.exp(1j * qT * h), np.exp(-1j * qT * h)

    # Shorthand from stress expression
    C = lam * k**2 + (lam + 2 * mu) * qL**2

    # Row 1: sigma(0) - Zw1*w(0) = 0
    r1A1 = -C - 1j * qL * Zw1
    r1A2 = -C + 1j * qL * Zw1
    r1B1 = -2 * mu * k * qT - 1j * k * Zw1
    r1B2 = 2 * mu * k * qT - 1j * k * Zw1

    # Row 2: tau(0) - Zu1*u(0) = 0
    r2A1 = 2 * mu * k * qL + 1j * k * Zu1
    r2A2 = -2 * mu * k * qL + 1j * k * Zu1
    r2B1 = mu * (k**2 - qT**2) - 1j * qT * Zu1
    r2B2 = mu * (k**2 - qT**2) + 1j * qT * Zu1

    # Row 3: sigma(h) + Zw3*w(h) = 0
    r3A1 = ELp * (-C + 1j * qL * Zw3)
    r3A2 = ELm * (-C - 1j * qL * Zw3)
    r3B1 = ETp * (-2 * mu * k * qT + 1j * k * Zw3)
    r3B2 = ETm * (2 * mu * k * qT + 1j * k * Zw3)

    # Row 4: tau(h) + Zu3*u(h) = 0
    r4A1 = ELp * (2 * mu * k * qL - 1j * k * Zu3)
    r4A2 = ELm * (-2 * mu * k * qL - 1j * k * Zu3)
    r4B1 = ETp * (mu * (k**2 - qT**2) + 1j * qT * Zu3)
    r4B2 = ETm * (mu * (k**2 - qT**2) - 1j * qT * Zu3)

    return np.array(
        [
            [r1A1, r1A2, r1B1, r1B2],
            [r2A1, r2A2, r2B1, r2B2],
            [r3A1, r3A2, r3B1, r3B2],
            [r4A1, r4A2, r4B1, r4B2],
        ],
        dtype=np.complex128,
    )


def detM_sandwich(k: complex, omega: float, p: Params) -> complex:
    return np.linalg.det(M_sandwich(k, omega, p))


def newton_polish_complex(k: complex, f, iters: int, dk_fd: float) -> complex:
    for _ in range(int(iters)):
        fk = f(k)
        if abs(fk) < 1e-14:
            break
        dk = dk_fd * max(1.0, abs(k))
        df = (f(k + dk) - f(k - dk)) / (2.0 * dk)
        if df == 0 or not np.isfinite(df):
            break
        k_new = k - fk / df
        if not (np.isfinite(k_new.real) and np.isfinite(k_new.imag)):
            break
        k = k_new
    return k


def solve_k_onefreq(omega: float, p: Params, k_guess: complex, opts: SolverOpts) -> complex:
    fdet = lambda kk: detM_sandwich(kk, omega, p)

    # SciPy-based solvers
    try:
        import scipy.optimize as opt  # type: ignore

        if opts.use_root:
            x0 = np.array([k_guess.real, k_guess.imag], dtype=float)

            def fun(x):
                val = fdet(x[0] + 1j * x[1])
                return np.array([val.real, val.imag], dtype=float)

            res = opt.root(fun, x0, method="hybr", options={"maxfev": int(opts.root_maxiter)})
            if res.success:
                k_sol = complex(res.x[0], res.x[1])
            else:
                # Fallback to Nelder-Mead minimize |det|^2
                def obj(x):
                    return abs(fdet(x[0] + 1j * x[1])) ** 2

                res2 = opt.minimize(
                    obj,
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": int(opts.nm_maxiter),
                        "xatol": float(opts.nm_xatol),
                        "fatol": float(opts.nm_fatol),
                        "disp": False,
                    },
                )
                k_sol = complex(res2.x[0], res2.x[1])
        else:
            x0 = np.array([k_guess.real, k_guess.imag], dtype=float)

            def obj(x):
                return abs(fdet(x[0] + 1j * x[1])) ** 2

            res = opt.minimize(
                obj,
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": int(opts.nm_maxiter),
                    "xatol": float(opts.nm_xatol),
                    "fatol": float(opts.nm_fatol),
                    "disp": False,
                },
            )
            k_sol = complex(res.x[0], res.x[1])

    except ModuleNotFoundError as e:
        raise RuntimeError(
            "SciPy is required for the root/minimization solve. "
            "Install with: python3 -m pip install scipy"
        ) from e

    if opts.do_newton_polish:
        k_sol = newton_polish_complex(k_sol, fdet, iters=opts.newton_iters, dk_fd=opts.dk_fd)

    return k_sol


def sandwich_wavenumber(freq_hz: Iterable[float], p: Params, opts: SolverOpts | None = None) -> Tuple[np.ndarray, Dict]:
    """Compute k(freq) tracking a single branch.

    Returns
    -------
    k : np.ndarray complex, shape (N,)
    info : dict
    """
    opts = SolverOpts() if opts is None else opts

    freq = np.asarray(list(freq_hz), dtype=float).reshape(-1)
    N = freq.size
    k_out = np.full(N, np.nan + 1j * np.nan, dtype=np.complex128)

    D0 = approx_static_bending_stiffness(p)
    mu_tot = (
        p.skin1.rho * p.skin1.t
        + p.skin3.rho * p.skin3.t
        + p.core.rho * p.core.h
    )

    k_prev = None

    for i, f in enumerate(freq):
        omega = 2.0 * np.pi * f

        if i == 0 or k_prev is None or not np.isfinite(k_prev.real):
            D0r = float(np.real(D0))
            if D0r <= 0:
                D0r = max(float(abs(D0)), 1e-9)
            k0 = (mu_tot * omega**2 / D0r) ** 0.25
            k_guess = complex(k0, opts.k0_imag)
        else:
            k_guess = k_prev

        k_sol = solve_k_onefreq(omega, p, k_guess, opts)
        k_out[i] = k_sol
        k_prev = k_sol

    info = {
        "model": "Core=2D isotropic elasticity (L+T), Skins=Kirchhoff bending + membrane impedance, det(M)=0",
        "tracking": True,
    }

    return k_out, info
