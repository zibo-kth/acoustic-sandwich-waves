"""multimode.py

Multi-mode root finding for det(M(k, ω)) = 0.

We solve the complex scalar equation by converting it to a 2D real system:
  Re(det) = 0, Im(det) = 0, unknowns: Re(k), Im(k).

For each frequency, we:
1) Generate a set of complex initial guesses ("seeds")
2) Solve from each seed
3) Cluster the resulting roots to remove duplicates

Across frequencies, we additionally *track* modes by matching roots at ω_i
with roots at ω_{i-1} by nearest-neighbor in the complex k-plane.

This is a pragmatic engineering approach: robust, easy to tweak, and good
for getting several branches without heavy continuation machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from sandwich_wavenumber import (
    Params,
    SolverOpts,
    approx_static_bending_stiffness,
    detM_sandwich,
    solve_k_onefreq,
)

from selection import NormalizeKOpts, normalize_k


@dataclass
class MultiModeOpts:
    max_modes: int = 6
    n_seeds: int = 18

    # Seed generation (real part spans around an initial guess k0)
    span_factor: float = 8.0
    min_factor: float = 0.25

    # Imaginary seed values (added to real seeds)
    im_values: Tuple[float, ...] = (0.0, 1.0, 5.0)

    # Root clustering + tracking
    cluster_tol_rel: float = 1e-3
    match_tol_rel: float = 5e-2

    # Root normalization (recommended)
    normalize_re_nonneg: bool = True
    normalize_im_nonneg: bool = True


def _rel_tol_dist(a: complex, b: complex, tol_rel: float) -> bool:
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol_rel * scale


def _cluster_roots(roots: List[complex], qualities: List[float], tol_rel: float) -> List[complex]:
    """Cluster roots by relative distance; keep best quality (smallest |det|)."""
    kept: List[complex] = []
    kept_q: List[float] = []

    for r, q in sorted(zip(roots, qualities), key=lambda t: t[1]):
        if not np.isfinite(r.real) or not np.isfinite(r.imag):
            continue
        found = False
        for j, rj in enumerate(kept):
            if _rel_tol_dist(r, rj, tol_rel):
                found = True
                # keep the better one
                if q < kept_q[j]:
                    kept[j] = r
                    kept_q[j] = q
                break
        if not found:
            kept.append(r)
            kept_q.append(q)

    # Stable ordering: by increasing Re(k), then Im(k)
    kept.sort(key=lambda z: (float(np.real(z)), float(np.imag(z))))
    return kept


def generate_seeds_for_frequency(
    omega: float,
    p: Params,
    n_seeds: int,
    span_factor: float,
    min_factor: float,
    im_values: Iterable[float],
    k0_imag: float = 0.0,
) -> List[complex]:
    """Generate a grid of initial guesses around a low-frequency bending-like k0."""
    D0 = approx_static_bending_stiffness(p)
    mu_tot = p.skin1.rho * p.skin1.t + p.skin3.rho * p.skin3.t + p.core.rho * p.core.h

    D0r = float(np.real(D0))
    if D0r <= 0:
        D0r = max(float(abs(D0)), 1e-9)

    k0 = (mu_tot * omega**2 / D0r) ** 0.25
    k0 = float(k0)

    # Real-part sweep
    k_re_min = max(1e-9, min_factor * k0)
    k_re_max = max(k_re_min * 1.01, span_factor * k0)

    re_vals = np.geomspace(k_re_min, k_re_max, num=max(2, int(n_seeds)))

    seeds: List[complex] = []
    for re in re_vals:
        for im in im_values:
            seeds.append(complex(float(re), float(im) + float(k0_imag)))

    return seeds


def solve_roots_one_frequency(
    omega: float,
    p: Params,
    solver_opts: SolverOpts,
    seeds: Iterable[complex],
    cluster_tol_rel: float,
    normalize_opts: NormalizeKOpts | None = None,
) -> List[complex]:
    roots: List[complex] = []
    qualities: List[float] = []

    for k0 in seeds:
        try:
            k = solve_k_onefreq(omega, p, k0, solver_opts)
            if normalize_opts is not None:
                k = normalize_k(k, normalize_opts)
            q = abs(detM_sandwich(k, omega, p))
            roots.append(k)
            qualities.append(float(q))
        except Exception:
            # Ignore failed seeds; we just want as many successful roots as possible.
            continue

    if not roots:
        return []

    return _cluster_roots(roots, qualities, tol_rel=float(cluster_tol_rel))


def _match_modes(prev: np.ndarray, curr: List[complex], match_tol_rel: float) -> Tuple[np.ndarray, List[complex]]:
    """Match current roots to previous mode slots by nearest neighbor."""
    out = np.full_like(prev, np.nan + 1j * np.nan)
    remaining = curr.copy()

    for j, pj in enumerate(prev):
        if not (np.isfinite(pj.real) and np.isfinite(pj.imag)):
            continue

        # Find best match
        best_idx = None
        best_dist = None
        for idx, ck in enumerate(remaining):
            dist = abs(ck - pj)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            continue

        ck = remaining[best_idx]
        if _rel_tol_dist(ck, pj, match_tol_rel):
            out[j] = ck
            remaining.pop(best_idx)

    return out, remaining


def multimode_wavenumber(
    freq_hz: Iterable[float],
    p: Params,
    solver_opts: Optional[SolverOpts] = None,
    mm: Optional[MultiModeOpts] = None,
) -> Tuple[np.ndarray, dict]:
    """Compute multiple roots k(freq) and track mode indexing across frequency.

    Returns
    -------
    k_modes: complex array of shape (Nfreq, max_modes). Missing entries are NaN.
    info: dict
    """
    solver_opts = SolverOpts() if solver_opts is None else solver_opts
    mm = MultiModeOpts() if mm is None else mm

    freq = np.asarray(list(freq_hz), dtype=float).reshape(-1)
    N = freq.size

    k_modes = np.full((N, int(mm.max_modes)), np.nan + 1j * np.nan, dtype=np.complex128)

    prev = None

    for i, f in enumerate(freq):
        omega = 2.0 * np.pi * float(f)

        # Base seeds + previous roots (for continuation)
        seeds = generate_seeds_for_frequency(
            omega,
            p,
            n_seeds=int(mm.n_seeds),
            span_factor=float(mm.span_factor),
            min_factor=float(mm.min_factor),
            im_values=mm.im_values,
            k0_imag=float(getattr(solver_opts, "k0_imag", 0.0)),
        )

        if prev is not None:
            seeds = list(prev[np.isfinite(prev.real)]) + seeds

        norm = NormalizeKOpts(
            enforce_re_nonneg=bool(mm.normalize_re_nonneg),
            enforce_im_nonneg=bool(mm.normalize_im_nonneg),
        )

        roots = solve_roots_one_frequency(
            omega,
            p,
            solver_opts=solver_opts,
            seeds=seeds,
            cluster_tol_rel=float(mm.cluster_tol_rel),
            normalize_opts=norm,
        )

        # Track / assign into fixed columns
        if prev is None:
            # First frequency: just take first max_modes roots
            for j, r in enumerate(roots[: int(mm.max_modes)]):
                k_modes[i, j] = r
            prev = k_modes[i].copy()
            continue

        matched, remaining = _match_modes(prev, roots, match_tol_rel=float(mm.match_tol_rel))

        # Fill matched
        k_modes[i] = matched

        # Append new/unmatched modes into free slots
        free = [j for j in range(k_modes.shape[1]) if not np.isfinite(k_modes[i, j].real)]
        for j, r in zip(free, remaining):
            k_modes[i, j] = r

        prev = k_modes[i].copy()

    info = {
        "model": "Core=2D isotropic elasticity (L+T), Skins=Kirchhoff bending + membrane impedance, det(M)=0",
        "tracking": True,
        "max_modes": int(mm.max_modes),
        "n_seeds": int(mm.n_seeds),
    }
    return k_modes, info
