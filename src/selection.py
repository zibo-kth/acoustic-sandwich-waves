"""selection.py

Post-processing utilities for selecting physically consistent roots.

Conventions (repository-wide):
- time dependence: exp(+i ω t)
- spatial dependence: exp(i k x)

Then exp(i k x) = exp(i Re(k) x) * exp(-Im(k) x).
So for a passive medium and propagation along +x, a convenient convention is:
- Re(k) >= 0  (forward-going)
- Im(k) >= 0  (non-negative attenuation)

The dispersion equation often admits symmetry-related solutions (e.g. ±k).
This module picks a preferred representative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class NormalizeKOpts:
    enforce_re_nonneg: bool = True
    enforce_im_nonneg: bool = True


def normalize_k(k: complex, opts: Optional[NormalizeKOpts] = None) -> complex:
    """Choose a physically preferred representative among symmetry-related roots.

    Candidates considered: {k, -k, conj(k), -conj(k)}.

    Scoring prefers:
    1) satisfy Re>=0 if enabled
    2) satisfy Im>=0 if enabled
    3) smallest Im (least attenuation) as tie-breaker
    4) then smallest Re as final tie-breaker

    This is intentionally simple and robust.
    """
    opts = NormalizeKOpts() if opts is None else opts

    cands = [k, -k, np.conj(k), -np.conj(k)]

    def score(z: complex):
        bad = 0
        if opts.enforce_re_nonneg and np.real(z) < 0:
            bad += 1
        if opts.enforce_im_nonneg and np.imag(z) < 0:
            bad += 1
        # Tie-breakers
        return (bad, float(max(0.0, np.imag(z))), float(abs(np.imag(z))), float(abs(np.real(z))))

    best = min(cands, key=score)

    # As a final guard, if we requested Im>=0 and still got Im<0 (pathological), flip it.
    if opts.enforce_im_nonneg and np.imag(best) < 0:
        best = np.conj(best)

    if opts.enforce_re_nonneg and np.real(best) < 0:
        best = -best

    return complex(best)


def pick_flexural_branch(k_roots: Iterable[complex]) -> complex | None:
    """Pick bending-dominated branch heuristically: minimum Re(k) among valid roots.

    Assumes roots are already normalized so that Re>=0, Im>=0.
    Returns None if no valid candidate exists.
    """
    valid = [k for k in k_roots if np.isfinite(np.real(k)) and np.isfinite(np.imag(k)) and np.real(k) > 0]
    if not valid:
        return None
    return min(valid, key=lambda z: float(np.real(z)))
