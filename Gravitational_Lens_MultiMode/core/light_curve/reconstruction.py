"""Bag et al. 2022 light-curve reconstruction.

Units: times are days, flux is arbitrary linear flux. Standard approximation
assumption: observational preprocessing only; independent of the SIE lens model.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def _n_terms(mu_abs: float, series_tol: float, max_terms: int) -> int:
    if mu_abs == 0:
        return 1
    n = int(math.ceil(math.log(series_tol) / math.log(mu_abs))) + 1
    return max(1, min(max_terms, n))


def _interpolator(t: np.ndarray, F: np.ndarray, method: str):
    if method == "cubic" and t.size >= 4:
        return CubicSpline(t, F, extrapolate=False)
    if method in {"linear", "cubic"}:
        return interp1d(t, F, kind="linear", bounds_error=False, fill_value=np.nan)
    raise ValueError(f"Unsupported interpolation: {method}")


def reconstruct_f1(
    F: np.ndarray,
    t: np.ndarray,
    dt_try: float,
    mu_try: float,
    series_tol: float = 1.0e-6,
    max_terms: int = 200,
    boundary: str = "extrapolate_zero",
    interpolation: str = "cubic",
) -> dict[str, np.ndarray | int]:
    """Reconstruct image-1 flux with Bag et al. eq. 2.

    Args:
        F: observed unresolved flux [n].
        t: observation times [days].
        dt_try: trial time delay [days].
        mu_try: trial magnification ratio, must satisfy ``abs(mu_try) < 1``.

    Returns a dict with ``f1_rec`` [n], ``n_terms_used`` and ``valid_mask`` [n].
    """
    assert abs(mu_try) < 1.0, "|mu_try| must be < 1 for series convergence"
    if boundary not in {"extrapolate_zero", "reflect", "drop"}:
        raise ValueError(f"Unsupported boundary: {boundary}")

    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    t_sorted, F_sorted = t[order], F[order]
    n_terms = _n_terms(abs(mu_try), series_tol, max_terms)
    terms = np.arange(n_terms, dtype=float)
    points = t_sorted[:, None] - dt_try * terms[None, :]

    if boundary == "reflect":
        lo, hi = t_sorted[0], t_sorted[-1]
        span = hi - lo
        reflected = lo + np.abs((points - lo) % (2 * span) - span) if span > 0 else points
        values = _interpolator(t_sorted, F_sorted, interpolation)(reflected)
        valid = np.ones_like(values, dtype=bool)
    else:
        values = _interpolator(t_sorted, F_sorted, interpolation)(points)
        valid = np.isfinite(values)
        values = np.where(valid, values, 0.0)

    weights = (-mu_try) ** np.arange(n_terms)
    f_sorted = values @ weights
    valid_mask = valid.all(axis=1) if boundary == "drop" else valid.any(axis=1)

    f1_rec = np.empty_like(f_sorted)
    mask_out = np.empty_like(valid_mask)
    f1_rec[order] = f_sorted
    mask_out[order] = valid_mask
    return {"f1_rec": f1_rec, "n_terms_used": n_terms, "valid_mask": mask_out}


def reconstruct_grid(
    F: np.ndarray,
    t: np.ndarray,
    dt_grid: np.ndarray,
    mu_grid: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Compute ``f1_rec`` over a ``(dt, mu)`` grid.

    Returns:
        Reconstructed cube with shape ``[n_dt, n_mu, n_t]``.
    """
    dt_grid = np.asarray(dt_grid, dtype=float)
    mu_grid = np.asarray(mu_grid, dtype=float)
    assert np.all(np.abs(mu_grid) < 1.0), "|mu_grid| must be < 1"

    rec_cfg = cfg.get("reconstruction", cfg)
    series_tol = float(rec_cfg.get("series_truncation_tol", 1.0e-6))
    max_terms = int(rec_cfg.get("max_series_terms", 200))
    boundary = rec_cfg.get("boundary_strategy", "extrapolate_zero")
    interpolation = rec_cfg.get("interpolation", "cubic")
    if boundary not in {"extrapolate_zero", "drop"}:
        raise ValueError("reconstruct_grid supports extrapolate_zero/drop boundaries")

    F = np.asarray(F, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    t_sorted, F_sorted = t[order], F[order]
    n_terms = _n_terms(float(np.max(np.abs(mu_grid))), series_tol, max_terms)
    term_idx = np.arange(n_terms, dtype=float)
    weights = (-mu_grid[:, None]) ** np.arange(n_terms)[None, :]
    interpolator = _interpolator(t_sorted, F_sorted, interpolation)

    n_dt, n_mu, n_t = dt_grid.size, mu_grid.size, t_sorted.size
    cube_sorted = np.empty((n_dt, n_mu, n_t), dtype=float)
    max_cells = 100_000_000
    dt_chunk = max(1, int(max_cells // max(n_mu * n_t, 1)))

    for start in range(0, n_dt, dt_chunk):
        dt_part = dt_grid[start : start + dt_chunk]
        points = t_sorted[None, None, :] - dt_part[:, None, None] * term_idx[None, :, None]
        values = interpolator(points)
        values = np.where(np.isfinite(values), values, 0.0)
        cube_sorted[start : start + dt_chunk] = np.einsum("dnt,mn->dmt", values, weights)

    cube = np.empty_like(cube_sorted)
    cube[:, :, order] = cube_sorted
    return cube
