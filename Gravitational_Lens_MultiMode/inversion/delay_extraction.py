"""Observation-facing Phase 1 time-delay extraction pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from core.light_curve.fluctuation import (
    compute_epsilon,
    compute_sigma_curve,
    find_minima,
    select_best_minimum,
)
from inversion.observation_io import ObservedLensSystem, ObservedLightCurves


def _default_cfg() -> dict[str, Any]:
    return {
        "grid": {
            "dt_try_range": [1.0, 200.0],
            "dt_try_step": 0.1,
            "mu_try_range": [-0.99, 0.99],
            "mu_try_step": 0.02,
        },
        "reconstruction": {
            "series_truncation_tol": 1.0e-6,
            "max_series_terms": 200,
            "interpolation": "cubic",
        },
        "fluctuation": {"diff_order": 1, "detrend": False},
        "selection": {
            "conservative": {"sigma_threshold": -2.0, "require_pair": True},
            "relaxed": {"sigma_threshold": -1.0, "depth_fraction": 0.5},
        },
    }


def _merge_cfg(user_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _default_cfg()
    if not user_cfg:
        return cfg
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def _grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    grid = cfg.get("grid", {})
    dt_min, dt_max = grid.get("dt_try_range", [1.0, 200.0])
    mu_min, mu_max = grid.get("mu_try_range", [-0.99, 0.99])
    dt_step = float(grid.get("dt_try_step", 0.1))
    mu_step = float(grid.get("mu_try_step", 0.02))
    dt_grid = np.round(np.arange(dt_min, dt_max + 0.5 * dt_step, dt_step), 10)
    mu_grid = np.round(np.arange(mu_min, mu_max + 0.5 * mu_step, mu_step), 10)
    if dt_grid.size == 0 or np.any(dt_grid <= 0):
        raise ValueError("dt_try grid must contain positive delays [days]")
    if mu_grid.size == 0 or not np.all(np.abs(mu_grid) < 1.0):
        raise ValueError("|mu_try| < 1 is required for all grid values")
    return dt_grid, mu_grid


def _as_2d(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{name} must have shape (n_series, n_epoch>=3)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or non-finite values")
    return arr


def _joint_light_curve(light_curves: ObservedLightCurves) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    F = _as_2d(light_curves.F, "light_curves.F")
    t = _as_2d(light_curves.t_obs, "light_curves.t_obs")
    sigma = _as_2d(light_curves.sigma_noise, "light_curves.sigma_noise")
    if t.shape[0] == 1 and F.shape[0] > 1:
        t = np.repeat(t, F.shape[0], axis=0)
    if sigma.shape[0] == 1 and F.shape[0] > 1:
        sigma = np.repeat(sigma, F.shape[0], axis=0)
    if F.shape != t.shape or F.shape != sigma.shape:
        raise ValueError("F, t_obs, and sigma_noise must have matching shapes")
    if np.any(sigma <= 0):
        raise ValueError("sigma_noise must be positive")

    order = np.argsort(t[0])
    t_ref = t[0, order]
    if F.shape[0] == 1:
        return t_ref, F[0, order], sigma[0, order]

    if not np.allclose(t[:, order], t_ref[None, :], rtol=0.0, atol=1.0e-8):
        raise ValueError("multi-image light curves must share the same t_obs grid")
    return t_ref, np.sum(F[:, order], axis=0), np.sqrt(np.sum(sigma[:, order] ** 2, axis=0))


def _n_terms(mu_grid: np.ndarray, series_tol: float, max_terms: int) -> int:
    mu_abs = float(np.max(np.abs(mu_grid)))
    if mu_abs == 0.0:
        return 1
    n_terms = int(np.ceil(np.log(series_tol) / np.log(mu_abs))) + 1
    return max(1, min(int(max_terms), n_terms))


def _reconstruct_grid_vectorized(
    F: np.ndarray,
    t: np.ndarray,
    dt_grid: np.ndarray,
    mu_grid: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Vectorized Bag et al. reconstruction over the full (Δt, μ) grid.

    Units: ``t`` and ``dt_grid`` [days], ``F`` arbitrary linear flux, return
    flux cube with shape ``(n_dt, n_mu, n_epoch)``. SIE 표준 근사 가정:
    observation-side preprocessing only; independent of the SIE lens model.
    """

    rec_cfg = cfg.get("reconstruction", cfg)
    series_tol = float(rec_cfg.get("series_truncation_tol", 1.0e-6))
    max_terms = int(rec_cfg.get("max_series_terms", 200))
    interpolation = rec_cfg.get("interpolation", "cubic")
    if interpolation not in {"linear", "cubic"}:
        raise ValueError("interpolation must be 'linear' or 'cubic'")
    if not np.all(np.abs(mu_grid) < 1.0):
        raise ValueError("|mu_try| < 1 is required for series convergence")

    n_terms = _n_terms(mu_grid, series_tol, max_terms)
    term_idx = np.arange(n_terms, dtype=float)
    points = t[None, None, :] - dt_grid[:, None, None] * term_idx[None, :, None]
    kind = "cubic" if interpolation == "cubic" and t.size >= 4 else "linear"
    interpolator = interp1d(t, F, kind=kind, bounds_error=False, fill_value=np.nan)
    values = interpolator(points)
    values = np.where(np.isfinite(values), values, 0.0)
    weights = (-mu_grid[:, None]) ** np.arange(n_terms, dtype=float)[None, :]
    return np.einsum("dnt,mn->dmt", values, weights)


def _local_dt_uncertainty(dt_grid: np.ndarray, sigma_map: np.ndarray, grid_index: tuple[int, int]) -> float:
    row, col = grid_index
    profile = sigma_map[:, col]
    threshold = float(sigma_map[row, col]) + 1.0
    support = np.isfinite(profile) & (profile <= threshold)
    if np.count_nonzero(support) < 2:
        if dt_grid.size < 2:
            return np.nan
        return float(np.median(np.diff(dt_grid)))
    return float(0.5 * (np.max(dt_grid[support]) - np.min(dt_grid[support])))


def extract_delay_from_observation(
    observation: ObservedLensSystem | ObservedLightCurves,
    cfg: dict[str, Any] | None = None,
    *,
    is_mock: bool = False,
    return_grid: bool = True,
) -> dict[str, Any]:
    """Extract observed time delay from an ``ObservedLensSystem`` light curve.

    Units: input times and returned ``dt_obs_days`` are [days], flux is
    arbitrary linear flux, and ``sigma_noise`` is flux uncertainty. SIE 표준
    근사 가정: Phase 1 is observation preprocessing and does not fit a lens;
    its output feeds the fixed SIE Mode 1 path downstream. The Δt/μ sweep is
    evaluated as one vectorized 2D grid, and every μ grid value must satisfy
    ``|μ| < 1``.
    """

    light_curves = observation.light_curves if isinstance(observation, ObservedLensSystem) else observation
    cfg_full = _merge_cfg(cfg)
    dt_grid, mu_grid = _grid(cfg_full)
    t, F_joint, sigma_joint = _joint_light_curve(light_curves)

    cube = _reconstruct_grid_vectorized(F_joint, t, dt_grid, mu_grid, cfg_full)
    epsilon = compute_epsilon(
        cube,
        diff_order=int(cfg_full.get("fluctuation", {}).get("diff_order", 1)),
    )
    sigma_map = compute_sigma_curve(
        epsilon,
        detrend=bool(cfg_full.get("fluctuation", {}).get("detrend", False)),
    )
    selection = cfg_full.get("selection", {})
    conservative = selection.get("conservative", {})
    minima = find_minima(
        sigma_map,
        dt_grid,
        mu_grid,
        sigma_threshold=float(conservative.get("sigma_threshold", -2.0)),
        require_pair=bool(conservative.get("require_pair", True)),
    )
    best = select_best_minimum(minima, selection)

    if best is None:
        result = {
            "dt_obs_days": np.nan,
            "dt_uncertainty_days": np.nan,
            "mu": np.nan,
            "mu_uncertainty": np.nan,
            "confidence_grade": "rejected",
            "sigma_min": float(np.nanmin(sigma_map)),
            "n_minima_found": len(minima),
            "mock": bool(is_mock),
            "diagnostics": {
                "minima": minima,
                "sigma_min": float(np.nanmin(sigma_map)),
            },
        }
    else:
        grid_index = tuple(best["grid_index"])
        dt_unc = _local_dt_uncertainty(dt_grid, sigma_map, grid_index)
        result = {
            "dt_obs_days": float(best["dt"]),
            "dt_uncertainty_days": dt_unc,
            "mu": float(best["mu"]),
            "mu_uncertainty": float(np.median(np.diff(mu_grid))) if mu_grid.size > 1 else np.nan,
            "confidence_grade": best["confidence_grade"],
            "sigma_min": float(best["sigma"]),
            "n_minima_found": len(minima),
            "mock": bool(is_mock),
            "diagnostics": {
                "best_minimum": best,
                "minima": minima,
                "sigma_min": float(best["sigma"]),
            },
        }
    result["dt"] = result["dt_obs_days"]
    if return_grid:
        result["grid"] = {
            "dt_grid": dt_grid,
            "mu_grid": mu_grid,
            "epsilon": epsilon,
            "sigma_map": sigma_map,
        }
        result["diagnostics"]["sigma_map"] = sigma_map
    return result
