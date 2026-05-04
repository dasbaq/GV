"""End-to-end time-delay extraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.light_curve.fluctuation import (
    compute_epsilon,
    compute_sigma_curve,
    find_minima,
    select_best_minimum,
)
from core.light_curve.reconstruction import reconstruct_grid
from core.light_curve.smoothing import shafieloo_smooth


def _grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    grid = cfg.get("grid", {})
    dt_min, dt_max = grid.get("dt_try_range", [1.0, 200.0])
    mu_min, mu_max = grid.get("mu_try_range", [-0.99, 0.99])
    dt_step = float(grid.get("dt_try_step", 0.1))
    mu_step = float(grid.get("mu_try_step", 0.02))
    dt_grid = np.round(np.arange(dt_min, dt_max + 0.5 * dt_step, dt_step), 10)
    mu_grid = np.round(np.arange(mu_min, mu_max + 0.5 * mu_step, mu_step), 10)
    assert np.all(np.abs(mu_grid) < 1.0), "|mu_grid| must be < 1"
    assert np.allclose(mu_grid, -mu_grid[::-1], atol=mu_step), "mu_grid must be symmetric"
    return dt_grid, mu_grid


def _extract_once(
    F: np.ndarray,
    t: np.ndarray,
    sigma_noise: np.ndarray,
    cfg: dict[str, Any],
    return_grid: bool,
) -> dict[str, Any]:
    dt_grid, mu_grid = _grid(cfg)
    y = np.asarray(F, dtype=float)
    if cfg.get("smoothing", {}).get("method") == "shafieloo":
        sm_cfg = cfg.get("smoothing", {})
        smooth = shafieloo_smooth(
            t,
            y,
            sigma_noise,
            n_iter=int(sm_cfg.get("n_iterations", 50)),
            bandwidth=sm_cfg.get("bandwidth_method", "silverman"),
        )
        y = np.asarray(smooth["f_smooth"], dtype=float)

    cube = reconstruct_grid(y, t, dt_grid, mu_grid, cfg)
    epsilon = compute_epsilon(cube, diff_order=int(cfg.get("fluctuation", {}).get("diff_order", 1)))
    sigma_map = compute_sigma_curve(epsilon, detrend=bool(cfg.get("fluctuation", {}).get("detrend", False)))
    minima = find_minima(
        sigma_map,
        dt_grid,
        mu_grid,
        sigma_threshold=float(cfg.get("selection", {}).get("conservative", {}).get("sigma_threshold", -2.0)),
        require_pair=bool(cfg.get("selection", {}).get("conservative", {}).get("require_pair", True)),
    )
    best = select_best_minimum(minima, cfg.get("selection", {}))
    if best is None:
        result = {
            "dt": np.nan,
            "dt_uncertainty": np.nan,
            "mu": np.nan,
            "mu_uncertainty": np.nan,
            "confidence_grade": "rejected",
            "sigma_min": float(np.nanmin(sigma_map)),
            "n_minima_found": len(minima),
        }
    else:
        result = {
            "dt": best["dt"],
            "dt_uncertainty": np.nan,
            "mu": best["mu"],
            "mu_uncertainty": np.nan,
            "confidence_grade": best["confidence_grade"],
            "sigma_min": best["sigma"],
            "n_minima_found": len(minima),
        }
    if return_grid:
        result["grid"] = {
            "dt_grid": dt_grid,
            "mu_grid": mu_grid,
            "epsilon": epsilon,
            "sigma_map": sigma_map,
        }
    return result


def extract_time_delay(
    F: np.ndarray,
    t: np.ndarray,
    sigma_noise: np.ndarray,
    cfg: dict[str, Any],
    return_grid: bool = False,
) -> dict[str, Any]:
    """Extract ``Delta t`` and ``mu`` from an unresolved light curve.

    Units: times are days and flux is arbitrary linear flux. Standard
    approximation assumption: observational processing only; independent of
    the project-wide SIE standard approximation.
    """
    result = _extract_once(F, t, sigma_noise, cfg, return_grid=return_grid)
    n_boot = int(cfg.get("uncertainty", {}).get("bootstrap_n", 0))
    if n_boot > 0 and np.isfinite(result["dt"]):
        boot = bootstrap_uncertainty(
            F,
            t,
            sigma_noise,
            cfg,
            n_boot=n_boot,
            seed=int(cfg.get("seed", 42)),
        )
        result.update(boot)
    return result


def bootstrap_uncertainty(
    F: np.ndarray,
    t: np.ndarray,
    sigma_noise: np.ndarray,
    cfg: dict[str, Any],
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate uncertainty by Gaussian noise realizations."""
    rng = np.random.default_rng(seed)
    dt_samples: list[float] = []
    mu_samples: list[float] = []
    rejected = 0
    boot_cfg = dict(cfg)
    boot_cfg["uncertainty"] = {**cfg.get("uncertainty", {}), "bootstrap_n": 0}
    for _ in range(int(n_boot)):
        noisy = np.asarray(F, dtype=float) + rng.normal(0.0, sigma_noise)
        res = _extract_once(noisy, t, sigma_noise, boot_cfg, return_grid=False)
        if res["confidence_grade"] == "rejected" or not np.isfinite(res["dt"]):
            rejected += 1
        else:
            dt_samples.append(float(res["dt"]))
            mu_samples.append(float(res["mu"]))
    dt_arr = np.asarray(dt_samples, dtype=float)
    mu_arr = np.asarray(mu_samples, dtype=float)
    rejection_rate = rejected / max(int(n_boot), 1)
    return {
        "dt_samples": dt_arr,
        "mu_samples": mu_arr,
        "dt_uncertainty": float(np.std(dt_arr, ddof=1)) if dt_arr.size > 1 else np.nan,
        "mu_uncertainty": float(np.std(mu_arr, ddof=1)) if mu_arr.size > 1 else np.nan,
        "rejection_rate": float(rejection_rate),
        "warning_high_rejection": bool(rejection_rate >= 0.5),
    }
