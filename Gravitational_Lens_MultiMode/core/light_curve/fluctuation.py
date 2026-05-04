"""Fluctuation statistics for Bag et al. 2022 time-delay extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def compute_epsilon(
    f1_rec_cube: np.ndarray,
    valid_mask: np.ndarray | None = None,
    diff_order: int = 1,
) -> np.ndarray:
    """Compute epsilon on a ``(Delta t, mu)`` grid.

    Units: summed squared flux differences. Standard approximation assumption:
    observational preprocessing only; independent of the SIE lens model.
    """
    diff = np.diff(np.asarray(f1_rec_cube, dtype=float), n=diff_order, axis=-1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        for _ in range(diff_order):
            mask = mask[..., 1:] & mask[..., :-1]
        diff = np.where(mask, diff, np.nan)
        return np.nansum(diff * diff, axis=-1)
    return np.sum(diff * diff, axis=-1)


def compute_sigma_curve(epsilon: np.ndarray, detrend: bool = False) -> np.ndarray:
    """Return ``Sigma(Delta t, mu) = (epsilon - mean) / std`` as a 2D map."""
    eps = np.asarray(epsilon, dtype=float)
    if detrend:
        row_center = np.nanmedian(eps, axis=1, keepdims=True)
        eps = eps - row_center + np.nanmedian(row_center)
    mean = np.nanmean(eps)
    std = np.nanstd(eps)
    return np.zeros_like(eps) if not np.isfinite(std) or std == 0 else (eps - mean) / std


def find_minima(
    sigma_map: np.ndarray,
    dt_grid: np.ndarray,
    mu_grid: np.ndarray,
    sigma_threshold: float = -2.0,
    require_pair: bool = True,
    pair_axis: str = "mu",
    pair_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Find thresholded 2D local minima and optional opposite-sign pairs."""
    sigma = np.asarray(sigma_map, dtype=float)
    dt_grid = np.asarray(dt_grid, dtype=float)
    mu_grid = np.asarray(mu_grid, dtype=float)
    local = sigma == ndimage.minimum_filter(sigma, size=3, mode="nearest")
    below = sigma < sigma_threshold
    rows, cols = np.where(local & below)
    if rows.size == 0:
        return []

    paired = np.zeros(rows.size, dtype=bool)
    partner = np.full(rows.size, -1, dtype=int)
    if require_pair:
        if pair_axis != "mu":
            raise ValueError("Only pair_axis='mu' is supported")
        dt_vals = dt_grid[rows]
        mu_vals = mu_grid[cols]
        sign_opposite = np.sign(mu_vals[:, None]) == -np.sign(mu_vals[None, :])
        dt_close = np.abs(dt_vals[:, None] - dt_vals[None, :]) / np.maximum(
            np.maximum(np.abs(dt_vals[:, None]), np.abs(dt_vals[None, :])), 1e-12
        ) < pair_tolerance
        candidates = sign_opposite & dt_close
        np.fill_diagonal(candidates, False)
        has_pair = candidates.any(axis=1)
        paired = has_pair
        partner = np.where(has_pair, np.argmax(candidates, axis=1), -1)
    else:
        paired[:] = True

    baseline = max(abs(float(np.nanmedian(sigma))), 1.0)
    keep = paired if require_pair else np.ones_like(paired, dtype=bool)
    out = []
    for out_idx, i in enumerate(np.where(keep)[0]):
        out.append(
            {
                "dt": float(dt_grid[rows[i]]),
                "mu": float(mu_grid[cols[i]]),
                "sigma": float(sigma[rows[i], cols[i]]),
                "paired": bool(paired[i]),
                "pair_partner_idx": int(partner[i]),
                "depth_fraction": float(abs(sigma[rows[i], cols[i]]) / baseline),
                "grid_index": (int(rows[i]), int(cols[i])),
                "candidate_index": int(out_idx),
            }
        )
    return sorted(out, key=lambda x: x["sigma"])


def select_best_minimum(
    minima: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply conservative then relaxed selection to local minima."""
    if not minima:
        return None
    cons = selection_cfg.get("conservative", {})
    rel = selection_cfg.get("relaxed", {})
    cons_thr = float(cons.get("sigma_threshold", -2.0))
    require_pair = bool(cons.get("require_pair", True))
    conservative = [
        m for m in minima if m["sigma"] < cons_thr and (m["paired"] or not require_pair)
    ]
    if conservative:
        best = min(conservative, key=lambda m: m["sigma"]).copy()
        best["confidence_grade"] = "conservative"
        return best
    rel_thr = float(rel.get("sigma_threshold", -1.0))
    depth = float(rel.get("depth_fraction", 0.5))
    relaxed = [m for m in minima if m["sigma"] < rel_thr and m["depth_fraction"] >= depth]
    if relaxed:
        best = min(relaxed, key=lambda m: m["sigma"]).copy()
        best["confidence_grade"] = "relaxed"
        return best
    return None
