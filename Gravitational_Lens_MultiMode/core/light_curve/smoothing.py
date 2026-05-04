"""Shafieloo-style nonparametric light-curve smoothing."""

from __future__ import annotations

import numpy as np


def _silverman(t: np.ndarray, h_min: float, h_max: float) -> float:
    h = 1.06 * np.std(t) * (t.size ** (-1 / 5))
    return float(np.clip(h, h_min, h_max))


def _smooth_update(
    target_t: np.ndarray,
    source_t: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    bandwidth: float,
    chunk_size: int = 5000,
) -> np.ndarray:
    updates = []
    weights_base = 1.0 / np.maximum(sigma, 1e-12) ** 2
    for start in range(0, target_t.size, chunk_size):
        part = target_t[start : start + chunk_size]
        kernel = np.exp(-0.5 * ((part[:, None] - source_t[None, :]) / bandwidth) ** 2)
        weights = kernel * weights_base[None, :]
        norm = np.sum(weights, axis=1)
        updates.append(np.sum(weights * residual[None, :], axis=1) / np.maximum(norm, 1e-12))
    return np.concatenate(updates)


def _loo_cv_bandwidth(t: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> float:
    candidates = np.linspace(1.0, 50.0, 20)
    scores = []
    for h in candidates:
        kernel = np.exp(-0.5 * ((t[:, None] - t[None, :]) / h) ** 2)
        np.fill_diagonal(kernel, 0.0)
        weights = kernel / np.maximum(sigma[None, :], 1e-12) ** 2
        pred = (weights @ y) / np.maximum(weights.sum(axis=1), 1e-12)
        scores.append(np.mean((y - pred) ** 2))
    return float(candidates[int(np.argmin(scores))])


def shafieloo_smooth(
    t: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    n_iter: int = 50,
    bandwidth: float | str = "silverman",
    t_eval: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Smooth a noisy light curve with iterative Shafieloo kernel updates.

    Units: ``t`` and ``t_eval`` are days, ``y`` is arbitrary flux, ``sigma`` is
    flux standard deviation. Standard approximation assumption: observational
    data processing only; independent of the SIE lens model.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    order = np.argsort(t)
    t, y, sigma = t[order], y[order], sigma[order]
    if t_eval is None:
        t_eval_arr = t
    else:
        t_eval_arr = np.asarray(t_eval, dtype=float)

    if isinstance(bandwidth, str):
        if bandwidth == "silverman":
            h = _silverman(t, 1.0, 50.0)
        elif bandwidth == "cv":
            h = _loo_cv_bandwidth(t, y, sigma)
        else:
            raise ValueError(f"Unsupported bandwidth method: {bandwidth}")
    else:
        h = float(bandwidth)

    f_obs = np.full_like(y, np.average(y, weights=1.0 / np.maximum(sigma, 1e-12) ** 2))
    for _ in range(int(n_iter)):
        residual = y - f_obs
        f_obs = f_obs + _smooth_update(t, t, residual, sigma, h)

    residual = y - f_obs
    if t_eval is None:
        f_eval = f_obs
    else:
        f_eval = np.interp(t_eval_arr, t, f_obs) + _smooth_update(t_eval_arr, t, residual, sigma, h)
    out = np.empty_like(f_obs)
    out[order] = f_obs
    return {
        "f_smooth": f_eval,
        "bandwidth": h,
        "residual": residual,
        "residual_std": float(np.std(residual)),
    }
