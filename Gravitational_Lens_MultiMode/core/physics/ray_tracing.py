"""Effective refractive-index ray tracing for Phase 2."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.physics.config import constants, numerics
from core.physics.refractive_index import optical_path_length


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("direction vectors must be non-zero")
    return arr / np.maximum(norm, 1.0e-300)


def trace_ray_in_refractive_field(
    x0: np.ndarray,
    direction0: np.ndarray,
    lens: Any,
    step_size_m: float | None = None,
    n_steps: int | None = None,
    method: str = "euler",
) -> np.ndarray:
    """Trace a ray through an effective refractive-index field.

    Args:
        x0: initial position [m], shape ``[3]``.
        direction0: initial propagation direction, dimensionless shape ``[3]``.
        lens: object exposing ``effective_refractive_index`` and
            ``grad_refractive_index``.
        step_size_m: integration step length [m]. Defaults to config numerics.
        n_steps: number of integration steps. Defaults to config numerics.
        method: ``"euler"`` or ``"semi_implicit"``.

    Returns:
        Path positions [m], shape ``[n_steps + 1, 3]``.

    Calculation assumption:
        Efficient Phase 2 optical ray tracing in ``n_eff``. This is not a full
        GR geodesic solver.
    """
    if method not in {"euler", "semi_implicit"}:
        raise ValueError("method must be 'euler' or 'semi_implicit'")
    step = float(step_size_m if step_size_m is not None else numerics()["ray_step_m"])
    steps = int(n_steps if n_steps is not None else numerics()["ray_n_steps"])
    if step <= 0 or steps < 1:
        raise ValueError("step_size_m must be positive and n_steps must be >= 1")

    path = np.empty((steps + 1, 3), dtype=float)
    path[0] = np.asarray(x0, dtype=float)
    direction = _normalize(np.asarray(direction0, dtype=float))

    for i in range(steps):
        grad_n = np.asarray(lens.grad_refractive_index(path[i]), dtype=float)
        n_eff = float(np.asarray(lens.effective_refractive_index(path[i])))
        transverse = grad_n - np.dot(grad_n, direction) * direction
        direction = _normalize(direction + step * transverse / max(abs(n_eff), 1.0e-12))
        if method == "semi_implicit":
            path[i + 1] = path[i] + step * direction
        else:
            path[i + 1] = path[i] + step * direction
    return path


def integrate_optical_path(path_positions: np.ndarray, lens: Any) -> float:
    """Integrate ``integral n_eff ds`` along path positions [m].

    Returns:
        Meter-equivalent optical path length.

    Calculation assumption:
        Central Phase 2 n_eff optical-path calculation.
    """
    return optical_path_length(path_positions, lens)


def travel_time_from_path(path_positions: np.ndarray, lens: Any) -> float:
    """Compute travel time from sampled path positions [m].

    Returns:
        Travel time in seconds using ``t = OPL / c``.
    """
    return integrate_optical_path(path_positions, lens) / constants()["c_m_s"]


def time_delay_between_paths(path_a: np.ndarray, path_b: np.ndarray, lens: Any) -> float:
    """Return travel-time difference between two n_eff paths.

    Args:
        path_a: first path [m], shape ``[n, 3]``.
        path_b: second path [m], shape ``[m, 3]``.
        lens: object exposing ``effective_refractive_index``.

    Returns:
        ``travel_time(path_a) - travel_time(path_b)`` in days.
    """
    dt_s = travel_time_from_path(path_a, lens) - travel_time_from_path(path_b, lens)
    return float(dt_s / constants()["day_s"])


def trace_ray_bundle(
    x0_array: np.ndarray,
    direction0_array: np.ndarray,
    lens: Any,
    step_size_m: float | None = None,
    n_steps: int | None = None,
) -> np.ndarray:
    """Trace multiple rays through the same n_eff field.

    Inputs are arrays ``[n_rays, 3]`` in meters/dimensionless directions.
    Returns paths with shape ``[n_rays, n_steps + 1, 3]``. The initial
    implementation loops over rays while preserving an API that can be swapped
    for vectorized integration later.
    """
    x0 = np.asarray(x0_array, dtype=float)
    d0 = np.asarray(direction0_array, dtype=float)
    if x0.shape != d0.shape or x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError("x0_array and direction0_array must both have shape [n_rays, 3]")
    paths = [
        trace_ray_in_refractive_field(x, d, lens, step_size_m=step_size_m, n_steps=n_steps)
        for x, d in zip(x0, d0)
    ]
    return np.stack(paths, axis=0)


def thin_lens_equation(theta: np.ndarray, lens: Any) -> np.ndarray:
    """Thin-lens sanity-check equation ``beta = theta - alpha(theta)``.

    Args:
        theta: image-plane angles [arcsec], shape ``[..., 2]``.
        lens: object exposing analytic sanity-check ``deflection(theta)``.

    Returns:
        Source-plane angle beta [arcsec], same shape as ``theta``.

    Calculation assumption:
        Auxiliary thin-lens helper only; central Phase 2 calculation is n_eff
        ray tracing and optical-path integration.
    """
    th = np.asarray(theta, dtype=float)
    return th - lens.deflection(th)


def thin_lens_time_delay(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    beta: np.ndarray,
    lens: Any,
    z_lens: float,
    z_source: float,
) -> float:
    """Thin-lens analytic sanity-check time delay in days."""
    return lens.analytic_time_delay(theta_a, theta_b, beta, z_lens, z_source)


def find_images_thin_lens(
    beta: np.ndarray,
    lens: Any,
    search_box_arcsec: float,
    grid_size: int = 128,
) -> np.ndarray:
    """Grid-search image candidates for thin-lens sanity checks.

    Args:
        beta: target source angle [arcsec], shape ``[2]``.
        lens: object exposing ``deflection``.
        search_box_arcsec: half-width of square image-plane search box [arcsec].
        grid_size: number of grid points per axis.

    Returns:
        Candidate image angles [arcsec], shape ``[n_candidates, 2]``.

    Calculation assumption:
        Stability-first helper for tests and initial guesses, not a production
        root finder.
    """
    if search_box_arcsec <= 0 or grid_size < 4:
        raise ValueError("search_box_arcsec must be positive and grid_size >= 4")
    beta_arr = np.asarray(beta, dtype=float)
    axis = np.linspace(-search_box_arcsec, search_box_arcsec, grid_size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    theta = np.stack([xx, yy], axis=-1)
    mapped = thin_lens_equation(theta, lens)
    residual = np.linalg.norm(mapped - beta_arr, axis=-1)
    flat = residual.ravel()
    n_pick = min(8, flat.size)
    idx = np.argpartition(flat, n_pick - 1)[:n_pick]
    coords = np.column_stack(np.unravel_index(idx, residual.shape))
    candidates = theta[coords[:, 0], coords[:, 1]]
    order = np.argsort(flat[idx])
    return candidates[order]
