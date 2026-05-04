"""Effective refractive-index utilities for Phase 2 ray tracing."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.physics.config import constants


def effective_refractive_index(phi: float | np.ndarray, c: float | None = None) -> float | np.ndarray:
    """Convert gravitational potential to effective refractive index.

    Args:
        phi: gravitational potential Phi in SI units [m^2 s^-2].
        c: speed of light [m s^-1]. If omitted, ``config/physics.yaml`` is used.

    Returns:
        Dimensionless ``n_eff = 1 - 2 Phi / c^2`` with scalar or ndarray shape
        matching ``phi``.

    Calculation assumption:
        This is a computational optical-path representation of gravitational
        lensing for efficient Phase 2 ray tracing, not a replacement for GR.
    """
    c_val = float(c if c is not None else constants()["c_m_s"])
    n_eff = 1.0 - 2.0 * np.asarray(phi) / (c_val * c_val)
    return float(n_eff) if np.ndim(phi) == 0 else n_eff


def refractive_index_from_potential(phi: float | np.ndarray, c: float | None = None) -> float | np.ndarray:
    """Alias for :func:`effective_refractive_index`.

    Units and calculation assumptions are identical: Phi is [m^2 s^-2], output
    is dimensionless, and the expression is a numerical n_eff representation.
    """
    return effective_refractive_index(phi, c=c)


def grad_refractive_index_from_grad_phi(grad_phi: np.ndarray, c: float | None = None) -> np.ndarray:
    """Convert potential gradient to refractive-index gradient.

    Args:
        grad_phi: gradient of gravitational potential, ``nabla Phi`` [m s^-2],
            shape ``[..., 3]``.
        c: speed of light [m s^-1]. If omitted, ``config/physics.yaml`` is used.

    Returns:
        ``nabla n_eff`` [m^-1], same shape as ``grad_phi``.

    Calculation assumption:
        Phase 2 effective-index ray tracing uses ``n_eff = 1 - 2 Phi / c^2``.
    """
    c_val = float(c if c is not None else constants()["c_m_s"])
    return -2.0 * np.asarray(grad_phi, dtype=float) / (c_val * c_val)


def optical_path_length(path_positions: np.ndarray, lens: Any) -> float:
    """Integrate optical path length along a ray path.

    Args:
        path_positions: ray samples with shape ``[n, 3]`` in meters.
        lens: object exposing ``effective_refractive_index(position)``.

    Returns:
        Optical path length ``integral n_eff ds`` in meter-equivalent units.

    Calculation assumption:
        The path is already sampled in the effective refractive-index field.
    """
    path = np.asarray(path_positions, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
        raise ValueError("path_positions must have shape [n>=2, 3] in meters")
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    mid = 0.5 * (path[:-1] + path[1:])
    n_eff = np.asarray(lens.effective_refractive_index(mid), dtype=float)
    return float(np.sum(n_eff * ds))


def travel_time_from_path(path_positions: np.ndarray, lens: Any) -> float:
    """Compute travel time from a sampled optical path.

    Args:
        path_positions: ray samples with shape ``[n, 3]`` in meters.
        lens: object exposing ``effective_refractive_index(position)``.

    Returns:
        Travel time ``OPL / c`` in seconds.

    Calculation assumption:
        Uses the Phase 2 ``n_eff`` optical-path approximation.
    """
    return optical_path_length(path_positions, lens) / constants()["c_m_s"]
