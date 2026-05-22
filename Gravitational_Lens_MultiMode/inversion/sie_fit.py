"""Fit the fixed SIE standard approximation to observed image positions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from core.physics.config import default_cosmology
from core.physics.lens_models import SIELens


def _validate_inputs(
    image_positions: np.ndarray,
    z_lens: float,
    z_source: float,
) -> tuple[np.ndarray, float, float]:
    positions = np.asarray(image_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("image_positions must have shape (n_img, 2) in arcsec")
    if positions.shape[0] < 2:
        raise ValueError("at least two image positions are required")
    if not np.isfinite(positions).all():
        raise ValueError("image_positions contains NaN or non-finite values")

    z_l = float(z_lens)
    z_s = float(z_source)
    if not np.isfinite([z_l, z_s]).all():
        raise ValueError("z_lens and z_source must be finite")
    if not z_s > z_l + 0.05:
        raise ValueError(f"z_source ({z_s}) must be greater than z_lens ({z_l}) + 0.05")
    return positions, z_l, z_s


def _lens_from_params(
    sigma_v: float,
    q: float,
    position_angle: float,
    z_lens: float,
    z_source: float,
    cosmology: dict[str, float],
) -> SIELens:
    return SIELens(
        sigma_v=float(sigma_v),
        q=float(q),
        position_angle=float(position_angle),
        z_lens=float(z_lens),
        z_source=float(z_source),
        cosmology=cosmology,
    )


def _sigma_from_theta_e(
    theta_e_arcsec: float,
    z_lens: float,
    z_source: float,
    cosmology: dict[str, float],
) -> float:
    target = max(float(theta_e_arcsec), 1.0e-3)

    def residual(sigma_v: float) -> float:
        lens = _lens_from_params(sigma_v, 1.0, 0.0, z_lens, z_source, cosmology)
        return lens.einstein_radius() - target

    try:
        return float(brentq(residual, 50.0, 500.0))
    except ValueError:
        return 220.0


def _initial_theta_e(image_positions: np.ndarray) -> float:
    radii = np.linalg.norm(image_positions, axis=1)
    pairwise = np.linalg.norm(
        image_positions[:, None, :] - image_positions[None, :, :],
        axis=-1,
    )
    max_sep = float(np.max(pairwise))
    radius_scale = float(np.median(radii))
    return max(radius_scale, 0.5 * max_sep, 1.0e-3)


def _principal_angle(image_positions: np.ndarray) -> float:
    centered = image_positions - np.mean(image_positions, axis=0, keepdims=True)
    if image_positions.shape[0] == 2:
        delta = image_positions[1] - image_positions[0]
        return float(np.arctan2(delta[1], delta[0]))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return float(np.arctan2(vh[0, 1], vh[0, 0]))


def _lens_equation_residuals(
    params: np.ndarray,
    image_positions: np.ndarray,
    z_lens: float,
    z_source: float,
    cosmology: dict[str, float],
) -> np.ndarray:
    sigma_v, q, position_angle, beta_x, beta_y = params
    lens = _lens_from_params(sigma_v, q, position_angle, z_lens, z_source, cosmology)
    beta = np.array([beta_x, beta_y], dtype=float)
    residual = image_positions - lens.deflection(image_positions) - beta
    return residual.reshape(-1)


def _magnification_proxy(image_positions: np.ndarray, theta_e_arcsec: float) -> np.ndarray:
    radius = np.linalg.norm(np.asarray(image_positions, dtype=float), axis=-1)
    denom = np.maximum(np.abs(1.0 - theta_e_arcsec / np.maximum(radius, 1.0e-3)), 0.05)
    return 1.0 / denom


def fit_sie_to_images(
    image_positions: np.ndarray,
    z_lens: float,
    z_source: float,
    cosmology: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit SIE parameters to observed image positions and return Δφ.

    Units: ``image_positions`` are [arcsec] with shape ``(n_img, 2)``;
    redshifts are dimensionless; fitted ``sigma_v`` is [km/s];
    ``source_pos_xy`` and returned main image positions are [arcsec];
    ``dphi_rad2`` is the absolute Fermat-potential difference [radian^2].
    SIE 표준 근사 가정: single lens plane, κ_ext=0, smooth SIE mass profile,
    and isotropic velocity dispersion. The fitted unknowns are fixed to
    ``(sigma_v, q, position_angle, source_pos_x, source_pos_y)`` and no
    approximation-selection parameter is accepted.
    """

    positions, z_l, z_s = _validate_inputs(image_positions, z_lens, z_source)
    cosmo = dict(default_cosmology() if cosmology is None else cosmology)
    if "H0" not in cosmo:
        raise KeyError("cosmology must contain H0")

    theta_e0 = _initial_theta_e(positions)
    sigma0 = _sigma_from_theta_e(theta_e0, z_l, z_s, cosmo)
    pa0 = _principal_angle(positions)

    starts: list[np.ndarray] = []
    for q0 in (0.8, 0.7, 0.9, 1.0):
        for pa in (pa0, pa0 + 0.5 * np.pi, 0.0):
            lens0 = _lens_from_params(sigma0, q0, pa, z_l, z_s, cosmo)
            beta0 = np.mean(positions - lens0.deflection(positions), axis=0)
            starts.append(np.array([sigma0, q0, pa, beta0[0], beta0[1]], dtype=float))

    lower = np.array([50.0, 0.2, -np.pi, -10.0, -10.0], dtype=float)
    upper = np.array([500.0, 1.0, np.pi, 10.0, 10.0], dtype=float)
    best = None
    for start in starts:
        start = np.clip(start, lower, upper)
        result = least_squares(
            _lens_equation_residuals,
            x0=start,
            bounds=(lower, upper),
            args=(positions, z_l, z_s, cosmo),
            method="trf",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=2000,
        )
        if best is None or result.cost < best.cost:
            best = result

    if best is None or not best.success:
        raise RuntimeError("SIE fit did not converge")

    sigma_v, q, position_angle, beta_x, beta_y = best.x
    lens = _lens_from_params(sigma_v, q, position_angle, z_l, z_s, cosmo)
    beta = np.array([beta_x, beta_y], dtype=float)
    residual = positions - lens.deflection(positions) - beta
    residual_norm = np.linalg.norm(residual, axis=1)

    theta_e = lens.einstein_radius()
    mags = _magnification_proxy(positions, theta_e)
    order = np.argsort(mags)[::-1]
    theta_1 = positions[order[0]]
    theta_2 = positions[order[1]]
    mu_fit = float(np.clip(mags[order[1]] / max(float(mags[order[0]]), 1.0e-12), 1.0e-6, 0.999999))
    if not abs(mu_fit) < 1.0:
        raise ValueError("|mu| < 1 convergence condition failed for fitted SIE")

    phi_1 = float(lens.fermat_potential(theta_1, beta))
    phi_2 = float(lens.fermat_potential(theta_2, beta))
    dphi_rad2 = abs(phi_1 - phi_2)
    if not np.isfinite(dphi_rad2):
        raise ValueError("fitted SIE produced non-finite dphi_rad2")

    return {
        "success": bool(best.success),
        "cost": float(best.cost),
        "residual_rms_arcsec": float(np.sqrt(np.mean(residual_norm**2))),
        "max_residual_arcsec": float(np.max(residual_norm)),
        "sigma_v": float(sigma_v),
        "q": float(q),
        "position_angle": float(position_angle),
        "source_pos_xy": beta.astype(np.float32),
        "theta_E": float(theta_e),
        "theta_1": theta_1.astype(np.float32),
        "theta_2": theta_2.astype(np.float32),
        "mu_fit": mu_fit,
        "dphi_rad2": float(dphi_rad2),
        "n_images": int(positions.shape[0]),
    }
