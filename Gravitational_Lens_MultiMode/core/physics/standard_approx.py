"""Project-wide SIE standard approximation for Phase 4 catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.physics.config import constants
from core.physics.distances import (
    angular_diameter_distance,
    angular_diameter_distance_between,
)
from core.physics.lens_models import SIELens
from core.physics.ray_tracing import find_images_thin_lens
_IGNORED_TRUTH_KEYS = frozenset({"M200", "concentration", "kappa_ext", "nfw_offset"})
_SIE_IMAGE_DEDUPE_ARCSEC = 0.01
_SIE_ROOT_RESIDUAL_ARCSEC = 1.0e-6


@dataclass(frozen=True)
class ApproxOutputs:
    """SIE 표준 근사 출력.

    Units: ``dt_approx`` [days], ``H0_approx`` [km/s/Mpc], image/source planes
    [arcsec], ``dm_params_approx`` in the stored project order
    ``[theta_E, q, position_angle, sigma_v]``. SIE 표준 근사 가정: 단일 평면,
    κ_ext=0, smooth mass profile, isotropic velocity dispersion.
    """

    dt_approx: float
    H0_approx: float
    dm_params_approx: np.ndarray
    S_approx: np.ndarray | None
    theta_1: np.ndarray
    theta_2: np.ndarray
    fermat_potential: float
    mu_approx: float
    theta_E: float


def reduced_time_delay_distance_h0_one(z_lens: float, z_source: float) -> float:
    """Return ``D_l D_s / D_ls`` [Mpc] at H0=1 for SIE H0 inversion.

    Units: redshifts are dimensionless and the return value is Mpc. SIE 표준
    근사 Mode 1 inversion multiplies this by ``(1 + z_lens)`` and assumes a
    single lens plane with κ_ext=0.
    """

    cosmo = {"H0": 1.0}
    d_l = angular_diameter_distance(z_lens, cosmo)
    d_s = angular_diameter_distance(z_source, cosmo)
    d_ls = angular_diameter_distance_between(z_lens, z_source, cosmo)
    return float(d_l * d_s / d_ls)


def invert_h0_from_delay_sie(
    delta_t_obs_days: float,
    delta_phi_sie_rad2: float,
    z_lens: float,
    z_source: float,
) -> float:
    """Invert H0 [km/s/Mpc] from observed delay under SIE 표준 근사.

    Units: ``delta_t_obs_days`` [days], ``delta_phi_sie_rad2`` [radian^2],
    redshifts dimensionless. SIE 표준 근사 가정: single plane, κ_ext=0, smooth
    SIE mass profile. This is the Phase 4 closed-form Mode 1 path, not MCMC.
    """

    if delta_t_obs_days <= 0:
        raise ValueError("delta_t_obs_days must be positive")
    d_reduced = reduced_time_delay_distance_h0_one(z_lens, z_source)
    numerator = (
        constants()["Mpc_m"]
        * (1.0 + float(z_lens))
        * d_reduced
        * float(delta_phi_sie_rad2)
    )
    denom = constants()["c_m_s"] * constants()["day_s"] * float(delta_t_obs_days)
    return float(numerator / denom)


def _mock_magnification(theta: np.ndarray, theta_e: float) -> np.ndarray:
    radius = np.linalg.norm(np.asarray(theta, dtype=float), axis=-1)
    return 1.0 / np.maximum(np.abs(1.0 - theta_e / np.maximum(radius, 1.0e-3)), 0.05)


def _refine_sie_images(candidates: np.ndarray, lens: SIELens, beta: np.ndarray) -> np.ndarray:
    roots: list[np.ndarray] = []
    try:
        from scipy.optimize import root
    except Exception:
        return np.asarray(candidates, dtype=np.float32)

    for seed in np.asarray(candidates, dtype=float):
        result = root(lambda th: th - lens.deflection(th) - beta, seed, method="hybr")
        theta = np.asarray(result.x, dtype=float)
        residual = theta - lens.deflection(theta) - beta
        if not result.success or not np.isfinite(theta).all() or np.linalg.norm(residual) >= _SIE_ROOT_RESIDUAL_ARCSEC:
            continue
        if any(np.linalg.norm(theta - prev) < _SIE_IMAGE_DEDUPE_ARCSEC for prev in roots):
            continue
        roots.append(theta)
    if len(roots) < 2:
        return np.asarray(candidates, dtype=np.float32)
    return np.stack(roots).astype(np.float32)


def _require_only_public_keys(system: Mapping[str, Any]) -> None:
    for key in _IGNORED_TRUTH_KEYS:
        if key in system:
            raise ValueError(
                f"solve_standard_approx received truth-only key {key!r}; "
                "pass only public SIE approximation inputs"
            )


def render_source_plane_gaussian(
    source_pos_xy: np.ndarray,
    image_size: int = 64,
    pixel_scale: float = 0.1,
    source_size_arcsec: float = 0.05,
) -> np.ndarray:
    """Render a source-plane Gaussian under SIE 표준 근사 bookkeeping.

    Units: source-plane coordinates and ``pixel_scale`` are [arcsec], output is
    dimensionless brightness in [0, 1]. SIE 표준 근사 가정: this source image is
    paired with SIE-only lensing outputs and contains no NFW or κ_ext truth
    information.
    """

    beta = np.asarray(source_pos_xy, dtype=float)
    if beta.shape != (2,):
        raise ValueError("source_pos_xy must have shape (2,) in arcsec")
    fov = int(image_size) * float(pixel_scale)
    axis = np.linspace(-fov / 2.0, fov / 2.0, int(image_size), dtype=float)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    delta2 = (xx - beta[0]) ** 2 + (yy - beta[1]) ** 2
    image = np.exp(-0.5 * delta2 / float(source_size_arcsec) ** 2)
    max_val = float(np.max(image))
    if max_val > 0:
        image = image / max_val
    return image.astype(np.float32)


def solve_standard_approx(system: Mapping[str, Any]) -> ApproxOutputs:
    """Solve one system with the fixed SIE 표준 근사.

    Required input keys: ``H0``, ``z_lens``, ``z_source``, ``sigma_v``, ``q``,
    ``source_pos_xy``. Optional keys: ``image_size``, ``pixel_scale``,
    ``position_angle``, ``delta_t_obs``. Truth-only keys are rejected and never
    read: ``M200``, ``concentration``, ``kappa_ext``, ``nfw_offset``.

    Units: H0 [km/s/Mpc], sigma_v [km/s], angles [arcsec], delays [days],
    source images dimensionless. SIE 표준 근사 가정: single plane, κ_ext=0,
    smooth mass profile, isotropic velocity dispersion. If ``delta_t_obs`` is
    supplied, ``H0_approx`` is obtained by closed-form SIE inversion from that
    observed full-truth delay; otherwise it equals the public input ``H0``.
    """

    _require_only_public_keys(system)
    required = ("H0", "z_lens", "z_source", "sigma_v", "q", "source_pos_xy")
    missing = [key for key in required if key not in system]
    if missing:
        raise KeyError(f"missing required SIE approximation input keys: {missing}")

    h0 = float(system["H0"])
    z_lens = float(system["z_lens"])
    z_source = float(system["z_source"])
    sigma_v = float(system["sigma_v"])
    q = float(system["q"])
    beta = np.asarray(system["source_pos_xy"], dtype=np.float32)
    if beta.shape != (2,):
        raise ValueError("source_pos_xy must have shape (2,) in arcsec")

    position_angle = float(system.get("position_angle", 0.0))
    cosmo = {"H0": h0}
    lens = SIELens(
        sigma_v=sigma_v,
        q=q,
        position_angle=position_angle,
        z_lens=z_lens,
        z_source=z_source,
        cosmology=cosmo,
    )
    theta_e = lens.einstein_radius()
    candidates = find_images_thin_lens(
        beta,
        lens,
        search_box_arcsec=max(3.0, 2.5 * theta_e),
        grid_size=int(system.get("image_grid_size", 96)),
    )
    images = _refine_sie_images(candidates, lens, beta)
    if images.shape[0] < 2:
        raise RuntimeError("SIE standard approximation found fewer than two images")

    mags = _mock_magnification(images, theta_e)
    order = np.argsort(mags)[::-1]
    theta_1 = np.asarray(images[order[0]], dtype=np.float32)
    theta_2 = np.asarray(images[order[1]], dtype=np.float32)
    mu = float(np.clip(mags[order[1]] / max(float(mags[order[0]]), 1.0e-6), 1.0e-4, 0.999))
    if not abs(mu) < 1.0:
        raise ValueError("|mu| < 1 convergence condition failed for SIE approximation")

    phi_1 = float(lens.fermat_potential(theta_1, beta))
    phi_2 = float(lens.fermat_potential(theta_2, beta))
    dphi = abs(phi_1 - phi_2)
    dt_obs = system.get("delta_t_obs")
    if dt_obs is None:
        dt_approx = (
            constants()["Mpc_m"]
            * (1.0 + z_lens)
            * reduced_time_delay_distance_h0_one(z_lens, z_source)
            * dphi
            / (h0 * constants()["c_m_s"] * constants()["day_s"])
        )
        h0_approx = h0
    else:
        dt_approx = float(dt_obs)
        h0_approx = invert_h0_from_delay_sie(float(dt_obs), dphi, z_lens, z_source)

    image_size = int(system.get("image_size", 64))
    pixel_scale = float(system.get("pixel_scale", 0.1))
    s_approx = render_source_plane_gaussian(
        beta,
        image_size=image_size,
        pixel_scale=pixel_scale,
    )

    dm_params = np.array([theta_e, q, position_angle, sigma_v], dtype=np.float32)
    return ApproxOutputs(
        dt_approx=float(dt_approx),
        H0_approx=float(h0_approx),
        dm_params_approx=dm_params,
        S_approx=s_approx,
        theta_1=theta_1,
        theta_2=theta_2,
        fermat_potential=float(dphi),
        mu_approx=mu,
        theta_E=float(theta_e),
    )


def solve_standard_approx_batch(systems: Sequence[Mapping[str, Any]]) -> list[ApproxOutputs]:
    """Solve a batch with the fixed SIE 표준 근사.

    Units and assumptions are identical to :func:`solve_standard_approx`: SIE
    single-plane lens, κ_ext=0, smooth profile, isotropic velocity dispersion.
    """

    return [solve_standard_approx(system) for system in systems]
