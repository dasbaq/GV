"""Phase 4 full-numerical vs SIE standard-approximation error catalog."""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from core.physics.config import constants, default_cosmology
from core.physics.distances import angular_diameter_distance
from core.physics.lens_models import NFWLens, SIELens
from core.physics.standard_approx import (
    invert_h0_from_delay_sie,
    render_source_plane_gaussian,
    reduced_time_delay_distance_h0_one,
    solve_standard_approx,
)
from ml.training.feature_schema import compute_light_curve_quality, dt_lc_sigma_from_sampler_config
from src_py.simulation.quasar_lc import QuasarLightCurve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_6_PATH = PROJECT_ROOT / "data" / "mock" / "real_phase3_v2_6.h5"
TRUTH_IMAGE_DEDUPE_ARCSEC = 0.01
TRUTH_ROOT_RESIDUAL_ARCSEC = 1.0e-6
TRUTH_ROOT_COND_MAX = 1.0e10
TRUTH_MIN_IMAGE_SEPARATION_ARCSEC = 0.1
TRUTH_MU_MAX = 0.98
TRUTH_H0_APPROX_RANGE = (45.0, 90.0)
TRUTH_DPHI_RATIO_RANGE = (0.5, 1.5)
TRUTH_V0_2_MU_MAX = 0.9699
TRUTH_V0_2_DPHI_RATIO_RANGE = (0.5878, 0.9201)
TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC = 0.6598
TRUTH_V0_2_DT_APPROX_MAX_DAYS = 444.7
TRUTH_V0_2_I_OBS_SUM_MAX = 77.79
TRUTH_V0_2_F_JOINT_ABSMAX_MAX = 3.408
TRUTH_V0_2_MODE1_CORRECTION_ABSMAX = 32.27


def _phase4_v0_3_policy_from_config() -> tuple[tuple[float, float], int]:
    with (PROJECT_ROOT / "config" / "ml.yaml").open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    policy = cfg["catalog"]["phase4_v0_3"]
    h0_range = tuple(float(v) for v in policy["h0_stratified_range"])
    if len(h0_range) != 2:
        raise ValueError("catalog.phase4_v0_3.h0_stratified_range must have two values")
    return (h0_range[0], h0_range[1]), int(policy["h0_stratified_bins"])


TRUTH_V0_3_H0_STRATIFIED_RANGE, TRUTH_V0_3_H0_STRATIFIED_BINS = _phase4_v0_3_policy_from_config()
DEFAULT_RESAMPLE_BUDGET = 50


def _observed_feature_policy_from_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "config" / "ml.yaml").open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return dict(cfg.get("data", {}).get("observed_features", {}))


def _sample_dt_lc_sigma(dt_lc: float, rng: np.random.Generator, policy: dict[str, Any]) -> tuple[float, float]:
    sampler = policy.get("dt_lc_sigma_sampler")
    if not isinstance(sampler, dict):
        raise ValueError("config data.observed_features.dt_lc_sigma_sampler is required")
    return dt_lc_sigma_from_sampler_config(dt_lc, sampler, rng=rng)


@dataclass(frozen=True)
class CatalogConfig:
    """Configuration for Phase 4 catalog generation.

    Units: angles [arcsec], delays [days], H0 [km/s/Mpc], masses [M_sun].
    The full-truth physics is deflection-additive:
    ``alpha_truth = alpha_SIE + alpha_NFW + kappa_ext * theta``. NFW can be
    disabled by flag; using ``M200 -> 0`` as an off-mode is intentionally not
    supported.
    """

    n_systems: int = 50
    seed: int = 42
    image_size: int = 64
    pixel_scale: float = 0.1
    n_epochs: int = 200
    sigma_curve_size: int = 50
    mode2_dm_dim: int = 4
    include_nfw: bool = True
    include_kappa_ext: bool = True
    truth_model: str = "SIE_plus_NFW_plus_LOS_kappa_deflection_additive_v0"
    log_path: Path | None = None
    reject_log_path: Path | None = None
    diagnosis_log_path: Path | None = None
    resample_budget: int | None = None
    validity_filter: str = "v0_2"


class DeflectionAdditiveTruthLens:
    """Truth lens with SIE + NFW + LOS κ_ext deflection.

    Units: input/output angles are [arcsec]. This is a full_numerical truth
    generator for Phase 4 labels, not a new standard approximation. Fermat
    potential uses ``psi = psi_SIE + psi_NFW + 0.5 kappa_ext |theta|^2`` with
    the NFW term obtained by radial numerical integration of the NFW deflection.
    """

    def __init__(
        self,
        sie_lens: SIELens,
        nfw_lens: NFWLens | None,
        kappa_ext: float,
    ) -> None:
        self.sie_lens = sie_lens
        self.nfw_lens = nfw_lens
        self.kappa_ext = float(kappa_ext)
        self._eps = float(np.finfo(float).eps)
        self._arcsec_to_rad = constants()["arcsec_to_rad"]
        self._rad_to_arcsec = constants()["rad_to_arcsec"]
        if nfw_lens is not None:
            z_l = float(sie_lens.z_lens)
            d_l = angular_diameter_distance(z_l, sie_lens.cosmology) * constants()["Mpc_m"]
            self._theta_s_arcsec = nfw_lens.rs / d_l * self._rad_to_arcsec
            c2 = constants()["c_m_s"] ** 2
            d_s = angular_diameter_distance(float(sie_lens.z_source), sie_lens.cosmology) * constants()["Mpc_m"]
            from core.physics.distances import angular_diameter_distance_between

            d_ls = angular_diameter_distance_between(
                float(sie_lens.z_lens),
                float(sie_lens.z_source),
                sie_lens.cosmology,
            ) * constants()["Mpc_m"]
            sigma_crit = c2 / (4.0 * np.pi * constants()["G_si"]) * d_s / (d_l * d_ls)
            self._kappa_s = nfw_lens.rho_s * nfw_lens.rs / sigma_crit
        else:
            self._theta_s_arcsec = 1.0
            self._kappa_s = 0.0

    def _nfw_g(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        near = np.isclose(x, 1.0, rtol=0.0, atol=1.0e-5)
        low = x < 1.0 - 1.0e-5
        high = x > 1.0 + 1.0e-5
        out[near] = 1.0 + np.log(0.5)
        xl = np.maximum(x[low], 1.0e-8)
        out[low] = np.log(xl / 2.0) + 2.0 / np.sqrt(1.0 - xl * xl) * np.arctanh(
            np.sqrt((1.0 - xl) / (1.0 + xl))
        )
        xh = x[high]
        out[high] = np.log(xh / 2.0) + 2.0 / np.sqrt(xh * xh - 1.0) * np.arctan(
            np.sqrt((xh - 1.0) / (1.0 + xh))
        )
        return out

    def nfw_deflection(self, theta: np.ndarray) -> np.ndarray:
        """Return NFW truth deflection [arcsec] for image angles [arcsec]."""

        th = np.asarray(theta, dtype=float)
        if self.nfw_lens is None:
            return np.zeros_like(th)
        r = np.linalg.norm(th, axis=-1, keepdims=True)
        x = np.maximum(r / max(self._theta_s_arcsec, self._eps), 1.0e-8)
        alpha_rad = 4.0 * self._kappa_s * self._theta_s_arcsec * self._arcsec_to_rad * self._nfw_g(x) / x
        alpha_arcsec = alpha_rad * self._rad_to_arcsec
        return alpha_arcsec * th / np.maximum(r, 1.0e-8)

    def deflection(self, theta: np.ndarray) -> np.ndarray:
        """Return truth deflection [arcsec] for image angles [arcsec]."""

        th = np.asarray(theta, dtype=float)
        return self.sie_lens.deflection(th) + self.nfw_deflection(th) + self.kappa_ext * th

    def _nfw_potential(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        if self.nfw_lens is None:
            return np.zeros(th.shape[:-1], dtype=float)
        r = np.linalg.norm(th, axis=-1)
        flat = r.reshape(-1)
        vals = np.zeros_like(flat)
        for i, radius in enumerate(flat):
            if radius <= 0:
                vals[i] = 0.0
                continue
            grid = np.linspace(0.0, radius, 96)
            pts = np.column_stack([grid, np.zeros_like(grid)])
            alpha = self.nfw_deflection(pts)[:, 0] * self._arcsec_to_rad
            vals[i] = np.trapezoid(alpha, grid * self._arcsec_to_rad)
        return vals.reshape(r.shape)

    def fermat_potential(self, theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Return truth Fermat potential [radian^2] for angles [arcsec]."""

        th = np.asarray(theta, dtype=float)
        be = np.asarray(beta, dtype=float)
        th_rad = th * self._arcsec_to_rad
        be_rad = be * self._arcsec_to_rad
        psi_sie = 0.5 * np.sum((th_rad - be_rad) ** 2, axis=-1) - self.sie_lens.fermat_potential(th, be)
        psi_nfw = self._nfw_potential(th)
        psi_kappa = 0.5 * self.kappa_ext * np.sum(th_rad * th_rad, axis=-1)
        return 0.5 * np.sum((th_rad - be_rad) ** 2, axis=-1) - psi_sie - psi_nfw - psi_kappa


def _sigma_v_from_theta_e(theta_e_arcsec: float, z_lens: float, z_source: float) -> float:
    c_m_s = constants()["c_m_s"]
    theta_rad = theta_e_arcsec * constants()["arcsec_to_rad"]
    from core.physics.distances import angular_diameter_distance_between

    d_s = angular_diameter_distance(z_source) * constants()["Mpc_m"]
    d_ls = angular_diameter_distance_between(z_lens, z_source) * constants()["Mpc_m"]
    sigma_m_s = c_m_s * np.sqrt(theta_rad * d_s / (4.0 * np.pi * d_ls))
    return float(sigma_m_s / 1000.0)


def _source_plane_shifted(
    beta: np.ndarray,
    shift: np.ndarray,
    image_size: int,
    pixel_scale: float,
) -> np.ndarray:
    return render_source_plane_gaussian(
        np.asarray(beta, dtype=float) + np.asarray(shift, dtype=float),
        image_size=image_size,
        pixel_scale=pixel_scale,
    )


def _render_lensed_from_deflection(
    lens: Any,
    beta: np.ndarray,
    image_size: int,
    pixel_scale: float,
    source_size_arcsec: float = 0.05,
) -> np.ndarray:
    fov = image_size * pixel_scale
    axis = np.linspace(-fov / 2.0, fov / 2.0, image_size, dtype=float)
    theta_y, theta_x = np.meshgrid(axis, axis, indexing="ij")
    theta = np.stack([theta_x.ravel(), theta_y.ravel()], axis=-1)
    beta_pixel = theta - lens.deflection(theta)
    delta = beta_pixel - np.asarray(beta, dtype=float)
    image = np.exp(-0.5 * np.sum(delta * delta, axis=1) / source_size_arcsec**2)
    image = image.reshape(image_size, image_size)
    max_val = float(np.max(image))
    if max_val > 0.0:
        image = image / max_val
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _joint_curve(flux: np.ndarray, flux_err: np.ndarray, t_obs: np.ndarray, delay: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    delayed = np.interp(t_obs - delay, t_obs, flux, left=flux[0], right=flux[-1])
    joint = flux + mu * delayed
    joint = (joint - np.mean(joint)) / (np.std(joint) + 1.0e-6)
    sigma = np.sqrt(flux_err**2 + (mu * flux_err) ** 2)
    return joint.astype(np.float32), sigma.astype(np.float32)


def _sigma_curve(rng: np.random.Generator, n_points: int, dt_lc: float) -> np.ndarray:
    grid = np.linspace(1.0, 100.0, n_points, dtype=np.float32)
    curve = rng.normal(0.0, 0.15, n_points).astype(np.float32)
    curve -= 2.5 * np.exp(-0.5 * ((grid - np.clip(dt_lc, 1.0, 100.0)) / 4.0) ** 2).astype(np.float32)
    return ((curve - curve.mean()) / (curve.std() + 1.0e-6)).astype(np.float32)


def _sample_base_system(rng: np.random.Generator) -> dict[str, float | np.ndarray]:
    z_l = float(rng.uniform(0.1, 1.0))
    z_s = float(rng.uniform(z_l + 0.1, 3.5))
    theta_e = float(rng.uniform(0.5, 2.0))
    return {
        "H0": float(rng.uniform(60.0, 80.0)),
        "z_lens": z_l,
        "z_source": z_s,
        "theta_E_seed": theta_e,
        "sigma_v": _sigma_v_from_theta_e(theta_e, z_l, z_s),
        "q": float(rng.uniform(0.6, 1.0)),
        "position_angle": float(rng.uniform(0.0, np.pi)),
        "source_pos_xy": rng.uniform(-0.35, 0.35, 2).astype(np.float32),
        "M200": float(10.0 ** rng.uniform(12.0, 14.0)),
        "concentration": float(rng.uniform(3.0, 15.0)),
        "kappa_ext": float(rng.uniform(0.0, 0.1)),
    }


def _sample_base_systems(config: CatalogConfig) -> list[dict[str, float | np.ndarray]]:
    rng = np.random.default_rng(config.seed)
    systems: list[dict[str, float | np.ndarray]] = []
    for _ in range(config.n_systems):
        systems.append(_sample_base_system(rng))
    return systems


def _lens_equation_residual(theta: np.ndarray, lens: Any, beta: np.ndarray) -> np.ndarray:
    return np.asarray(theta, dtype=float) - lens.deflection(theta) - np.asarray(beta, dtype=float)


def _numeric_jacobian(theta: np.ndarray, lens: Any, beta: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    jac = np.zeros((2, 2), dtype=float)
    for col in range(2):
        delta = np.zeros(2, dtype=float)
        delta[col] = step
        fp = _lens_equation_residual(theta_arr + delta, lens, beta)
        fm = _lens_equation_residual(theta_arr - delta, lens, beta)
        jac[:, col] = (fp - fm) / (2.0 * step)
    return jac


def _magnification_abs(theta: np.ndarray, lens: Any, beta: np.ndarray) -> float:
    jac = _numeric_jacobian(theta, lens, beta)
    det = float(np.linalg.det(jac))
    return float(1.0 / max(abs(det), 1.0e-6))


def solve_truth_images_from_sie(
    sie_images: np.ndarray,
    truth_lens: DeflectionAdditiveTruthLens,
    beta: np.ndarray,
    dedupe_arcsec: float = TRUTH_IMAGE_DEDUPE_ARCSEC,
) -> dict[str, Any]:
    """Solve truth image positions from SIE seeds with root finding.

    Units: seeds, roots, and residual norms are [arcsec]. Full-truth assumption:
    deflection-additive SIE + NFW + LOS κ_ext. This v0.1 solver is explicitly
    SIE-anchored and does not search for truth-only extra images.
    """

    try:
        from scipy.optimize import root
    except Exception as exc:  # pragma: no cover - scipy is expected in CI/env
        return {"success": False, "reason": "scipy_unavailable", "error": str(exc)}

    roots: list[np.ndarray] = []
    diagnostics: list[dict[str, float | bool | str]] = []
    dedupe_count = 0
    for seed in np.asarray(sie_images, dtype=float):
        result = root(lambda th: _lens_equation_residual(th, truth_lens, beta), seed, method="hybr")
        theta = np.asarray(result.x, dtype=float)
        residual = _lens_equation_residual(theta, truth_lens, beta)
        residual_norm = float(np.linalg.norm(residual))
        jac = _numeric_jacobian(theta, truth_lens, beta)
        try:
            cond = float(np.linalg.cond(jac))
        except Exception:
            cond = float("inf")
        x_norm = float(np.linalg.norm(theta))
        fun_x_ratio = residual_norm / max(x_norm, 1.0e-12)
        converged = (
            bool(result.success)
            and np.isfinite(theta).all()
            and np.isfinite(residual_norm)
            and residual_norm < TRUTH_ROOT_RESIDUAL_ARCSEC
            and np.isfinite(cond)
            and cond < TRUTH_ROOT_COND_MAX
        )
        diagnostics.append(
            {
                "success": bool(result.success),
                "accepted": bool(converged),
                "residual_norm": residual_norm,
                "condition_number": cond,
                "fun_x_ratio": float(fun_x_ratio),
            }
        )
        if not converged:
            return {
                "success": False,
                "reason": "root_find_residual",
                "diagnostics": diagnostics,
                "residual_norm_max": residual_norm,
                "condition_number_max": cond,
            }
        if any(np.linalg.norm(theta - prev) < dedupe_arcsec for prev in roots):
            dedupe_count += 1
            continue
        roots.append(theta)

    if len(roots) < 2:
        return {
            "success": False,
            "reason": "dedupe_lt2",
            "diagnostics": diagnostics,
            "dedupe_count": dedupe_count,
            "n_unique": len(roots),
        }
    roots_arr = np.stack(roots).astype(np.float32)
    mags = np.array([_magnification_abs(theta, truth_lens, beta) for theta in roots_arr], dtype=float)
    order = np.argsort(mags)[::-1]
    return {
        "success": True,
        "theta": roots_arr[order],
        "magnification_abs": mags[order],
        "diagnostics": diagnostics,
        "dedupe_count": dedupe_count,
        "residual_norm_max": float(max(float(d["residual_norm"]) for d in diagnostics)),
        "residual_norm_median": float(np.median([float(d["residual_norm"]) for d in diagnostics])),
        "condition_number_max": float(max(float(d["condition_number"]) for d in diagnostics)),
        "search_mode": "SIE-anchored search; truth-only extra images are not searched in v0.1",
    }


def _compute_pair(
    base: dict[str, Any],
    config: CatalogConfig,
    include_nfw: bool,
    include_kappa_ext: bool,
    validate: bool = True,
) -> dict[str, Any]:
    h0 = float(base["H0"])
    z_l = float(base["z_lens"])
    z_s = float(base["z_source"])
    cosmo = default_cosmology().copy()
    cosmo["H0"] = h0
    sie = SIELens(
        sigma_v=float(base["sigma_v"]),
        q=float(base["q"]),
        position_angle=float(base["position_angle"]),
        z_lens=z_l,
        z_source=z_s,
        cosmology=cosmo,
    )
    nfw = (
        NFWLens(
            M200=float(base["M200"]),
            concentration=float(base["concentration"]),
            z_lens=z_l,
            z_source=z_s,
            cosmology=cosmo,
        )
        if include_nfw
        else None
    )
    kappa_ext = float(base["kappa_ext"]) if include_kappa_ext else 0.0
    truth_lens = DeflectionAdditiveTruthLens(sie, nfw, kappa_ext)

    public = {
        "H0": h0,
        "z_lens": z_l,
        "z_source": z_s,
        "sigma_v": float(base["sigma_v"]),
        "q": float(base["q"]),
        "position_angle": float(base["position_angle"]),
        "source_pos_xy": np.asarray(base["source_pos_xy"], dtype=np.float32),
        "image_size": config.image_size,
        "pixel_scale": config.pixel_scale,
    }
    approx_no_delay = solve_standard_approx(public)
    beta = np.asarray(base["source_pos_xy"], dtype=np.float32)
    sie_images = np.stack([approx_no_delay.theta_1, approx_no_delay.theta_2]).astype(np.float32)
    truth_solution = solve_truth_images_from_sie(sie_images, truth_lens, beta)
    if not truth_solution["success"]:
        return {"valid": False, "reject_reason": truth_solution["reason"], "truth_solution": truth_solution}

    theta_truth = np.asarray(truth_solution["theta"], dtype=np.float32)
    theta_1_truth = theta_truth[0]
    theta_2_truth = theta_truth[1]
    phi_1_truth = float(truth_lens.fermat_potential(theta_1_truth, beta))
    phi_2_truth = float(truth_lens.fermat_potential(theta_2_truth, beta))
    phi_truth_signed = phi_2_truth - phi_1_truth
    phi_truth = abs(phi_truth_signed)
    phi_sie = float(approx_no_delay.fermat_potential)
    dt_true = (
        constants()["Mpc_m"]
        * (1.0 + z_l)
        * reduced_time_delay_distance_h0_one(z_l, z_s)
        * phi_truth
        / (h0 * constants()["c_m_s"] * constants()["day_s"])
    )
    h0_approx = invert_h0_from_delay_sie(float(dt_true), phi_sie, z_l, z_s)
    approx = approx_no_delay.__class__(
        dt_approx=approx_no_delay.dt_approx,
        H0_approx=h0_approx,
        dm_params_approx=approx_no_delay.dm_params_approx,
        S_approx=approx_no_delay.S_approx,
        theta_1=approx_no_delay.theta_1,
        theta_2=approx_no_delay.theta_2,
        fermat_potential=approx_no_delay.fermat_potential,
        mu_approx=approx_no_delay.mu_approx,
        theta_E=approx_no_delay.theta_E,
    )
    mu_truth = float(
        min(truth_solution["magnification_abs"][1] / max(truth_solution["magnification_abs"][0], 1.0e-6), 0.999)
    )
    sep_truth = float(np.linalg.norm(theta_1_truth - theta_2_truth))
    dphi_ratio = float(phi_sie / phi_truth) if phi_truth > 0 else float("inf")
    image_shift = float(np.mean(np.linalg.norm(theta_truth[:2] - sie_images[:2], axis=1)))

    source_true = render_source_plane_gaussian(beta, config.image_size, config.pixel_scale)
    source_shift = 0.04 * kappa_ext * beta
    if include_nfw:
        source_shift = source_shift + 0.002 * np.log10(float(base["M200"]) / 1.0e12)
    source_approx = _source_plane_shifted(beta, source_shift, config.image_size, config.pixel_scale)
    i_obs = _render_lensed_from_deflection(truth_lens, beta, config.image_size, config.pixel_scale)
    validity_mode = _validity_mode(config)
    invalid_reason = None
    if validity_mode != "off":
        invalid_reason = _validity_reject_reason(
            dt_true=float(dt_true),
            dt_approx=float(approx.dt_approx),
            h0_true=h0,
            h0_approx=float(h0_approx),
            phi_truth=float(phi_truth),
            phi_sie=float(phi_sie),
            mu_truth=mu_truth,
            separation_truth=sep_truth,
            dphi_ratio=dphi_ratio,
            i_obs_sum=float(np.sum(i_obs)),
            apply_v0_2_filters=validity_mode == "v0_2",
            validity_filter=validity_mode,
        )
    if validate and invalid_reason is not None:
        return {
            "valid": False,
            "reject_reason": invalid_reason,
            "truth_solution": truth_solution,
            "dt_true": float(dt_true),
            "dt_approx": float(approx.dt_approx),
            "H0_approx": float(h0_approx),
            "mu_truth": mu_truth,
            "separation_truth": sep_truth,
            "dphi_ratio": dphi_ratio,
            "I_obs_sum": float(np.sum(i_obs)),
        }

    return {
        "valid": True,
        "approx": approx,
        "truth_lens": truth_lens,
        "dt_true": float(dt_true),
        "mu_true": mu_truth,
        "theta_E": approx.theta_E,
        "H0_true": h0,
        "dm_params_true": np.array(
            [approx.theta_E, float(base["q"]), float(base["position_angle"]), float(base["sigma_v"])],
            dtype=np.float32,
        ),
        "D_delta_t": (1.0 + z_l) * reduced_time_delay_distance_h0_one(z_l, z_s) / h0,
        "S_true": source_true,
        "S_approx": source_approx,
        "I_obs": i_obs,
        "theta_truth_1": theta_1_truth,
        "theta_truth_2": theta_2_truth,
        "theta_approx_1": approx_no_delay.theta_1,
        "theta_approx_2": approx_no_delay.theta_2,
        "truth_fermat_potential": float(phi_truth),
        "truth_fermat_signed": float(phi_truth_signed),
        "dphi_ratio": dphi_ratio,
        "image_shift_arcsec": image_shift,
        "truth_solution": truth_solution,
        "separation_truth": sep_truth,
    }


def _baseline_v2_6_std() -> dict[str, Any]:
    if not V2_6_PATH.exists():
        return {"status": "baseline_unavailable", "path": str(V2_6_PATH)}
    with h5py.File(V2_6_PATH, "r") as f:
        arr = np.asarray(f["simplification_errors/mode1_H0_error"], dtype=float)
    return {"status": "ok", "path": str(V2_6_PATH), "v2_6_baseline_std": float(np.std(arr))}


def _distribution(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _validity_mode(config: CatalogConfig) -> str:
    mode = str(config.validity_filter)
    if mode not in {"v0_4", "v0_3_1", "v0_3", "v0_2", "v0_1", "off"}:
        raise ValueError(f"Unsupported validity_filter={mode!r}; expected v0_4, v0_3_1, v0_3, v0_2, v0_1, or off.")
    if mode == "v0_2" and not (config.include_nfw or config.include_kappa_ext):
        return "v0_1"
    return mode


def _validity_reject_reason(
    *,
    dt_true: float,
    dt_approx: float,
    h0_true: float,
    h0_approx: float,
    phi_truth: float,
    phi_sie: float,
    mu_truth: float,
    separation_truth: float,
    dphi_ratio: float,
    i_obs_sum: float | None = None,
    f_joint_absmax: float | None = None,
    apply_v0_2_filters: bool = True,
    validity_filter: str | None = None,
) -> str | None:
    """Return the first Phase 4 validity rejection reason, or ``None``.

    Units: delays [days], H0 [km/s/Mpc], image separation [arcsec].
    ``v0_3`` keeps only H0-neutral physical validity checks here; its H0
    distribution matching is handled by stratified quota in pair collection.
    ``v0_3_1`` keeps the same H0-neutral quota but restores only input-side
    v0.2 CUDA-stability p99 filters. ``v0_1`` retains the legacy
    H0_approx/dphi gates, and ``v0_2`` composes CUDA-stability p99 filters
    after those legacy checks.
    """

    mode = validity_filter
    if mode is None:
        mode = "v0_2" if apply_v0_2_filters else "v0_1"
    if mode not in {"v0_4", "v0_3_1", "v0_3", "v0_2", "v0_1"}:
        raise ValueError(f"Unsupported validity_filter={mode!r}")
    if not (np.isfinite(dt_true) and np.isfinite(h0_approx) and np.isfinite(phi_truth) and np.isfinite(phi_sie)):
        return "nonfinite_values"
    if dt_true <= 0.0:
        return "dt_true_nonpositive"
    if abs(mu_truth) >= TRUTH_MU_MAX:
        return "mu_truth_ge_0p98"
    if separation_truth < TRUTH_MIN_IMAGE_SEPARATION_ARCSEC:
        return "image_separation_lt_0p1"
    # v0_4: 물리 validity only (root/finite/dt>0/|mu|<0.98/separation>=0.1).
    # label-상관 cap, tail cap, H0 stratified quota를 모두 제거해 train 분포를
    # unfiltered(root-converged) 분포와 일치시켜 selection bias를 구조적으로 없앤다.
    if mode in {"v0_3", "v0_4"}:
        return None
    if mode == "v0_3_1":
        if abs(mu_truth) > TRUTH_V0_2_MU_MAX:
            return "mu_truth_gt_v0_2_p99"
        if separation_truth < TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC:
            return "image_separation_lt_v0_2_p01"
        if not (TRUTH_V0_2_DPHI_RATIO_RANGE[0] <= dphi_ratio <= TRUTH_V0_2_DPHI_RATIO_RANGE[1]):
            return "dphi_ratio_outside_v0_2_p01_p99"
        if dt_approx > TRUTH_V0_2_DT_APPROX_MAX_DAYS:
            return "dt_approx_gt_v0_2_p99"
        if i_obs_sum is not None and i_obs_sum > TRUTH_V0_2_I_OBS_SUM_MAX:
            return "image_sum_gt_v0_2_p99"
        if f_joint_absmax is not None and f_joint_absmax > TRUTH_V0_2_F_JOINT_ABSMAX_MAX:
            return "lc_absmax_gt_v0_2_p99"
        return None
    if not (TRUTH_H0_APPROX_RANGE[0] <= h0_approx <= TRUTH_H0_APPROX_RANGE[1]):
        return "H0_approx_outside_45_90"
    if not (TRUTH_DPHI_RATIO_RANGE[0] <= dphi_ratio <= TRUTH_DPHI_RATIO_RANGE[1]):
        return "dphi_ratio_outside_0p5_1p5"
    if mode != "v0_2":
        return None

    correction = float(h0_true) - float(h0_approx)
    if abs(mu_truth) > TRUTH_V0_2_MU_MAX:
        return "mu_truth_gt_v0_2_p99"
    if separation_truth < TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC:
        return "image_separation_lt_v0_2_p01"
    if not (TRUTH_V0_2_DPHI_RATIO_RANGE[0] <= dphi_ratio <= TRUTH_V0_2_DPHI_RATIO_RANGE[1]):
        return "dphi_ratio_outside_v0_2_p01_p99"
    if dt_approx > TRUTH_V0_2_DT_APPROX_MAX_DAYS:
        return "dt_approx_gt_v0_2_p99"
    if abs(correction) > TRUTH_V0_2_MODE1_CORRECTION_ABSMAX + 1.0e-9:
        return "mode1_correction_abs_gt_v0_2_p99"
    if i_obs_sum is not None and i_obs_sum > TRUTH_V0_2_I_OBS_SUM_MAX:
        return "image_sum_gt_v0_2_p99"
    if f_joint_absmax is not None and f_joint_absmax > TRUTH_V0_2_F_JOINT_ABSMAX_MAX:
        return "lc_absmax_gt_v0_2_p99"
    return None


def _default_log_paths(output_path: Path, config: CatalogConfig) -> tuple[Path, Path, Path]:
    if config.log_path is not None:
        log_path = Path(config.log_path)
    elif output_path.resolve() == (PROJECT_ROOT / "data" / "mock" / "phase4_v0.h5").resolve():
        log_path = PROJECT_ROOT / "data" / "logs" / "phase4_v0_label_distribution.json"
    elif output_path.resolve() == (PROJECT_ROOT / "data" / "mock" / "phase4_v0_1.h5").resolve():
        log_path = PROJECT_ROOT / "data" / "logs" / "phase4_v0_1_label_distribution.json"
    elif output_path.resolve() == (PROJECT_ROOT / "data" / "mock" / "phase4_v0_2.h5").resolve():
        log_path = PROJECT_ROOT / "data" / "logs" / "phase4_v0_2_label_distribution.json"
    elif output_path.resolve() == (PROJECT_ROOT / "data" / "mock" / "phase4_v0_3.h5").resolve():
        log_path = PROJECT_ROOT / "data" / "logs" / "phase4_v0_3_label_distribution.json"
    elif output_path.resolve() == (PROJECT_ROOT / "data" / "mock" / "phase4_v0_3_1.h5").resolve():
        log_path = PROJECT_ROOT / "data" / "logs" / "phase4_v0_3_1_label_distribution.json"
    elif output_path.resolve() == (PROJECT_ROOT / "data" / "mock" / "phase4_v0_4.h5").resolve():
        log_path = PROJECT_ROOT / "data" / "logs" / "phase4_v0_4_label_distribution.json"
    else:
        log_path = output_path.with_suffix(".label_distribution.json")
    reject_path = (
        Path(config.reject_log_path)
        if config.reject_log_path is not None
        else log_path.with_name(log_path.stem.replace("label_distribution", "reject_log") + log_path.suffix)
    )
    diagnosis_path = (
        Path(config.diagnosis_log_path)
        if config.diagnosis_log_path is not None
        else PROJECT_ROOT / "data" / "logs" / "phase4_v0_diagnosis.json"
    )
    return log_path, reject_path, diagnosis_path


def _resample_budget(config: CatalogConfig) -> int:
    if config.resample_budget is not None:
        return int(config.resample_budget)
    return int(os.environ.get("LENS_RESAMPLE_BUDGET", DEFAULT_RESAMPLE_BUDGET))


def _empty_reject_log(config: CatalogConfig) -> dict[str, Any]:
    validity_mode = _validity_mode(config)
    return {
        "n_systems_target": config.n_systems,
        "n_achieved": 0,
        "resample_budget_per_system": _resample_budget(config),
        "validity_filter": validity_mode,
        "search_mode": "SIE-anchored search; truth-only extra images are not searched in v0.1",
        "v0_2_thresholds": {
            "mu_truth_abs_max": TRUTH_V0_2_MU_MAX,
            "dphi_sie_over_truth_range": TRUTH_V0_2_DPHI_RATIO_RANGE,
            "truth_image_separation_min_arcsec": TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC,
            "dt_approx_max_days": TRUTH_V0_2_DT_APPROX_MAX_DAYS,
            "I_obs_sum_max": TRUTH_V0_2_I_OBS_SUM_MAX,
            "F_joint_absmax_max": TRUTH_V0_2_F_JOINT_ABSMAX_MAX,
            "mode1_H0_correction_absmax": TRUTH_V0_2_MODE1_CORRECTION_ABSMAX,
        },
        "v0_3_policy": {
            "validity": "root/finite/dt_positive/mu_abs_lt_0p98/separation_ge_0p1",
            "removed_label_dependent_gates": [
                "H0_approx_outside_45_90",
                "mode1_correction_abs_gt_v0_2_p99",
            ],
            "removed_support_tail_gates": [
                "dphi_ratio_outside_0p5_1p5",
                "dphi_ratio_outside_v0_2_p01_p99",
                "dt_approx_gt_v0_2_p99",
                "image_sum_gt_v0_2_p99",
                "lc_absmax_gt_v0_2_p99",
                "image_separation_lt_v0_2_p01",
                "mu_truth_gt_v0_2_p99",
            ],
            "h0_stratified_range": TRUTH_V0_3_H0_STRATIFIED_RANGE,
            "h0_stratified_bins": TRUTH_V0_3_H0_STRATIFIED_BINS,
        },
        "v0_3_1_policy": {
            "validity": (
                "v0_3 H0-neutral physical checks plus v0_2 input-side tail gates; "
                "label-dependent H0_approx/correction gates remain removed"
            ),
            "kept_h0_neutral_quota_from": "v0_3",
            "restored_input_tail_gates": [
                "mu_truth_gt_v0_2_p99",
                "image_separation_lt_v0_2_p01",
                "dphi_ratio_outside_v0_2_p01_p99",
                "dt_approx_gt_v0_2_p99",
                "image_sum_gt_v0_2_p99",
                "lc_absmax_gt_v0_2_p99",
            ],
            "excluded_label_dependent_gates": [
                "H0_approx_outside_45_90",
                "mode1_correction_abs_gt_v0_2_p99",
            ],
        },
        "dedupe_arcsec": TRUTH_IMAGE_DEDUPE_ARCSEC,
        "residual_arcsec_threshold": TRUTH_ROOT_RESIDUAL_ARCSEC,
        "condition_number_max": TRUTH_ROOT_COND_MAX,
        "reject_counts": {},
        "attempts_per_accepted": [],
        "dedupe_count_total": 0,
        "root_residual_norms": [],
        "condition_numbers": [],
    }


def _record_reject(log: dict[str, Any], reason: str) -> None:
    counts = log["reject_counts"]
    counts[reason] = int(counts.get(reason, 0)) + 1


def _h0_bin_index(h0: float) -> int:
    low, high = TRUTH_V0_3_H0_STRATIFIED_RANGE
    if not (low <= h0 <= high):
        return -1
    scaled = (h0 - low) / (high - low)
    idx = int(np.floor(scaled * TRUTH_V0_3_H0_STRATIFIED_BINS))
    return min(max(idx, 0), TRUTH_V0_3_H0_STRATIFIED_BINS - 1)


def _h0_bin_quotas(n_systems: int) -> np.ndarray:
    bins = TRUTH_V0_3_H0_STRATIFIED_BINS
    quotas = np.full(bins, n_systems // bins, dtype=int)
    quotas[: n_systems % bins] += 1
    return quotas


def _record_pair_diagnostics(reject_log: dict[str, Any], pair: dict[str, Any], attempt: int) -> None:
    reject_log["attempts_per_accepted"].append(attempt)
    sol = pair["truth_solution"]
    reject_log["dedupe_count_total"] += int(sol.get("dedupe_count", 0))
    reject_log["root_residual_norms"].extend(float(d["residual_norm"]) for d in sol.get("diagnostics", []))
    reject_log["condition_numbers"].extend(float(d["condition_number"]) for d in sol.get("diagnostics", []))


def _attach_light_curve(
    pair: dict[str, Any],
    config: CatalogConfig,
    lc_rng: np.random.Generator,
    reject_log: dict[str, Any],
    validity_mode: str,
) -> bool:
    lc = QuasarLightCurve(seed=int(lc_rng.integers(0, 2**31 - 1))).generate(
        n_epochs=config.n_epochs,
        total_days=1000.0,
        survey="ztf",
    )
    joint, noise = _joint_curve(lc["flux"], lc["flux_err"], lc["t_obs"], pair["dt_true"], pair["mu_true"])
    sigma_curve = _sigma_curve(lc_rng, config.sigma_curve_size, pair["dt_true"])
    lc_invalid_reason = None
    if validity_mode != "off":
        lc_invalid_reason = _validity_reject_reason(
            dt_true=float(pair["dt_true"]),
            dt_approx=float(pair["approx"].dt_approx),
            h0_true=float(pair["H0_true"]),
            h0_approx=float(pair["approx"].H0_approx),
            phi_truth=float(pair["truth_fermat_potential"]),
            phi_sie=float(pair["approx"].fermat_potential),
            mu_truth=float(pair["mu_true"]),
            separation_truth=float(pair["separation_truth"]),
            dphi_ratio=float(pair["dphi_ratio"]),
            i_obs_sum=float(np.sum(pair["I_obs"])),
            f_joint_absmax=float(np.max(np.abs(joint))),
            apply_v0_2_filters=validity_mode == "v0_2",
            validity_filter=validity_mode,
        )
    if lc_invalid_reason is not None:
        _record_reject(reject_log, lc_invalid_reason)
        return False
    pair["F_joint"] = joint
    pair["sigma_noise"] = noise
    pair["t_obs"] = lc["t_obs"]
    pair["sigma_curve"] = sigma_curve
    pair["dt_lc"] = abs(float(pair["dt_true"]))
    obs_policy = _observed_feature_policy_from_config()
    dt_sigma, rel_sigma = _sample_dt_lc_sigma(float(pair["dt_lc"]), lc_rng, obs_policy)
    pair["dt_lc_sigma"] = dt_sigma
    pair["dt_lc_sigma_relative_error"] = rel_sigma
    pair["light_curve_quality"] = compute_light_curve_quality(
        t_obs=lc["t_obs"],
        sigma_noise=noise,
        n_epochs=config.n_epochs,
    )
    return True


def _collect_valid_pairs_stratified(
    config: CatalogConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    lc_rng = np.random.default_rng(config.seed + 1000)
    pairs: list[dict[str, Any]] = []
    bases: list[dict[str, Any]] = []
    reject_log = _empty_reject_log(config)
    validity_mode = _validity_mode(config)
    quotas = _h0_bin_quotas(config.n_systems)
    counts = np.zeros_like(quotas)
    max_attempts = config.n_systems * _resample_budget(config)
    attempts_since_accept = 0
    for _ in range(max_attempts):
        if len(pairs) >= config.n_systems:
            break
        attempts_since_accept += 1
        base = _sample_base_system(rng)
        bin_idx = _h0_bin_index(float(base["H0"]))
        if bin_idx < 0:
            _record_reject(reject_log, "h0_outside_v0_3_stratified_range")
            continue
        if counts[bin_idx] >= quotas[bin_idx]:
            _record_reject(reject_log, "h0_stratified_bin_full")
            continue
        pair = _compute_pair(base, config, config.include_nfw, config.include_kappa_ext, validate=True)
        if not pair.get("valid"):
            _record_reject(reject_log, str(pair.get("reject_reason", "unknown")))
            continue
        if not _attach_light_curve(pair, config, lc_rng, reject_log, validity_mode):
            continue
        bases.append(base)
        pairs.append(pair)
        counts[bin_idx] += 1
        _record_pair_diagnostics(reject_log, pair, attempts_since_accept)
        attempts_since_accept = 0
    if len(pairs) < config.n_systems:
        reject_log["n_achieved"] = len(pairs)
        raise RuntimeError(
            f"Phase 4 {validity_mode} stratified resample budget exceeded; "
            f"n_systems_target={config.n_systems}, n_achieved={len(pairs)}, "
            f"h0_bin_counts={counts.tolist()}, h0_bin_quotas={quotas.tolist()}"
        )
    reject_log["h0_stratified_counts"] = counts.astype(int).tolist()
    reject_log["h0_stratified_quotas"] = quotas.astype(int).tolist()
    reject_log["h0_stratified_edges"] = np.linspace(
        TRUTH_V0_3_H0_STRATIFIED_RANGE[0],
        TRUTH_V0_3_H0_STRATIFIED_RANGE[1],
        TRUTH_V0_3_H0_STRATIFIED_BINS + 1,
    ).astype(float).tolist()
    reject_log["n_achieved"] = len(pairs)
    attempts = np.asarray(reject_log["attempts_per_accepted"], dtype=float)
    residuals = np.asarray(reject_log["root_residual_norms"], dtype=float)
    conds = np.asarray(reject_log["condition_numbers"], dtype=float)
    reject_log["attempts_mean"] = float(np.mean(attempts)) if attempts.size else 0.0
    reject_log["attempts_max"] = int(np.max(attempts)) if attempts.size else 0
    reject_log["root_residual_norm_median"] = float(np.median(residuals)) if residuals.size else 0.0
    reject_log["root_residual_norm_max"] = float(np.max(residuals)) if residuals.size else 0.0
    reject_log["condition_number_median"] = float(np.median(conds)) if conds.size else 0.0
    reject_log["condition_number_max_observed"] = float(np.max(conds)) if conds.size else 0.0
    return bases, pairs, reject_log


def _collect_valid_pairs(config: CatalogConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    lc_rng = np.random.default_rng(config.seed + 1000)
    validity_mode = _validity_mode(config)
    if validity_mode in {"v0_3", "v0_3_1"}:
        return _collect_valid_pairs_stratified(config)
    pairs: list[dict[str, Any]] = []
    bases: list[dict[str, Any]] = []
    reject_log = _empty_reject_log(config)
    budget = _resample_budget(config)
    for system_idx in range(config.n_systems):
        accepted = False
        for attempt in range(1, budget + 1):
            base = _sample_base_system(rng)
            pair = _compute_pair(base, config, config.include_nfw, config.include_kappa_ext, validate=True)
            if pair.get("valid"):
                if not _attach_light_curve(pair, config, lc_rng, reject_log, validity_mode):
                    continue
                bases.append(base)
                pairs.append(pair)
                _record_pair_diagnostics(reject_log, pair, attempt)
                accepted = True
                break
            _record_reject(reject_log, str(pair.get("reject_reason", "unknown")))
        if not accepted:
            reject_log["n_achieved"] = len(pairs)
            raise RuntimeError(
                f"Phase 4 resample budget exceeded at system {system_idx}; "
                f"n_systems_target={config.n_systems}, n_achieved={len(pairs)}"
            )
    reject_log["n_achieved"] = len(pairs)
    attempts = np.asarray(reject_log["attempts_per_accepted"], dtype=float)
    residuals = np.asarray(reject_log["root_residual_norms"], dtype=float)
    conds = np.asarray(reject_log["condition_numbers"], dtype=float)
    reject_log["attempts_mean"] = float(np.mean(attempts)) if attempts.size else 0.0
    reject_log["attempts_max"] = int(np.max(attempts)) if attempts.size else 0
    reject_log["root_residual_norm_median"] = float(np.median(residuals)) if residuals.size else 0.0
    reject_log["root_residual_norm_max"] = float(np.max(residuals)) if residuals.size else 0.0
    reject_log["condition_number_median"] = float(np.median(conds)) if conds.size else 0.0
    reject_log["condition_number_max_observed"] = float(np.max(conds)) if conds.size else 0.0
    return bases, pairs, reject_log


def write_phase4_v0_diagnosis(path: Path | None = None) -> dict[str, Any]:
    """Write read-only diagnosis for v0 SIE-position artifact and stale tests."""

    out_path = Path(path) if path is not None else PROJECT_ROOT / "data" / "logs" / "phase4_v0_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v0_path = PROJECT_ROOT / "data" / "mock" / "phase4_v0.h5"
    diagnosis: dict[str, Any] = {
        "phase4_v0_path": str(v0_path),
        "d4_legacy_test_classification": {
            "A_param_image_fixture_stale": {
                "count": 8,
                "tests": ["tests/test_corrector.py::6", "tests/test_encoders.py::2"],
            },
            "B_legacy_mock_hdf5_schema": {
                "count": 11,
                "tests": ["tests/test_dataset.py::7", "tests/test_trainer.py::4"],
            },
            "C_phase4_regression": {"count": 0, "tests": []},
        },
        "e3_n_systems": None,
    }
    if v0_path.exists():
        with h5py.File(v0_path, "r") as f:
            corr = np.asarray(f["correction_targets/mode1_H0_correction"], dtype=float)
            outlier_idx = np.where(np.abs(corr) > 30.0)[0]
            diagnosis["e3_n_systems"] = int(f["metadata"].attrs["n_systems"])
            diagnosis["d1_outliers"] = {
                "threshold_abs_km_s_mpc": 30.0,
                "count": int(outlier_idx.size),
                "fraction": float(outlier_idx.size / max(corr.size, 1)),
                "indices": outlier_idx.astype(int).tolist(),
            }
            diagnosis["d2_conclusion"] = (
                "v0 evaluated truth Fermat at SIE image positions; image_shift=0 across toggles, "
                "so broad labels/cross_term are SIE-position artifact rather than full numerical truth."
            )
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(diagnosis, fp, indent=2, sort_keys=True)
    return diagnosis


def build_phase4_catalog(output_path: Path, config: CatalogConfig = CatalogConfig()) -> dict[str, Any]:
    """Build a Phase 4 HDF5 catalog with correction targets ``true - approx``.

    Units follow ARCHITECTURE.md. Full truth is SIE + NFW + LOS κ_ext with
    deflection-additive physics; standard approximation is fixed SIE, κ_ext=0.
    v2.* scalers/checkpoints are not read or loaded. Any ML smoke must create a
    Phase 4-specific scaler such as ``target_scaler_phase4_v0.pkl`` or use
    identity scaling.
    """

    output_path = Path(output_path)
    validity_mode = _validity_mode(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path, reject_path, diagnosis_path = _default_log_paths(output_path, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
    bases, pairs, reject_log = _collect_valid_pairs(config)
    target_rng = np.random.default_rng(config.seed + 2000)
    corrections = np.array([p["H0_true"] - p["approx"].H0_approx for p in pairs], dtype=np.float32)
    n = config.n_systems

    mode2_true = np.stack([p["dm_params_true"] for p in pairs]).astype(np.float32)
    mode2_approx = np.stack([p["approx"].dm_params_approx for p in pairs]).astype(np.float32)
    mode2_corr = np.zeros((n, config.mode2_dm_dim), dtype=np.float32)
    source_corr = np.stack([p["S_true"] - p["S_approx"] for p in pairs]).astype(np.float32)

    psf = np.zeros((n, 11, 11), dtype=np.float32)
    yy, xx = np.mgrid[-5:6, -5:6]
    psf_base = np.exp(-(xx**2 + yy**2) / (2.0 * 1.4**2)).astype(np.float32)
    psf_base /= psf_base.sum()
    psf[:] = psf_base

    f_joint = np.stack([p["F_joint"] for p in pairs]).astype(np.float32)
    sigma_noise = np.stack([p["sigma_noise"] for p in pairs]).astype(np.float32)
    t_obs = np.stack([p["t_obs"] for p in pairs]).astype(np.float32)
    sigma_curves = np.stack([p["sigma_curve"] for p in pairs]).astype(np.float32)

    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["n_systems"] = n
        meta.attrs["random_seed"] = config.seed
        meta.attrs["full_truth_available"] = True
        meta.attrs["generator_version"] = (
            "phase4-v0.4"
            if validity_mode == "v0_4"
            else "phase4-v0.3.1"
            if validity_mode == "v0_3_1"
            else "phase4-v0.3"
            if validity_mode == "v0_3"
            else "phase4-v0.2"
        )
        meta.attrs["truth_model"] = config.truth_model
        meta.attrs["validity_filter"] = (
            "off_for_eval; root convergence required" if validity_mode == "off" else validity_mode
        )
        meta.attrs["truth_deflection"] = "alpha_SIE + alpha_NFW + kappa_ext * theta"
        meta.attrs["truth_image_solver"] = "scipy.optimize.root(theta - alpha_truth(theta) - beta), seeded by SIE images"
        meta.attrs["truth_image_search_mode"] = "SIE-anchored search; truth-only extra images are not searched in v0.1"
        meta.attrs["truth_image_dedupe_arcsec"] = TRUTH_IMAGE_DEDUPE_ARCSEC
        meta.attrs["truth_root_residual_arcsec"] = TRUTH_ROOT_RESIDUAL_ARCSEC
        meta.attrs["nfw_offset"] = "origin_aligned_v0"
        meta.attrs["kappa_ext_range"] = "[0, 0.1]"
        meta.attrs["correction_sign"] = "true_minus_approx"
        meta.attrs["incompatible_with_v2_scalers_checkpoints"] = True
        meta.attrs["image_size_note"] = "Phase 4 v0 uses 64 with pixel_scale=0.1; v1 should return to 128."
        meta.attrs["v0_2_mu_truth_abs_max"] = TRUTH_V0_2_MU_MAX
        meta.attrs["v0_2_dphi_sie_over_truth_min"] = TRUTH_V0_2_DPHI_RATIO_RANGE[0]
        meta.attrs["v0_2_dphi_sie_over_truth_max"] = TRUTH_V0_2_DPHI_RATIO_RANGE[1]
        meta.attrs["v0_2_truth_image_separation_min_arcsec"] = TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC
        meta.attrs["v0_2_dt_approx_max_days"] = TRUTH_V0_2_DT_APPROX_MAX_DAYS
        meta.attrs["v0_2_I_obs_sum_max"] = TRUTH_V0_2_I_OBS_SUM_MAX
        meta.attrs["v0_2_F_joint_absmax_max"] = TRUTH_V0_2_F_JOINT_ABSMAX_MAX
        meta.attrs["v0_2_mode1_H0_correction_absmax"] = TRUTH_V0_2_MODE1_CORRECTION_ABSMAX
        meta.attrs["v0_3_h0_neutral_filter"] = validity_mode in {"v0_3", "v0_3_1"}
        meta.attrs["v0_3_h0_stratified_min"] = TRUTH_V0_3_H0_STRATIFIED_RANGE[0]
        meta.attrs["v0_3_h0_stratified_max"] = TRUTH_V0_3_H0_STRATIFIED_RANGE[1]
        meta.attrs["v0_3_h0_stratified_bins"] = TRUTH_V0_3_H0_STRATIFIED_BINS
        meta.attrs["v0_3_1_restores_input_tail_gates"] = validity_mode == "v0_3_1"
        meta.attrs["v0_3_1_excludes_label_dependent_gates"] = validity_mode == "v0_3_1"

        params = f.create_group("params")
        for key in ("H0", "z_lens", "z_source", "sigma_v", "M200", "concentration", "q"):
            params.create_dataset(key, data=np.asarray([b[key] for b in bases], dtype=np.float32))
        params.create_dataset("theta_E", data=np.asarray([p["theta_E"] for p in pairs], dtype=np.float32))
        params.create_dataset("phi", data=np.asarray([b["position_angle"] for b in bases], dtype=np.float32))
        params.create_dataset("lens_truth_model", data=np.array([config.truth_model.encode()] * n))
        params.create_dataset("lens_model", data=np.array([b"SIE_standard_approx_v0"] * n))

        lc_g = f.create_group("light_curves")
        lc_g.create_dataset("F_joint", data=f_joint)
        lc_g.create_dataset("sigma_noise", data=sigma_noise)
        lc_g.create_dataset("t_obs", data=t_obs)
        lc_g.create_dataset("n_epochs", data=np.full(n, config.n_epochs, dtype=np.int32))

        obs_g = f.create_group("observed_features")
        obs_g.create_dataset("dt_lc", data=np.asarray([p["dt_lc"] for p in pairs], dtype=np.float32))
        obs_g.create_dataset("dt_lc_sigma", data=np.asarray([p["dt_lc_sigma"] for p in pairs], dtype=np.float32))
        obs_g.create_dataset(
            "dt_lc_sigma_relative_error",
            data=np.asarray([p["dt_lc_sigma_relative_error"] for p in pairs], dtype=np.float32),
        )
        for key in ("n_epochs_quality", "baseline_days", "median_cadence_days", "median_photometric_error"):
            obs_g.create_dataset(
                key,
                data=np.asarray([p["light_curve_quality"][key] for p in pairs], dtype=np.float32),
            )
        obs_g.attrs["dt_lc_source"] = "simulated_Bag22_proxy_primary_pair_positive_delay"
        obs_policy = _observed_feature_policy_from_config()
        sampler = obs_policy["dt_lc_sigma_sampler"]
        rel_cfg = sampler["relative_error"]
        obs_g.attrs["dt_lc_sigma_model"] = str(sampler["mode"])
        obs_g.attrs["dt_lc_sigma_relative_distribution"] = str(rel_cfg["distribution"])
        obs_g.attrs["dt_lc_sigma_relative_min"] = float(rel_cfg["min"])
        obs_g.attrs["dt_lc_sigma_relative_max"] = float(rel_cfg["max"])
        obs_g.attrs["dt_lc_sigma_absolute_floor_days"] = float(sampler["absolute_floor_days"])
        obs_g.attrs["dt_lc_sigma_absolute_ceiling_days"] = float(sampler["absolute_ceiling_days"])

        lcq_g = f.create_group("light_curve_quality")
        for key in ("n_epochs_quality", "baseline_days", "median_cadence_days", "median_photometric_error"):
            lcq_g.create_dataset(key, data=obs_g[key][:])

        images = f.create_group("images")
        images.create_dataset("I_obs", data=np.stack([p["I_obs"] for p in pairs]).astype(np.float32))
        images.create_dataset("S_true", data=np.stack([p["S_true"] for p in pairs]).astype(np.float32))
        images.create_dataset("psf", data=psf)
        images.create_dataset("pixel_scale", data=np.full(n, config.pixel_scale, dtype=np.float32))

        truth = f.create_group("true_values")
        truth.create_dataset("dt_true", data=np.asarray([p["dt_true"] for p in pairs], dtype=np.float32))
        truth.create_dataset("mu_true", data=np.asarray([p["mu_true"] for p in pairs], dtype=np.float32))
        truth.create_dataset("theta_E", data=np.asarray([p["theta_E"] for p in pairs], dtype=np.float32))
        truth.create_dataset("H0_true", data=np.asarray([p["H0_true"] for p in pairs], dtype=np.float32))
        truth.create_dataset("dm_params_true", data=mode2_true)
        truth.create_dataset("dm_dim", data=np.full(n, min(config.mode2_dm_dim, 4), dtype=np.int32))
        truth.create_dataset("D_delta_t", data=np.asarray([p["D_delta_t"] for p in pairs], dtype=np.float32))

        rays = f.create_group("ray_paths")
        rays.create_dataset("theta_1", data=np.stack([p["theta_truth_1"] for p in pairs]).astype(np.float32))
        rays.create_dataset("theta_2", data=np.stack([p["theta_truth_2"] for p in pairs]).astype(np.float32))
        rays.create_dataset("theta_1_approx", data=np.stack([p["theta_approx_1"] for p in pairs]).astype(np.float32))
        rays.create_dataset("theta_2_approx", data=np.stack([p["theta_approx_2"] for p in pairs]).astype(np.float32))
        rays.create_dataset("fermat_potential", data=np.asarray([p["truth_fermat_potential"] for p in pairs], dtype=np.float32))
        rays.create_dataset("fermat_potential_approx", data=np.asarray([p["approx"].fermat_potential for p in pairs], dtype=np.float32))
        rays.create_dataset("image_shift_arcsec", data=np.asarray([p["image_shift_arcsec"] for p in pairs], dtype=np.float32))
        rays.create_dataset("dphi_sie_over_truth", data=np.asarray([p["dphi_ratio"] for p in pairs], dtype=np.float32))
        rays.create_dataset("root_residual_norm_max", data=np.asarray([p["truth_solution"]["residual_norm_max"] for p in pairs], dtype=np.float32))

        approx_g = f.create_group("approx_outputs")
        approx_g.create_dataset("dt_approx", data=np.asarray([p["approx"].dt_approx for p in pairs], dtype=np.float32))
        approx_g.create_dataset("H0_approx", data=np.asarray([p["approx"].H0_approx for p in pairs], dtype=np.float32))
        approx_g.create_dataset("dm_params_approx", data=mode2_approx)
        approx_g.create_dataset("S_approx", data=np.stack([p["S_approx"] for p in pairs]).astype(np.float32))

        corr_g = f.create_group("correction_targets")
        corr_g.create_dataset("mode1_H0_correction", data=corrections)
        corr_g.create_dataset("mode2_dm_correction", data=mode2_corr)
        corr_g.create_dataset("mode3_source_correction", data=source_corr)

        legacy = f.create_group("simplification_errors")
        legacy.create_dataset("mode1_H0_error", data=corrections)
        legacy.create_dataset("mode2_dm_error", data=mode2_corr)
        legacy.create_dataset("mode3_source_residual", data=source_corr)
        legacy.create_dataset("approx_level_used", data=np.ones(n, dtype=np.int32))

        pert = f.create_group("perturbations")
        pert.create_dataset("kappa_ext", data=np.asarray([b["kappa_ext"] for b in bases], dtype=np.float32))
        pert.create_dataset("nfw_enabled", data=np.full(n, config.include_nfw, dtype=np.bool_))
        pert.create_dataset("kappa_ext_enabled", data=np.full(n, config.include_kappa_ext, dtype=np.bool_))

        f.create_dataset("sigma_curve", data=sigma_curves)
        f.create_dataset("target_mode", data=target_rng.integers(1, 4, n, dtype=np.int32))

    def _toggle_corrections(nfw_on: bool, kappa_on: bool) -> np.ndarray:
        vals: list[float] = []
        for base in bases:
            pair = _compute_pair(base, config, nfw_on, kappa_on, validate=False)
            if pair.get("valid"):
                vals.append(float(pair["H0_true"] - pair["approx"].H0_approx))
        return np.asarray(vals, dtype=float)

    combos = {
        "off_off_var": _toggle_corrections(False, False),
        "on_off_var": _toggle_corrections(True, False),
        "off_on_var": _toggle_corrections(False, True),
        "on_on_var": corrections.astype(float),
    }
    variance = {key: float(np.var(val)) for key, val in combos.items()}
    variance["cross_term"] = variance["on_on_var"] - variance["on_off_var"] - variance["off_on_var"]
    off_off = combos["off_off_var"]
    baseline = _baseline_v2_6_std()
    baseline_std = baseline.get("v2_6_baseline_std")
    summary: dict[str, Any] = {
        "catalog_path": str(output_path),
        "mode1_H0_correction": _distribution(corrections),
        "variance_decomposition": variance,
        "off_off_sanity": {
            "abs_mean": float(abs(np.mean(off_off))),
            "max_abs": float(np.max(np.abs(off_off))),
        },
        "v2_6_baseline": baseline,
        "phase4_v0_std": float(np.std(corrections)),
        "phase4_v0_1_std": float(np.std(corrections)),
        "phase4_v0_2_std": float(np.std(corrections)),
        "phase4_v0_3_std": float(np.std(corrections)),
        "correction_sign": "true_minus_approx",
        "validity_filter": validity_mode,
        "image_size": config.image_size,
        "pixel_scale": config.pixel_scale,
        "v0_2_thresholds": {
            "mu_truth_abs_max": TRUTH_V0_2_MU_MAX,
            "dphi_sie_over_truth_range": TRUTH_V0_2_DPHI_RATIO_RANGE,
            "truth_image_separation_min_arcsec": TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC,
            "dt_approx_max_days": TRUTH_V0_2_DT_APPROX_MAX_DAYS,
            "I_obs_sum_max": TRUTH_V0_2_I_OBS_SUM_MAX,
            "F_joint_absmax_max": TRUTH_V0_2_F_JOINT_ABSMAX_MAX,
            "mode1_H0_correction_absmax": TRUTH_V0_2_MODE1_CORRECTION_ABSMAX,
        },
        "v0_3_policy": reject_log["v0_3_policy"],
        "v0_3_1_policy": reject_log["v0_3_1_policy"],
        "image_shift_arcsec": _distribution(np.asarray([p["image_shift_arcsec"] for p in pairs], dtype=float)),
        "root_residual_norm": {
            "median": reject_log["root_residual_norm_median"],
            "max": reject_log["root_residual_norm_max"],
        },
        "resample": {
            "attempts_mean": reject_log["attempts_mean"],
            "attempts_max": reject_log["attempts_max"],
            "reject_counts": reject_log["reject_counts"],
            "dedupe_count_total": reject_log["dedupe_count_total"],
            "h0_stratified_counts": reject_log.get("h0_stratified_counts"),
            "h0_stratified_quotas": reject_log.get("h0_stratified_quotas"),
            "h0_stratified_edges": reject_log.get("h0_stratified_edges"),
        },
    }
    if isinstance(baseline_std, float) and baseline_std > 0:
        summary["v2_6_baseline_std"] = baseline_std
        summary["ratio"] = float(np.std(corrections) / baseline_std)
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    with reject_path.open("w", encoding="utf-8") as fp:
        json.dump(reject_log, fp, indent=2, sort_keys=True)
    write_phase4_v0_diagnosis(diagnosis_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-systems", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "mock" / "phase4_v0.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--reject-log-path", type=Path, default=None)
    parser.add_argument("--diagnosis-log-path", type=Path, default=None)
    parser.add_argument("--resample-budget", type=int, default=None)
    parser.add_argument("--validity-filter", choices=("v0_4", "v0_3_1", "v0_3", "v0_2", "v0_1", "off"), default="v0_2")
    parser.add_argument("--eval-role", default=None)
    args = parser.parse_args()
    summary = build_phase4_catalog(
        args.output,
        CatalogConfig(
            n_systems=args.n_systems,
            seed=args.seed,
            log_path=args.log_path,
            reject_log_path=args.reject_log_path,
            diagnosis_log_path=args.diagnosis_log_path,
            resample_budget=args.resample_budget,
            validity_filter=args.validity_filter,
        ),
    )
    if args.eval_role is not None:
        with h5py.File(args.output, "a") as f:
            f["metadata"].attrs["eval_role"] = args.eval_role
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
