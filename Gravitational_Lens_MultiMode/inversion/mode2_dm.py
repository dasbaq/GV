"""Mode 2 — SIE standard-approximation dark-matter parameter inversion.

Inputs are public observation-side image positions ``theta_i`` [arcsec],
positive primary-pair delay ``dt_obs`` [days], fixed H0 [km/s/Mpc], and
redshifts. The solver fits the project-wide fixed SIE standard approximation
and returns parameters in the Phase 4 HDF5 order
``[theta_E, q, position_angle, sigma_v]``.

SIE 표준 근사 가정: single lens plane, κ_ext=0, smooth SIE mass profile,
isotropic velocity dispersion. No approximation/profile switch is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from core.physics.config import constants, default_cosmology
from core.physics.lens_models import SIELens
from inversion.mode1_h0 import _d_delta_t


LENS_PARAM_DIM = {"SIE": 4}
LENS_PARAM_BOUNDS = {"SIE": [(0.0, np.inf), (0.2, 1.0), (-np.pi, np.pi), (50.0, 500.0)]}
PARAM_NAMES = ["theta_E", "q", "position_angle", "sigma_v"]

_SIGMA_BOUNDS = (50.0, 500.0)
_Q_BOUNDS = (0.2, 1.0)
_PA_BOUNDS = (-np.pi, np.pi)
_BETA_BOUNDS = (-10.0, 10.0)


@dataclass(frozen=True)
class Mode2Weights:
    """Residual weights for SIE Mode 2 inversion.

    Units: position residuals are normalized arcsec offsets, delay residuals
    are normalized [days], and magnification is dimensionless. SIE 표준 근사
    가정: μ is not part of the default public likelihood and ``mu`` defaults
    to zero to avoid truth-side leakage.
    """

    position: float = 1.0
    delay: float = 1.0
    mu: float = 0.0
    prior: float = 1.0e-4


def _as_weights(weights: Mapping[str, float] | Mode2Weights | None) -> Mode2Weights:
    if weights is None:
        return Mode2Weights()
    if isinstance(weights, Mode2Weights):
        return weights
    return Mode2Weights(
        position=float(weights.get("position", 1.0)),
        delay=float(weights.get("delay", 1.0)),
        mu=float(weights.get("mu", 0.0)),
        prior=float(weights.get("prior", 1.0e-4)),
    )


def _validate_inputs(
    dt_obs: np.ndarray,
    theta_obs: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Validate public Mode 2 inputs.

    Units: ``dt_obs`` [days], ``theta_obs`` [arcsec], H0 [km/s/Mpc], redshifts
    dimensionless. SIE 표준 근사 가정: only public observation-side values are
    accepted; truth-side magnification or DM labels are not required.
    """

    dt = np.asarray(dt_obs, dtype=float).reshape(-1)
    theta = np.asarray(theta_obs, dtype=float)
    h0 = float(H0)
    z_l = float(z_lens)
    z_s = float(z_source)
    if dt.size < 1:
        raise ValueError("dt_obs must contain at least one positive delay [days]")
    if theta.ndim != 2 or theta.shape[1] != 2 or theta.shape[0] < 2:
        raise ValueError("theta_obs must have shape (n_images>=2, 2) in arcsec")
    if not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise ValueError("dt_obs must be finite and positive [days]")
    if not np.isfinite(theta).all():
        raise ValueError("theta_obs must be finite [arcsec]")
    if not np.isfinite([h0, z_l, z_s]).all() or h0 <= 0.0:
        raise ValueError("H0 and redshifts must be finite and H0 must be positive")
    if not z_s > z_l + 0.05:
        raise ValueError(f"z_source ({z_s}) must be greater than z_lens ({z_l}) + 0.05")
    return dt, theta, h0, z_l, z_s


def _lens_from_fit_params(
    fit_params: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
) -> tuple[SIELens, np.ndarray]:
    """Build an SIE lens and source position from optimizer parameters.

    Optimizer parameters are ``[sigma_v, q, position_angle, beta_x, beta_y]``.
    Units: ``sigma_v`` [km/s], angles/source plane [arcsec], redshifts
    dimensionless. SIE 표준 근사 가정: single plane, κ_ext=0, smooth SIE.
    """

    sigma_v, q, position_angle, beta_x, beta_y = np.asarray(fit_params, dtype=float)
    cosmo = dict(default_cosmology())
    cosmo["H0"] = float(H0)
    lens = SIELens(
        sigma_v=float(sigma_v),
        q=float(q),
        position_angle=float(position_angle),
        z_lens=float(z_lens),
        z_source=float(z_source),
        cosmology=cosmo,
    )
    beta = np.array([beta_x, beta_y], dtype=float)
    return lens, beta


def _pair_indices(n_images: int) -> tuple[np.ndarray, np.ndarray]:
    if int(n_images) < 2:
        raise ValueError("at least two images are required")
    return np.array([0], dtype=int), np.array([1], dtype=int)


def _time_delay_from_dphi(
    dphi_rad2: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
    approx_level: int = 0,
) -> np.ndarray:
    """Convert SIE Fermat-potential differences to delays [days].

    Units: ``dphi_rad2`` [rad²], H0 [km/s/Mpc], redshifts dimensionless,
    return [days]. SIE 표준 근사 가정: distance scaling matches Mode 1's
    Δφ [rad²] convention.
    """

    c = constants()["c_m_s"]
    day_s = constants()["day_s"]
    mpc_m = constants()["Mpc_m"]
    d_reduced = _d_delta_t(float(H0), float(z_lens), float(z_source), approx_level)
    return ((1.0 + float(z_lens)) * d_reduced * mpc_m / c) * np.asarray(dphi_rad2, dtype=float) / day_s


def _magnification_proxy(image_positions: np.ndarray, theta_e_arcsec: float) -> np.ndarray:
    """Return a finite SIE magnification proxy for convergence diagnostics.

    Units: image positions and ``theta_e_arcsec`` are [arcsec]. SIE 표준 근사
    가정: this proxy is a diagnostic only and is not used by the default
    μ-free likelihood.
    """

    radius = np.linalg.norm(np.asarray(image_positions, dtype=float), axis=-1)
    denom = np.maximum(np.abs(1.0 - float(theta_e_arcsec) / np.maximum(radius, 1.0e-3)), 0.05)
    return 1.0 / denom


def _predict_observables_sie(
    fit_params: np.ndarray,
    theta_obs: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
    approx_level: int = 0,
) -> dict[str, Any]:
    """Predict SIE lens-equation and delay observables from fit parameters.

    Units: optimizer parameters are ``[sigma_v, q, position_angle, beta_x,
    beta_y]`` with sigma [km/s] and angles [arcsec]; ``theta_obs`` [arcsec];
    returned Δφ [rad²] and delays [days]. SIE 표준 근사 가정: fixed project
    standard approximation with no profile/approximation selector.
    """

    lens, beta = _lens_from_fit_params(fit_params, H0, z_lens, z_source)
    theta = np.asarray(theta_obs, dtype=float)
    beta_images = theta - lens.deflection(theta)
    beta_residual = beta_images - beta[None, :]
    phi = np.asarray(lens.fermat_potential(theta, beta), dtype=float)
    first, second = _pair_indices(theta.shape[0])
    dphi = np.abs(phi[first] - phi[second])
    dt_model = _time_delay_from_dphi(dphi, H0, z_lens, z_source, approx_level)
    theta_e = lens.einstein_radius()
    mags = _magnification_proxy(theta, theta_e)
    if theta.shape[0] >= 2:
        order = np.argsort(mags)[::-1]
        mu_ratio = float(np.clip(mags[order[1]] / max(float(mags[order[0]]), 1.0e-12), 1.0e-6, 0.999999))
    else:
        mu_ratio = float("nan")
    return {
        "lens": lens,
        "beta": beta,
        "beta_images": beta_images,
        "position_residual": beta_residual,
        "fermat_potential": phi,
        "dphi_rad2": dphi,
        "dt_model_days": dt_model,
        "theta_E": float(theta_e),
        "mu_proxy": mags,
        "mu_ratio": mu_ratio,
    }


def _initial_theta_e(image_positions: np.ndarray) -> float:
    radii = np.linalg.norm(np.asarray(image_positions, dtype=float), axis=1)
    pairwise = np.linalg.norm(
        image_positions[:, None, :] - image_positions[None, :, :],
        axis=-1,
    )
    return max(float(np.median(radii)), 0.5 * float(np.max(pairwise)), 1.0e-3)


def _sigma_from_theta_e(
    theta_e_arcsec: float,
    H0: float,
    z_lens: float,
    z_source: float,
) -> float:
    """Invert SIE/SIS Einstein radius to a velocity dispersion [km/s].

    Units: ``theta_e_arcsec`` [arcsec], H0 [km/s/Mpc], redshifts dimensionless.
    SIE 표준 근사 가정: Einstein-radius conversion uses the same isotropic
    SIE/SIS thin-lens relation as ``SIELens.einstein_radius``.
    """

    from scipy.optimize import brentq

    target = max(float(theta_e_arcsec), 1.0e-3)

    def residual(sigma_v: float) -> float:
        lens, _ = _lens_from_fit_params(
            np.array([sigma_v, 1.0, 0.0, 0.0, 0.0], dtype=float),
            H0,
            z_lens,
            z_source,
        )
        return lens.einstein_radius() - target

    try:
        return float(brentq(residual, _SIGMA_BOUNDS[0], _SIGMA_BOUNDS[1]))
    except ValueError:
        return 220.0


def _principal_angle(image_positions: np.ndarray) -> float:
    centered = np.asarray(image_positions, dtype=float) - np.mean(image_positions, axis=0, keepdims=True)
    if centered.shape[0] == 2:
        delta = centered[1] - centered[0]
        return float(np.arctan2(delta[1], delta[0]))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return float(np.arctan2(vh[0, 1], vh[0, 0]))


def _residual_vector(
    fit_params: np.ndarray,
    theta_obs: np.ndarray,
    dt_obs: np.ndarray,
    dt_sigma: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
    weights: Mode2Weights,
    mu_obs: np.ndarray | None,
) -> np.ndarray:
    pred = _predict_observables_sie(fit_params, theta_obs, H0, z_lens, z_source, approx_level=0)
    theta_scale = max(_initial_theta_e(theta_obs), 0.1)
    pos = pred["position_residual"].reshape(-1) / theta_scale

    n_pairs = pred["dt_model_days"].shape[0]
    dt_target = np.resize(np.asarray(dt_obs, dtype=float), n_pairs)
    dt_unc = np.maximum(np.resize(np.asarray(dt_sigma, dtype=float), n_pairs), 1.0e-6)
    dt = (pred["dt_model_days"] - dt_target) / dt_unc

    residuals: list[np.ndarray] = []
    if weights.position > 0.0:
        residuals.append(np.sqrt(weights.position) * pos)
    if weights.delay > 0.0:
        residuals.append(np.sqrt(weights.delay) * dt)
    if weights.mu > 0.0 and mu_obs is not None:
        mu = np.asarray(mu_obs, dtype=float).reshape(-1)
        if mu.size:
            mu_target = np.resize(mu, pred["mu_proxy"].shape[0])
            mu_resid = (pred["mu_proxy"] - mu_target) / np.maximum(np.abs(mu_target), 1.0e-6)
            residuals.append(np.sqrt(weights.mu) * mu_resid)
    if weights.prior > 0.0:
        sigma_v, q, position_angle, beta_x, beta_y = np.asarray(fit_params, dtype=float)
        prior = np.array(
            [
                (sigma_v - 260.0) / 220.0,
                (q - 0.75) / 0.35,
                np.sin(position_angle) / 1.0,
                beta_x / 5.0,
                beta_y / 5.0,
            ],
            dtype=float,
        )
        residuals.append(np.sqrt(weights.prior) * prior)
    return np.concatenate(residuals) if residuals else np.zeros(1, dtype=float)


def _fit_sie(
    dt_obs: np.ndarray,
    theta_obs: np.ndarray,
    dt_sigma: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
    weights: Mode2Weights,
    mu_obs: np.ndarray | None,
    rng_seed: int,
) -> least_squares:
    """Fit internal SIE parameters to image positions and delay.

    Units: delays [days], positions/source plane [arcsec], sigma [km/s].
    SIE 표준 근사 가정: fixed single-plane SIE; q and position angle affect
    both lens-equation and Fermat-potential residuals.
    """

    theta_e0 = _initial_theta_e(theta_obs)
    sigma0 = _sigma_from_theta_e(theta_e0, H0, z_lens, z_source)
    pa0 = _principal_angle(theta_obs)
    rng = np.random.default_rng(rng_seed)

    starts: list[np.ndarray] = []
    for q0 in (0.55, 0.7, 0.85, 0.98):
        for pa in (pa0, pa0 + 0.5 * np.pi, 0.0, 0.25 * np.pi):
            lens, _ = _lens_from_fit_params(np.array([sigma0, q0, pa, 0.0, 0.0]), H0, z_lens, z_source)
            beta0 = np.mean(theta_obs - lens.deflection(theta_obs), axis=0)
            starts.append(np.array([sigma0, q0, pa, beta0[0], beta0[1]], dtype=float))
    for _ in range(12):
        q0 = rng.uniform(_Q_BOUNDS[0], _Q_BOUNDS[1])
        pa = rng.uniform(_PA_BOUNDS[0], _PA_BOUNDS[1])
        sig = np.clip(sigma0 * rng.uniform(0.7, 1.3), *_SIGMA_BOUNDS)
        lens, _ = _lens_from_fit_params(np.array([sig, q0, pa, 0.0, 0.0]), H0, z_lens, z_source)
        beta0 = np.mean(theta_obs - lens.deflection(theta_obs), axis=0)
        starts.append(np.array([sig, q0, pa, beta0[0], beta0[1]], dtype=float))

    lower = np.array([_SIGMA_BOUNDS[0], _Q_BOUNDS[0], _PA_BOUNDS[0], _BETA_BOUNDS[0], _BETA_BOUNDS[0]], dtype=float)
    upper = np.array([_SIGMA_BOUNDS[1], _Q_BOUNDS[1], _PA_BOUNDS[1], _BETA_BOUNDS[1], _BETA_BOUNDS[1]], dtype=float)

    best = None
    for start in starts:
        start = np.clip(start, lower, upper)
        result = least_squares(
            _residual_vector,
            x0=start,
            bounds=(lower, upper),
            args=(theta_obs, dt_obs, dt_sigma, H0, z_lens, z_source, weights, mu_obs),
            method="trf",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=3000,
        )
        if best is None or result.cost < best.cost:
            best = result
    if best is None or not best.success:
        raise RuntimeError("Mode 2 SIE fit did not converge")
    return best


def _public_params_from_fit(
    fit_params: np.ndarray,
    H0: float,
    z_lens: float,
    z_source: float,
) -> np.ndarray:
    lens, _ = _lens_from_fit_params(fit_params, H0, z_lens, z_source)
    sigma_v, q, position_angle, _, _ = np.asarray(fit_params, dtype=float)
    return np.array([lens.einstein_radius(), q, position_angle, sigma_v], dtype=float)


def invert_dm(
    dt_obs: np.ndarray,
    theta_obs: np.ndarray,
    mu_obs: np.ndarray | None = None,
    H0: float = 70.0,
    z_lens: float = 0.3,
    z_source: float = 1.5,
    lens_model: str = "SIE",
    approx_level: int = 0,
    n_bootstrap: int = 200,
    rng_seed: int = 42,
    dt_sigma: np.ndarray | None = None,
    residual_weights: Mapping[str, float] | Mode2Weights | None = None,
) -> dict[str, Any]:
    """Invert SIE dark-matter parameters from positions and time delay.

    Parameters
    ----------
    dt_obs:
        Positive primary-pair observed delays [days].
    theta_obs:
        Image positions [arcsec], shape ``(n_images, 2)``.
    mu_obs:
        Optional observation-side magnification values. Ignored by default
        because ``residual_weights.mu`` is zero.
    H0:
        Fixed Hubble constant [km/s/Mpc].
    z_lens, z_source:
        Redshifts, dimensionless.
    lens_model:
        Must be ``"SIE"``. SIE is the project-wide fixed standard
        approximation.
    approx_level:
        Must be ``0``. Mode 2 reuses the exact Mode 1 distance path for
        Δφ [rad²] to delay [days] conversion.
    n_bootstrap:
        Number of uncertainty bootstrap samples.
    rng_seed:
        Reproducibility seed.
    dt_sigma:
        Delay uncertainty [days]. Defaults to 5% relative uncertainty with a
        small floor.
    residual_weights:
        Optional weights. Defaults are μ-free.

    Returns
    -------
    dict
        ``dm_params`` and ``dm_uncertainty`` in Phase 4 order
        ``[theta_E, q, position_angle, sigma_v]``.

    SIE 표준 근사 가정
    ----------------
    The fitted model is a single-plane, κ_ext=0, smooth SIE lens with isotropic
    velocity dispersion. No ``approximation_*`` or profile selector is accepted.
    """

    if lens_model != "SIE":
        raise ValueError("Mode 2 supports only the fixed SIE standard approximation")
    if int(approx_level) != 0:
        raise ValueError("Mode 2 supports only approx_level=0 for Mode 1-consistent Δφ units")

    dt, theta, h0, z_l, z_s = _validate_inputs(dt_obs, theta_obs, H0, z_lens, z_source)
    if dt_sigma is None:
        dt_unc = np.maximum(np.abs(dt) * 0.05, 1.0e-3)
    else:
        dt_unc = np.asarray(dt_sigma, dtype=float).reshape(-1)
        if dt_unc.size < 1 or not np.isfinite(dt_unc).all() or np.any(dt_unc <= 0.0):
            raise ValueError("dt_sigma must be finite and positive [days]")
    weights = _as_weights(residual_weights)
    mu = None if mu_obs is None else np.asarray(mu_obs, dtype=float)

    best = _fit_sie(dt, theta, dt_unc, h0, z_l, z_s, weights, mu, rng_seed)
    pred = _predict_observables_sie(best.x, theta, h0, z_l, z_s, approx_level=0)
    if not abs(float(pred["mu_ratio"])) < 1.0:
        raise ValueError("|mu| < 1 convergence condition failed for fitted SIE")

    params_best = _public_params_from_fit(best.x, h0, z_l, z_s)
    pos_norm = np.linalg.norm(pred["position_residual"], axis=1)

    rng = np.random.default_rng(rng_seed + 1000)
    n_boot = max(int(n_bootstrap), 0)
    samples = np.empty((n_boot, len(PARAM_NAMES)), dtype=float)
    for i in range(n_boot):
        try:
            if theta.shape[0] > 2:
                idx = rng.integers(0, theta.shape[0], size=theta.shape[0])
                theta_i = theta[idx]
            else:
                theta_i = theta
            dt_i = dt * rng.normal(1.0, np.maximum(np.resize(dt_unc, dt.size) / dt, 1.0e-6))
            fit_i = _fit_sie(dt_i, theta_i, dt_unc, h0, z_l, z_s, weights, mu, rng_seed + 2000 + i)
            samples[i] = _public_params_from_fit(fit_i.x, h0, z_l, z_s)
        except Exception:
            samples[i] = params_best
    unc = np.zeros_like(params_best) if n_boot == 0 else np.std(samples, axis=0)

    return {
        "dm_params": params_best,
        "dm_uncertainty": unc,
        "lens_model": "SIE",
        "approx_level": 0,
        "param_names": list(PARAM_NAMES),
        "theta_E": float(params_best[0]),
        "q": float(params_best[1]),
        "position_angle": float(params_best[2]),
        "sigma_v": float(params_best[3]),
        "source_pos_xy": np.asarray(pred["beta"], dtype=np.float32),
        "dphi_rad2": float(np.asarray(pred["dphi_rad2"]).reshape(-1)[0]),
        "dt_model_days": float(np.asarray(pred["dt_model_days"]).reshape(-1)[0]),
        "dt_obs_days": float(dt[0]),
        "position_residual_rms_arcsec": float(np.sqrt(np.mean(pos_norm**2))),
        "max_residual_arcsec": float(np.max(pos_norm)),
        "mu_fit": float(pred["mu_ratio"]),
        "residual_weights": {
            "position": float(weights.position),
            "delay": float(weights.delay),
            "mu": float(weights.mu),
            "prior": float(weights.prior),
        },
        "input_audit": {
            "uses_theta_obs": True,
            "uses_dt_obs": True,
            "uses_mu_obs": bool(weights.mu > 0.0 and mu_obs is not None),
            "uses_truth_mu_true": False,
        },
        "optimizer": {
            "cost": float(best.cost),
            "success": bool(best.success),
            "nfev": int(best.nfev),
        },
    }


# Backward-compatible helper name used by legacy tests; Mode 2 no longer
# exposes SIS as a fit model.
def _sis_einstein_radius(sigma_v: float, z_lens: float, z_source: float, H0: float) -> float:
    """Return the q=1 SIE/SIS Einstein radius [arcsec].

    Units: ``sigma_v`` [km/s], H0 [km/s/Mpc], redshifts dimensionless. SIE
    표준 근사 가정: q=1 special case of the project-wide SIE model.
    """

    lens, _ = _lens_from_fit_params(
        np.array([float(sigma_v), 1.0, 0.0, 0.0, 0.0], dtype=float),
        float(H0),
        float(z_lens),
        float(z_source),
    )
    return float(lens.einstein_radius())
