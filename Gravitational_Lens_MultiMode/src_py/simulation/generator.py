from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.physics.config import constants, default_cosmology
from core.physics.distances import angular_diameter_distance, angular_diameter_distance_between
from core.physics.lens_models import SIELens
from core.physics.ray_tracing import find_images_thin_lens, thin_lens_time_delay
from src_py.simulation.image_renderer import render_lensed_image
from src_py.simulation.quasar_lc import QuasarLightCurve


def _normalize_unit(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _standardize(x: np.ndarray) -> np.ndarray:
    return ((x - np.mean(x)) / (np.std(x) + 1e-6)).astype(np.float32)


def _sigma_v_from_theta_e(theta_e_arcsec: float, z_lens: float, z_source: float) -> float:
    c_m_s = constants()["c_m_s"]
    theta_rad = theta_e_arcsec * constants()["arcsec_to_rad"]
    d_s = angular_diameter_distance(z_source) * constants()["Mpc_m"]
    d_ls = angular_diameter_distance_between(z_lens, z_source) * constants()["Mpc_m"]
    sigma_m_s = c_m_s * np.sqrt(theta_rad * d_s / (4.0 * np.pi * d_ls))
    return float(sigma_m_s / 1000.0)


def _unique_images(theta_list: np.ndarray, min_sep: float = 0.05) -> np.ndarray:
    unique: list[np.ndarray] = []
    for theta in np.asarray(theta_list, dtype=float):
        if all(np.linalg.norm(theta - prev) >= min_sep for prev in unique):
            unique.append(theta)
    return np.array(unique, dtype=np.float32)


def _mock_magnification(theta: np.ndarray, theta_e: float) -> np.ndarray:
    radius = np.linalg.norm(theta, axis=-1)
    return 1.0 / np.maximum(np.abs(1.0 - theta_e / np.maximum(radius, 1e-3)), 0.05)


def _delayed_joint_curve(
    flux: np.ndarray,
    flux_err: np.ndarray,
    t_obs: np.ndarray,
    delay_days: float,
    mu_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    delayed = np.interp(t_obs - delay_days, t_obs, flux, left=flux[0], right=flux[-1])
    joint = flux + mu_b * delayed
    joint = _standardize(joint)
    sigma = np.sqrt(flux_err**2 + (mu_b * flux_err) ** 2).astype(np.float32)
    return joint.astype(np.float32), sigma


def _sigma_curve(
    rng: np.random.Generator,
    n_points: int,
    dt_lc: float,
    dt_lc_sigma: float,
) -> np.ndarray:
    grid = np.linspace(1.0, 100.0, n_points, dtype=np.float32)
    curve = rng.normal(0.0, 0.15, n_points).astype(np.float32)
    width = max(float(dt_lc_sigma), 1.0)
    curve -= 2.5 * np.exp(-0.5 * ((grid - np.clip(dt_lc, 1.0, 100.0)) / width) ** 2).astype(np.float32)
    return _standardize(curve)


def simulate_dataset(
    n_systems: int,
    output_path: Path,
    survey: str = "ztf",
    n_epochs: int = 200,
    image_size: int = 64,
    sigma_curve_size: int = 50,
    mode2_dm_dim: int = 4,
    seed: int = 42,
    h0_truth: float = 70.0,
) -> None:
    """
    Generate SIE-based Phase 3 lens systems into the LensCorrectionDataset HDF5 schema.

    Units: angular quantities in arcsec, redshifts dimensionless, time delays in days,
    H0 in km/s/Mpc, light curves in normalized magnitude residual units. Standard
    approximation assumption: SIE lens, single plane, kappa_ext=0.
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "H0": np.zeros(n_systems, dtype=np.float32),
        "z_lens": np.zeros(n_systems, dtype=np.float32),
        "z_source": np.zeros(n_systems, dtype=np.float32),
        "sigma_v": np.zeros(n_systems, dtype=np.float32),
        "M200": np.zeros(n_systems, dtype=np.float32),
        "concentration": np.zeros(n_systems, dtype=np.float32),
        "q": np.zeros(n_systems, dtype=np.float32),
        "theta_E": np.zeros(n_systems, dtype=np.float32),
        "phi": np.zeros(n_systems, dtype=np.float32),
        "dt_true": np.zeros(n_systems, dtype=np.float32),
        "mu_true": np.zeros(n_systems, dtype=np.float32),
        "D_delta_t": np.zeros(n_systems, dtype=np.float32),
    }
    f_joint = np.zeros((n_systems, n_epochs), dtype=np.float32)
    sigma_noise = np.zeros((n_systems, n_epochs), dtype=np.float32)
    t_obs_all = np.zeros((n_systems, n_epochs), dtype=np.float32)
    sigma_curves = np.zeros((n_systems, sigma_curve_size), dtype=np.float32)
    i_obs = np.zeros((n_systems, image_size, image_size), dtype=np.float32)
    s_true = np.zeros_like(i_obs)
    s_approx = np.zeros_like(i_obs)
    kappa_ext_arr = np.zeros(n_systems, dtype=np.float32)
    delta_t_sie_arr = np.zeros(n_systems, dtype=np.float32)
    delta_t_obs_arr = np.zeros(n_systems, dtype=np.float32)
    dt_sigma_rel_arr = np.zeros(n_systems, dtype=np.float32)
    dt_approx_noise_factor_arr = np.zeros(n_systems, dtype=np.float32)
    dt_approx_noisy_arr = np.zeros(n_systems, dtype=np.float32)
    mean_i_sie_arr = np.zeros(n_systems, dtype=np.float32)
    mean_i_obs_preclip_arr = np.zeros(n_systems, dtype=np.float32)
    mean_i_obs_arr = np.zeros(n_systems, dtype=np.float32)
    clip_fraction_arr = np.zeros(n_systems, dtype=np.float32)
    mode1_h0_error_arr = np.zeros(n_systems, dtype=np.float32)
    mode3_source_residual = np.zeros_like(i_obs)
    dm_params = np.zeros((n_systems, mode2_dm_dim), dtype=np.float32)
    theta_1 = np.zeros((n_systems, 2), dtype=np.float32)
    theta_2 = np.zeros((n_systems, 2), dtype=np.float32)
    fermat = np.zeros(n_systems, dtype=np.float32)

    accepted = 0
    attempts = 0
    max_attempts = max(1000, n_systems * 100)

    while accepted < n_systems:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Unable to generate {n_systems} valid systems after {attempts} attempts")

        h0_system = float(rng.uniform(60.0, 80.0))
        cosmo = default_cosmology().copy()
        cosmo["H0"] = h0_system
        theta_e = float(rng.uniform(0.5, 2.0))
        q = float(rng.uniform(0.5, 1.0))
        phi = float(rng.uniform(0.0, np.pi))
        z_lens = float(rng.uniform(0.2, 0.8))
        z_source = float(rng.uniform(z_lens + 0.5, 2.5))
        beta = rng.uniform(-0.5, 0.5, 2).astype(np.float32)
        sigma_v = _sigma_v_from_theta_e(theta_e, z_lens, z_source)
        lens = SIELens(sigma_v=sigma_v, q=q, position_angle=phi, z_lens=z_lens, z_source=z_source, cosmology=cosmo)

        images = _unique_images(find_images_thin_lens(beta, lens, search_box_arcsec=max(3.0, 2.5 * theta_e), grid_size=96))
        if images.shape[0] < 2:
            continue

        mags = _mock_magnification(images, theta_e)
        order = np.argsort(mags)[::-1]
        img_a = images[order[0]]
        img_b = images[order[1]]
        dt_days = abs(float(thin_lens_time_delay(img_a, img_b, beta, lens, z_lens, z_source)))
        if not np.isfinite(dt_days):
            continue

        # Magnification is a placeholder because the current SIE API exposes
        # deflection but not a Jacobian; Phase 4/image_renderer will replace it.
        mu_b = float(np.clip(mags[order[1]] / max(mags[order[0]], 1e-6), 0.3, 1.0))
        kappa_ext = float(rng.uniform(0.0, 0.10))
        delta_t_obs = dt_days * (1.0 - kappa_ext)
        dt_sigma_rel = float(rng.uniform(0.02, 0.07))
        dt_approx_noise_factor = float(rng.normal(0.0, dt_sigma_rel))
        dt_approx_noisy = max(0.5 * dt_days, dt_days * (1.0 + dt_approx_noise_factor))
        dt_approx_noise_factor = dt_approx_noisy / dt_days - 1.0
        dt_lc_sigma = dt_approx_noisy * dt_sigma_rel
        lc_seed = int(rng.integers(0, 2**31 - 1))
        lc = QuasarLightCurve(seed=lc_seed).generate(n_epochs=n_epochs, total_days=1000.0, survey=survey)
        # The light curve should expose only the measured delay estimate used by
        # the inversion, not the noiseless truth-side MSD delay.
        delta_t_lc_embedded = dt_approx_noisy
        joint, noise = _delayed_joint_curve(lc["flux"], lc["flux_err"], lc["t_obs"], delta_t_lc_embedded, mu_b)
        image_sie = render_lensed_image(lens, beta, image_size=image_size, rng=rng)
        # MSD also rescales image-plane brightness; clipping preserves a pixel-level
        # kappa_ext signature paired with the shortened observed time delay.
        image_preclip = image_sie / (1.0 - kappa_ext)
        image = np.clip(image_preclip, 0.0, 1.0).astype(np.float32)
        # Mass-Sheet Degeneracy: an SIE-only fit to hidden external convergence
        # biases the inferred H0 by approximately (1 - kappa_ext).
        h0_approx_inferred = h0_system * (1.0 - kappa_ext) * (dt_days / dt_approx_noisy)
        h0_error = h0_approx_inferred - h0_system
        img_residual = (image - image_sie).astype(np.float32)

        i = accepted
        arrays["H0"][i] = h0_system
        arrays["z_lens"][i] = z_lens
        arrays["z_source"][i] = z_source
        arrays["sigma_v"][i] = sigma_v
        arrays["M200"][i] = np.float32(10.0 ** rng.uniform(12.0, 14.0))
        arrays["concentration"][i] = np.float32(rng.uniform(3.0, 15.0))
        arrays["q"][i] = q
        arrays["theta_E"][i] = theta_e
        arrays["phi"][i] = phi
        arrays["dt_true"][i] = dt_days
        arrays["mu_true"][i] = mu_b
        arrays["D_delta_t"][i] = 0.0
        f_joint[i] = joint
        sigma_noise[i] = noise
        t_obs_all[i] = lc["t_obs"]
        sigma_curves[i] = _sigma_curve(rng, sigma_curve_size, dt_approx_noisy, dt_lc_sigma)
        i_obs[i] = image
        s_true[i] = image
        s_approx[i] = np.clip(image - img_residual, 0.0, 1.0)
        kappa_ext_arr[i] = kappa_ext
        delta_t_sie_arr[i] = dt_days
        delta_t_obs_arr[i] = delta_t_obs
        dt_sigma_rel_arr[i] = dt_sigma_rel
        dt_approx_noise_factor_arr[i] = dt_approx_noise_factor
        dt_approx_noisy_arr[i] = dt_approx_noisy
        mean_i_sie_arr[i] = float(image_sie.mean())
        mean_i_obs_preclip_arr[i] = float(image_preclip.mean())
        mean_i_obs_arr[i] = float(image.mean())
        clip_fraction_arr[i] = float((image_preclip > 1.0).mean())
        mode1_h0_error_arr[i] = h0_error
        mode3_source_residual[i] = img_residual
        dm_params[i, : min(4, mode2_dm_dim)] = np.array([theta_e, q, phi, sigma_v], dtype=np.float32)[: min(4, mode2_dm_dim)]
        theta_1[i] = img_a
        theta_2[i] = img_b
        fermat[i] = dt_days

        accepted += 1
        if accepted % 10 == 0 or accepted == n_systems:
            print(f"generated {accepted}/{n_systems} systems")

    zeros_dm = np.zeros((n_systems, mode2_dm_dim), dtype=np.float32)
    # TODO(Phase 4): replace mode2 placeholder corrections with DM full_numerical - SIE labels.
    psf = np.zeros((n_systems, 11, 11), dtype=np.float32)
    yy, xx = np.mgrid[-5:6, -5:6]
    psf_base = np.exp(-(xx**2 + yy**2) / (2.0 * 1.4**2)).astype(np.float32)
    psf_base /= psf_base.sum()
    psf[:] = psf_base

    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["n_systems"] = int(n_systems)
        meta.attrs["random_seed"] = int(seed)
        meta.attrs["survey"] = survey
        meta.attrs["h0_truth_reference"] = float(h0_truth)
        meta.attrs["h0_sampling"] = "uniform_60_80_per_system"
        meta.attrs["generator_version"] = "phase3-v2.5"
        meta.attrs["round_tag"] = "v2.6_scale_up"
        meta.attrs["dt_noise_model"] = "gaussian_relative_uniform_0p02_0p07"
        meta.attrs["lc_delay_source"] = "dt_approx_noisy"
        meta.attrs["full_truth_available"] = False
        meta.attrs["perturbation_model"] = "kappa_ext_msd_v0"

        params = f.create_group("params")
        params.create_dataset("H0", data=arrays["H0"])
        params.create_dataset("z_lens", data=arrays["z_lens"])
        params.create_dataset("z_source", data=arrays["z_source"])
        params.create_dataset("lens_truth_model", data=np.array([b"SIE_phase3_v0"] * n_systems))
        params.create_dataset("lens_model", data=np.array([b"SIE_phase3_v0"] * n_systems))
        params.create_dataset("sigma_v", data=arrays["sigma_v"])
        params.create_dataset("M200", data=arrays["M200"])
        params.create_dataset("concentration", data=arrays["concentration"])
        params.create_dataset("q", data=arrays["q"])
        params.create_dataset("theta_E", data=arrays["theta_E"])
        params.create_dataset("phi", data=arrays["phi"])

        light_curves = f.create_group("light_curves")
        light_curves.create_dataset("F_joint", data=f_joint)
        light_curves.create_dataset("sigma_noise", data=sigma_noise)
        light_curves.create_dataset("t_obs", data=t_obs_all)
        light_curves.create_dataset("n_epochs", data=np.full(n_systems, n_epochs, dtype=np.int32))

        images_g = f.create_group("images")
        images_g.create_dataset("I_obs", data=i_obs)
        images_g.create_dataset("S_true", data=s_true)
        images_g.create_dataset("psf", data=psf)
        images_g.create_dataset("pixel_scale", data=np.full(n_systems, 5.0 / image_size, dtype=np.float32))

        truth = f.create_group("true_values")
        truth.create_dataset("dt_true", data=arrays["dt_true"])
        truth.create_dataset("mu_true", data=arrays["mu_true"])
        truth.create_dataset("theta_E", data=arrays["theta_E"])
        truth.create_dataset("H0_true", data=arrays["H0"])
        truth.create_dataset("dm_params_true", data=dm_params)
        truth.create_dataset("dm_dim", data=np.full(n_systems, min(mode2_dm_dim, 4), dtype=np.int32))
        truth.create_dataset("D_delta_t", data=arrays["D_delta_t"])

        ray_paths = f.create_group("ray_paths")
        ray_paths.create_dataset("theta_1", data=theta_1)
        ray_paths.create_dataset("theta_2", data=theta_2)
        ray_paths.create_dataset("fermat_potential", data=fermat)

        approx = f.create_group("approx_outputs")
        approx.create_dataset("dt_approx", data=dt_approx_noisy_arr)
        approx.create_dataset("H0_approx", data=arrays["H0"] + mode1_h0_error_arr)
        approx.create_dataset("dm_params_approx", data=dm_params)
        approx.create_dataset("S_approx", data=s_approx)

        correction = f.create_group("correction_targets")
        correction.create_dataset("mode1_H0_correction", data=mode1_h0_error_arr)
        correction.create_dataset("mode2_dm_correction", data=zeros_dm)
        correction.create_dataset("mode3_source_correction", data=mode3_source_residual)

        legacy = f.create_group("simplification_errors")
        legacy.create_dataset("mode1_H0_error", data=mode1_h0_error_arr)
        legacy.create_dataset("mode2_dm_error", data=zeros_dm)
        legacy.create_dataset("mode3_source_residual", data=mode3_source_residual)
        legacy.create_dataset("approx_level_used", data=np.ones(n_systems, dtype=np.int32))

        perturbations = f.create_group("perturbations")
        perturbations.create_dataset("kappa_ext", data=kappa_ext_arr)
        perturbations.create_dataset("delta_t_sie", data=delta_t_sie_arr)
        perturbations.create_dataset("delta_t_obs", data=delta_t_obs_arr)
        perturbations.create_dataset("dt_sigma_rel", data=dt_sigma_rel_arr)
        perturbations.create_dataset("dt_approx_noise_factor", data=dt_approx_noise_factor_arr)
        perturbations.create_dataset("mean_I_sie", data=mean_i_sie_arr)
        perturbations.create_dataset("mean_I_obs_preclip", data=mean_i_obs_preclip_arr)
        perturbations.create_dataset("mean_I_obs", data=mean_i_obs_arr)
        perturbations.create_dataset("clip_fraction", data=clip_fraction_arr)

        f.create_dataset("sigma_curve", data=sigma_curves)
        f.create_dataset("target_mode", data=rng.integers(1, 4, n_systems, dtype=np.int32))

    print(f"wrote HDF5: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_systems", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "mock" / "real_phase3_v0.h5")
    parser.add_argument("--survey", type=str, default="ztf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    simulate_dataset(args.n_systems, args.output, survey=args.survey, seed=args.seed)


if __name__ == "__main__":
    main()
