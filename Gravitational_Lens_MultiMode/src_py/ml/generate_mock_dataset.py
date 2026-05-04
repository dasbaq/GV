"""
DRW 기반 mock HDF5 생성기.

Phase 3 full simulation과 Phase 4 error catalog 부재를 임시로 메우는 학습용
데이터를 만든다. 표준 근사 가정: SIE 렌즈, 단일 평면, κ_ext=0.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.dataset import LensCorrectionDataset
from ml.training.losses import composite_loss


def _normalize_unit(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _standardize(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def _drw_light_curve(
    rng: np.random.Generator,
    n_epochs: int,
    tau_days: float = 200.0,
    sigma_drw_mag: float = 0.1,
    sigma_noise_mag: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DRW/CARMA AR(1) quasar light curve.

    Units: time in days, flux channel in normalized magnitude residual units.
    표준 근사 가정: light-curve variability is observational input, independent
    of the fixed SIE lens approximation.
    """
    t_obs = np.linspace(0.0, 1000.0, n_epochs, dtype=np.float32)
    dt = float(t_obs[1] - t_obs[0])
    phi = float(np.exp(-dt / tau_days))
    innovations = rng.normal(0.0, sigma_drw_mag, n_epochs).astype(np.float32)

    flux = np.zeros(n_epochs, dtype=np.float32)
    for j in range(1, n_epochs):
        flux[j] = phi * flux[j - 1] + np.sqrt(1.0 - phi**2) * innovations[j]

    noisy = flux + rng.normal(0.0, sigma_noise_mag, n_epochs).astype(np.float32)
    normalized = _standardize(noisy)
    noise = np.full(n_epochs, sigma_noise_mag / (noisy.std() + 1e-6), dtype=np.float32)
    return normalized, noise, t_obs


def _gaussian_source(
    rng: np.random.Generator,
    image_size: int,
    theta_e: float,
    q: float,
) -> np.ndarray:
    """
    Elliptical Gaussian source image.

    Units: dimensionless normalized surface brightness on a pixel grid.
    표준 근사 가정: SIE-compatible extended source for Mode 3 mock truth.
    """
    y, x = np.mgrid[-1.0:1.0:complex(image_size), -1.0:1.0:complex(image_size)]
    cx = rng.uniform(-0.18, 0.18)
    cy = rng.uniform(-0.18, 0.18)
    sx = rng.uniform(0.06, 0.16) * (1.0 + 0.15 * theta_e)
    sy = sx * rng.uniform(0.55, 1.0) * q
    angle = rng.uniform(0.0, np.pi)
    ca = np.cos(angle)
    sa = np.sin(angle)
    xr = ca * (x - cx) + sa * (y - cy)
    yr = -sa * (x - cx) + ca * (y - cy)
    img = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return _normalize_unit(img)


def _sie_lensed_image(
    source: np.ndarray,
    theta_e: float,
    q: float,
    gamma_ext: float,
) -> np.ndarray:
    """
    SIE-inspired lensed image from an extended source.

    Units: dimensionless normalized surface brightness on a pixel grid.
    표준 근사 가정: SIE 렌즈, 단일 평면, κ_ext=0; gamma_ext is stored as truth
    structure and only used to perturb mock morphology.
    """
    image_size = source.shape[0]
    y, x = np.mgrid[-1.0:1.0:complex(image_size), -1.0:1.0:complex(image_size)]
    r_ell = np.sqrt((x * q) ** 2 + (y / max(q, 0.2)) ** 2) + 1e-4
    ring = np.exp(-0.5 * ((r_ell - 0.38 * theta_e) / 0.055) ** 2)
    shear = 1.0 + gamma_ext * np.cos(2.0 * np.arctan2(y, x))
    shifted_1 = np.roll(source, int(round(theta_e * 5)), axis=1)
    shifted_2 = np.roll(source, -int(round(theta_e * 4)), axis=0)
    image = ring * (0.7 * shifted_1 + 0.5 * shifted_2) * shear
    image += 0.20 * source
    return _normalize_unit(image)


def _correlated_image_error(
    rng: np.random.Generator,
    image_size: int,
    level: float,
) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0:complex(image_size), -1.0:1.0:complex(image_size)]
    err = np.zeros((image_size, image_size), dtype=np.float32)
    for _ in range(4):
        cx = rng.uniform(-0.7, 0.7)
        cy = rng.uniform(-0.7, 0.7)
        width = rng.uniform(0.12, 0.35)
        amp = rng.normal(0.0, level)
        err += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * width**2))
    return err.astype(np.float32)


def _sigma_curve(
    rng: np.random.Generator,
    n_points: int,
    dt_true: float,
) -> np.ndarray:
    grid = np.linspace(1.0, 100.0, n_points, dtype=np.float32)
    curve = rng.normal(0.0, 0.15, n_points).astype(np.float32)
    curve -= 2.5 * np.exp(-0.5 * ((grid - dt_true) / 4.0) ** 2).astype(np.float32)
    return _standardize(curve)


def create_mock_dataset(
    out_path: Path,
    n_samples: int = 10_000,
    seed: int = 42,
    t_len: int = 200,
    sigma_len: int = 50,
    image_size: int = 64,
) -> Path:
    """
    Production-compatible mock HDF5 dataset.

    Units: H0 [km/s/Mpc], theta_E [arcsec], sigma_v [km/s], light-curve time [days],
    images dimensionless normalized brightness. 표준 근사 가정: SIE 렌즈,
    단일 평면, κ_ext=0; full_numerical truth is represented by controlled
    correlated deviations from SIE approx.
    """
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h0_true = np.clip(rng.normal(70.0, 4.0, n_samples), 60.0, 80.0).astype(np.float32)
    z_lens = rng.uniform(0.1, 1.0, n_samples).astype(np.float32)
    z_source = np.maximum(z_lens + 0.15, rng.uniform(0.5, 3.5, n_samples)).astype(np.float32)
    z_source = np.clip(z_source, 0.5, 3.5).astype(np.float32)
    theta_e = rng.uniform(0.5, 2.0, n_samples).astype(np.float32)
    q = rng.uniform(0.5, 1.0, n_samples).astype(np.float32)
    gamma_ext = rng.uniform(0.0, 0.1, n_samples).astype(np.float32)
    sigma_v = (150.0 + (theta_e - 0.5) / 1.5 * 200.0).astype(np.float32)
    m200 = (10.0 ** rng.uniform(12.0, 14.0, n_samples)).astype(np.float32)
    concentration = rng.uniform(3.0, 15.0, n_samples).astype(np.float32)
    mu_true = rng.uniform(0.2, 0.95, n_samples).astype(np.float32)
    dt_true = rng.uniform(5.0, 95.0, n_samples).astype(np.float32)
    d_delta_t = rng.uniform(500.0, 3000.0, n_samples).astype(np.float32)
    target_mode = rng.integers(1, 4, n_samples, dtype=np.int32)

    f_joint = np.zeros((n_samples, t_len), dtype=np.float32)
    sigma_noise = np.zeros((n_samples, t_len), dtype=np.float32)
    t_obs = np.zeros((n_samples, t_len), dtype=np.float32)
    sigma_curve = np.zeros((n_samples, sigma_len), dtype=np.float32)
    i_obs = np.zeros((n_samples, image_size, image_size), dtype=np.float32)
    s_true = np.zeros_like(i_obs)
    s_approx = np.zeros_like(i_obs)
    mode3_correction = np.zeros_like(i_obs)

    dm_truth = np.column_stack([theta_e, q, gamma_ext, np.log10(m200)]).astype(np.float32)
    dm_dim = np.full(n_samples, 4, dtype=np.int32)

    h0_frac_error = rng.uniform(0.05, 0.15, n_samples).astype(np.float32)
    h0_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), n_samples)
    h0_approx = (h0_true * (1.0 + h0_sign * h0_frac_error)).astype(np.float32)
    mode1_correction = (h0_true - h0_approx).astype(np.float32)

    dm_level = rng.uniform(0.05, 0.15, (n_samples, 1)).astype(np.float32)
    dm_corr_driver = rng.normal(0.0, 1.0, (n_samples, 1)).astype(np.float32)
    dm_scale = np.array([1.0, 0.10, 0.03, 0.25], dtype=np.float32)
    mode2_correction = (dm_level * dm_corr_driver * dm_scale).astype(np.float32)
    dm_approx = (dm_truth - mode2_correction).astype(np.float32)

    psf = np.zeros((n_samples, 11, 11), dtype=np.float32)
    yy, xx = np.mgrid[-5:6, -5:6]
    psf_base = np.exp(-(xx**2 + yy**2) / (2.0 * 1.4**2)).astype(np.float32)
    psf_base /= psf_base.sum()
    psf[:] = psf_base

    for i in range(n_samples):
        f_joint[i], sigma_noise[i], t_obs[i] = _drw_light_curve(rng, t_len)
        sigma_curve[i] = _sigma_curve(rng, sigma_len, float(dt_true[i]))
        source = _gaussian_source(rng, image_size, float(theta_e[i]), float(q[i]))
        image = _sie_lensed_image(source, float(theta_e[i]), float(q[i]), float(gamma_ext[i]))
        image += rng.normal(0.0, 0.02, image.shape).astype(np.float32)
        image = _normalize_unit(image)
        level = float(rng.uniform(0.05, 0.15))
        residual = (_correlated_image_error(rng, image_size, level) * np.maximum(source, 0.15)).astype(np.float32)
        approx_source = np.clip(source - residual, 0.0, 1.0).astype(np.float32)
        s_true[i] = source
        i_obs[i] = image
        s_approx[i] = approx_source
        mode3_correction[i] = (source - approx_source).astype(np.float32)

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["n_systems"] = int(n_samples)
        meta.attrs["git_commit"] = "mock"
        meta.attrs["random_seed"] = int(seed)
        meta.attrs["full_truth_available"] = True

        params = f.create_group("params")
        params.create_dataset("H0", data=h0_true)
        params.create_dataset("z_lens", data=z_lens)
        params.create_dataset("z_source", data=z_source)
        params.create_dataset("lens_truth_model", data=np.array([b"DRW_mock_irregular_2D"] * n_samples))
        params.create_dataset("lens_model", data=np.array([b"DRW_mock_irregular_2D"] * n_samples))
        params.create_dataset("sigma_v", data=sigma_v)
        params.create_dataset("M200", data=m200)
        params.create_dataset("concentration", data=concentration)
        params.create_dataset("q", data=q)
        params.create_dataset("gamma_ext", data=gamma_ext)

        light_curves = f.create_group("light_curves")
        light_curves.create_dataset("F_joint", data=f_joint)
        light_curves.create_dataset("sigma_noise", data=sigma_noise)
        light_curves.create_dataset("t_obs", data=t_obs)
        light_curves.create_dataset("n_epochs", data=np.full(n_samples, t_len, dtype=np.int32))

        images = f.create_group("images")
        images.create_dataset("I_obs", data=i_obs)
        images.create_dataset("S_true", data=s_true)
        images.create_dataset("psf", data=psf)
        images.create_dataset("pixel_scale", data=np.full(n_samples, 0.05, dtype=np.float32))

        truth = f.create_group("true_values")
        truth.create_dataset("dt_true", data=dt_true)
        truth.create_dataset("mu_true", data=mu_true)
        truth.create_dataset("theta_E", data=theta_e)
        truth.create_dataset("H0_true", data=h0_true)
        truth.create_dataset("dm_params_true", data=dm_truth)
        truth.create_dataset("dm_dim", data=dm_dim)
        truth.create_dataset("D_delta_t", data=d_delta_t)

        ray_paths = f.create_group("ray_paths")
        ray_paths.create_dataset("theta_1", data=rng.uniform(-2.0, 2.0, (n_samples, 2)).astype(np.float32))
        ray_paths.create_dataset("theta_2", data=rng.uniform(-2.0, 2.0, (n_samples, 2)).astype(np.float32))
        ray_paths.create_dataset("fermat_potential", data=rng.uniform(0.1, 5.0, n_samples).astype(np.float32))

        approx = f.create_group("approx_outputs")
        approx.create_dataset("dt_approx", data=(dt_true * rng.uniform(0.95, 1.05, n_samples)).astype(np.float32))
        approx.create_dataset("H0_approx", data=h0_approx)
        approx.create_dataset("dm_params_approx", data=dm_approx)
        approx.create_dataset("S_approx", data=s_approx)

        targets = f.create_group("correction_targets")
        targets.create_dataset("mode1_H0_correction", data=mode1_correction)
        targets.create_dataset("mode2_dm_correction", data=mode2_correction)
        targets.create_dataset("mode3_source_correction", data=mode3_correction)

        legacy = f.create_group("simplification_errors")
        legacy.create_dataset("mode1_H0_error", data=mode1_correction)
        legacy.create_dataset("mode2_dm_error", data=mode2_correction)
        legacy.create_dataset("mode3_source_residual", data=mode3_correction)
        legacy.create_dataset("approx_level_used", data=np.ones(n_samples, dtype=np.int32))

        f.create_dataset("sigma_curve", data=sigma_curve)
        f.create_dataset("target_mode", data=target_mode)

    return out_path


def verify_dataset(path: Path) -> None:
    """
    End-to-end compatibility smoke test.

    Units follow LensCorrectionDataset tensors; 표준 근사 가정: generated labels are
    full_numerical mock truth minus fixed SIE approx outputs.
    """
    with open(PROJECT_ROOT / "config" / "ml.yaml") as f:
        norm_cfg = yaml.safe_load(f)["data"]["param_normalization"]

    dataset = LensCorrectionDataset(
        h5_paths=[path],
        split="train",
        modes=(1, 2, 3),
        approx_levels=(1, 2),
        max_len=200,
        sigma_curve_size=50,
        image_size=64,
        mode2_max_dm_dim=4,
        param_norm=norm_cfg,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=6, shuffle=False)
    batch = next(iter(loader))

    model = MultiModalErrorCorrector(
        {
            "d_model": 32,
            "n_heads": 4,
            "dropout": 0.1,
            "mode2_max_dm_dim": 4,
            "image_size": 64,
            "param_in_dim": 11,
        }
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["lc"],
            batch["lc_mask"],
            batch["params"],
            batch["sigma_curve"],
            batch["image"],
            batch["use_image"],
            batch["target_mode"],
        )
        losses = composite_loss(
            outputs,
            batch,
            {
                "mode1": 1.0,
                "mode2": 1.0,
                "mode3": 0.5,
                "physics": 0.1,
                "calibration": 0.1,
                "ssim": 0.1,
            },
        )

    assert torch.isfinite(losses["total"]).item(), "composite_loss produced NaN/Inf"
    assert outputs["mode1"]["h0_correction"].ndim == 1
    assert outputs["mode2"]["dm_correction"].shape[1] == 4
    assert outputs["mode3"]["source_residual"].shape[1:] == (1, 64, 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "mock" / "mock_dataset.h5")
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    out_path = create_mock_dataset(args.out, n_samples=args.n_samples, seed=args.seed)
    if not args.skip_verify:
        verify_dataset(out_path)
    print(f"Mock dataset saved: {out_path}")


if __name__ == "__main__":
    main()
