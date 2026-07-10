"""Mode 1 ML correction domain-membership scoring.

The scorer uses only inference-side quantities. Units: delay features are
[days], lens angular quantities are [arcsec], and H0 is [km/s/Mpc]. SIE 표준
근사 가정: this module gates the learned correction around the fixed SIE
approximation and does not introduce a new physical approximation axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from ml.training.feature_schema import legacy_delay_sigma_from_config
from ml.utils.normalize import build_param_vector


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "data" / "mock" / "phase4_v0_4.h5"
DEFAULT_PROFILE = ROOT / "data" / "logs" / "mode1_domain_profile_phase4_v0_4.json"
FEATURE_NAMES = tuple(
    [
        "H0_approx",
        "z_lens",
        "z_source",
        "sigma_v",
        "q",
        "theta_E",
        "dt_lc",
        "dt_lc_sigma",
        "n_epochs_quality",
        "baseline_days",
        "median_cadence_days",
        "median_photometric_error",
        "missing_sigma_v",
        "missing_theta_E",
        "missing_q",
        "lc_flux_absmax",
        "lc_noise_absmax",
        "image_sum",
    ]
)
DEFAULT_BORDERLINE_SIGMA_MULTIPLIER = 2.0
MAHALANOBIS_REGULARIZATION = 1.0e-6


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _load_cfg(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    if not isinstance(loaded, dict):
        raise ValueError("ML config must contain a mapping")
    return loaded


def _quality_from_catalog(h5: h5py.File, idx: int, n_epochs: int) -> dict[str, float]:
    quality_paths = {
        "n_epochs_quality": "light_curve_quality/n_epochs_quality",
        "baseline_days": "light_curve_quality/baseline_days",
        "median_cadence_days": "light_curve_quality/median_cadence_days",
        "median_photometric_error": "light_curve_quality/median_photometric_error",
    }
    out: dict[str, float] = {"n_epochs_quality": float(n_epochs)}
    for key, path in quality_paths.items():
        if path in h5:
            out[key] = _finite_float(h5[path][idx], out.get(key, 0.0))
    if "observed_features" in h5:
        for key in quality_paths:
            path = f"observed_features/{key}"
            if path in h5:
                out[key] = _finite_float(h5[path][idx], out.get(key, 0.0))
    return out


def _catalog_feature_matrix(catalog_path: str | Path, config_path: str | Path) -> np.ndarray:
    cfg = _load_cfg(config_path)
    data_cfg = cfg["data"]
    norm_cfg = data_cfg["param_normalization"]
    observed_cfg = data_cfg.get("observed_features", {})
    rows: list[np.ndarray] = []
    with h5py.File(catalog_path, "r") as h5:
        n = int(h5["approx_outputs/H0_approx"].shape[0])
        for idx in range(n):
            n_epochs = int(np.asarray(h5["light_curves/n_epochs"][idx]).reshape(-1)[0])
            raw = {
                "H0_approx": h5["approx_outputs/H0_approx"][idx],
                "z_lens": h5["params/z_lens"][idx],
                "z_source": h5["params/z_source"][idx],
                "sigma_v": h5["params/sigma_v"][idx],
                "q": h5["params/q"][idx],
                "theta_E": h5["params/theta_E"][idx],
                "dt_lc": h5["observed_features/dt_lc"][idx]
                if "observed_features/dt_lc" in h5
                else h5["approx_outputs/dt_approx"][idx],
                "dt_lc_sigma": h5["observed_features/dt_lc_sigma"][idx]
                if "observed_features/dt_lc_sigma" in h5
                else None,
                **_quality_from_catalog(h5, idx, n_epochs),
            }
            if raw["dt_lc_sigma"] is None:
                raw["dt_lc_sigma"] = legacy_delay_sigma_from_config(
                    _finite_float(raw["dt_lc"], 0.0),
                    observed_cfg,
                )
            param = build_param_vector(raw, norm_cfg)
            flux = np.asarray(h5["light_curves/F_joint"][idx], dtype=np.float64)[:n_epochs]
            noise = np.asarray(h5["light_curves/sigma_noise"][idx], dtype=np.float64)[:n_epochs]
            image_sum = (
                float(np.sum(np.asarray(h5["images/I_obs"][idx], dtype=np.float64)))
                if "images/I_obs" in h5
                else 0.0
            )
            tail = np.asarray(
                [
                    np.nanmax(np.abs(flux)) if flux.size else np.nan,
                    np.nanmax(np.abs(noise)) if noise.size else np.nan,
                    image_sum,
                ],
                dtype=np.float64,
            )
            row = np.concatenate([param.astype(np.float64), tail])
            if np.isfinite(row).all():
                rows.append(row)
    if not rows:
        raise ValueError(f"no finite Mode 1 domain rows could be read from {catalog_path}")
    return np.vstack(rows)


def _default_profile_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    n = len(cfg["data"]["param_normalization"]) + 3
    center = np.zeros(n, dtype=np.float64)
    center[: len(cfg["data"]["param_normalization"])] = 0.5
    scale = np.ones(n, dtype=np.float64)
    return {
        "version": "config_fallback_v1",
        "source_catalog": None,
        "feature_names": list(FEATURE_NAMES),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "cov_inv": np.eye(n, dtype=np.float64).tolist(),
        "mahalanobis_sq_p95": float(n * 2.0),
        "mahalanobis_sq_p99": float(n * 3.0),
        "scalar_quantiles": {},
        "borderline_sigma_multiplier": DEFAULT_BORDERLINE_SIGMA_MULTIPLIER,
        "note": "Fallback profile used because no Phase4 catalog/profile artifact was available.",
    }


def build_mode1_domain_profile(
    catalog_path: str | Path = DEFAULT_CATALOG,
    *,
    config_path: str | Path = ROOT / "config" / "ml.yaml",
) -> dict[str, Any]:
    matrix = _catalog_feature_matrix(catalog_path, config_path)
    center = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25.0, axis=0)
    q75 = np.percentile(matrix, 75.0, axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1.0e-6)
    z = (matrix - center) / scale
    cov = np.cov(z, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64) + np.eye(z.shape[1]) * MAHALANOBIS_REGULARIZATION
    cov_inv = np.linalg.pinv(cov)
    md2 = np.einsum("ij,jk,ik->i", z, cov_inv, z)
    quantiles: dict[str, dict[str, float]] = {}
    for i, name in enumerate(FEATURE_NAMES):
        vals = matrix[:, i]
        quantiles[name] = {
            "p01": float(np.percentile(vals, 1.0)),
            "p05": float(np.percentile(vals, 5.0)),
            "p50": float(np.percentile(vals, 50.0)),
            "p95": float(np.percentile(vals, 95.0)),
            "p99": float(np.percentile(vals, 99.0)),
        }
    return {
        "version": "phase4_v0_4_domain_profile_v1",
        "source_catalog": str(Path(catalog_path)),
        "feature_names": list(FEATURE_NAMES),
        "n_systems": int(matrix.shape[0]),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "cov_inv": cov_inv.tolist(),
        "mahalanobis_sq_p95": float(np.percentile(md2, 95.0)),
        "mahalanobis_sq_p99": float(np.percentile(md2, 99.0)),
        "scalar_quantiles": quantiles,
        "inference_side_v0_2_thresholds": {
            "lc_flux_absmax_max": 3.408,
            "image_sum_max": 77.79,
            "dt_lc_max_days": 444.7,
            "mu_time_delay_abs_max": 0.9699,
            "image_separation_min_arcsec": 0.6598,
        },
        "borderline_sigma_multiplier": DEFAULT_BORDERLINE_SIGMA_MULTIPLIER,
    }


def write_mode1_domain_profile(
    output_path: str | Path = DEFAULT_PROFILE,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG,
    config_path: str | Path = ROOT / "config" / "ml.yaml",
) -> dict[str, Any]:
    profile = build_mode1_domain_profile(catalog_path, config_path=config_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(profile, fp, indent=2, sort_keys=True)
    return profile


def load_or_build_mode1_domain_profile(
    profile_path: str | Path | None,
    *,
    config_path: str | Path,
    catalog_path: str | Path = DEFAULT_CATALOG,
) -> tuple[dict[str, Any], str]:
    if profile_path is not None and Path(profile_path).exists():
        with Path(profile_path).open("r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        return loaded, str(Path(profile_path))
    if Path(catalog_path).exists():
        return build_mode1_domain_profile(catalog_path, config_path=config_path), f"built_from:{Path(catalog_path)}"
    return _default_profile_from_config(config_path), "config_fallback"


def _image_separation_arcsec(sie_fit: Mapping[str, Any]) -> float | None:
    try:
        theta1 = np.asarray(sie_fit["theta_1"], dtype=float).reshape(-1)
        theta2 = np.asarray(sie_fit["theta_2"], dtype=float).reshape(-1)
    except Exception:
        return None
    if theta1.size < 2 or theta2.size < 2:
        return None
    sep = float(np.linalg.norm(theta1[:2] - theta2[:2]))
    return sep if np.isfinite(sep) else None


def build_mode1_domain_features(
    *,
    params_vector: np.ndarray,
    lc_tensor: np.ndarray,
    n_valid_lc: int,
    image_tensor: np.ndarray,
    use_image: bool,
) -> dict[str, Any]:
    params = np.asarray(params_vector, dtype=np.float64).reshape(-1)
    n_base = len(FEATURE_NAMES) - 3
    if params.size < n_base:
        raise ValueError(f"params vector must contain at least {n_base} normalized features")
    lc = np.asarray(lc_tensor, dtype=np.float64)
    if lc.ndim != 2 or lc.shape[0] < 2:
        raise ValueError("lc tensor must have shape [2, T]")
    n = max(0, min(int(n_valid_lc), int(lc.shape[1])))
    flux = lc[0, :n]
    noise = lc[1, :n]
    image_sum = float(np.sum(np.asarray(image_tensor, dtype=np.float64))) if use_image else 0.0
    vector = np.concatenate(
        [
            params[:n_base],
            np.asarray(
                [
                    np.nanmax(np.abs(flux)) if flux.size else np.nan,
                    np.nanmax(np.abs(noise)) if noise.size else np.nan,
                    image_sum,
                ],
                dtype=np.float64,
            ),
        ]
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "vector": vector,
        "n_valid_lc": int(n),
        "use_image": bool(use_image),
        "lc_flux_absmax": float(vector[-3]),
        "lc_noise_absmax": float(vector[-2]),
        "image_sum": float(vector[-1]),
    }


def score_mode1_domain_membership(
    *,
    features: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_artifact: str,
    delay: Mapping[str, Any],
    sie_fit: Mapping[str, Any],
    lc_normalization: Mapping[str, Any],
) -> dict[str, Any]:
    vector = np.asarray(features["vector"], dtype=np.float64).reshape(-1)
    center = np.asarray(profile["center"], dtype=np.float64).reshape(-1)
    scale = np.asarray(profile["scale"], dtype=np.float64).reshape(-1)
    cov_inv = np.asarray(profile["cov_inv"], dtype=np.float64)
    failed: list[str] = []
    warnings: list[str] = []
    critical_tail = False

    if vector.shape != center.shape or cov_inv.shape != (vector.size, vector.size):
        failed.append("profile_shape_mismatch")
    if not np.isfinite(vector).all():
        failed.append("non_finite_domain_feature")
    if _finite_float(lc_normalization.get("flux_std"), -1.0) <= 0.0:
        failed.append("invalid_lc_normalization")
    if int(lc_normalization.get("n_valid", 0)) <= 1:
        failed.append("invalid_lc_n_valid")
    mu = _finite_float(delay.get("mu"), np.nan)
    if not np.isfinite(mu) or abs(mu) >= 1.0:
        failed.append("mu_time_delay_abs_ge_1")
    dt_lc = _finite_float(delay.get("dt_obs_days"), np.nan)
    if not np.isfinite(dt_lc) or dt_lc <= 0.0:
        failed.append("nonpositive_delay")

    thresholds = profile.get("inference_side_v0_2_thresholds", {})
    if thresholds:
        if np.isfinite(dt_lc) and dt_lc > float(thresholds.get("dt_lc_max_days", np.inf)):
            warnings.append("dt_lc_above_inference_tail")
            critical_tail = True
        if np.isfinite(mu) and abs(mu) > float(thresholds.get("mu_time_delay_abs_max", np.inf)):
            warnings.append("mu_time_delay_above_v0_2_tail")
            critical_tail = True
        if features.get("use_image", False):
            image_sum = _finite_float(features.get("image_sum"), np.nan)
            if image_sum > float(thresholds.get("image_sum_max", np.inf)):
                warnings.append("image_sum_above_v0_2_tail")
                critical_tail = True
        else:
            warnings.append("image_missing_borderline")
        sep = _image_separation_arcsec(sie_fit)
        if sep is not None and sep < float(thresholds.get("image_separation_min_arcsec", -np.inf)):
            warnings.append("image_separation_below_v0_2_tail")
            critical_tail = True
    if _finite_float(features.get("lc_flux_absmax"), 0.0) > float(
        thresholds.get("lc_flux_absmax_max", np.inf)
    ):
        warnings.append("lc_flux_absmax_above_v0_2_tail")
        critical_tail = True

    if failed:
        return {
            "domain_score": None,
            "domain_grade": "ood_abstain",
            "failed_checks": failed,
            "warnings": warnings,
            "sigma_scale_regime": "abstain",
            "sigma_scale_multiplier": None,
            "benchmark_use": False,
            "profile_artifact": profile_artifact,
        }

    z = (vector - center) / np.maximum(scale, 1.0e-6)
    md2 = float(np.einsum("i,ij,j->", z, cov_inv, z))
    p95 = float(profile["mahalanobis_sq_p95"])
    p99 = float(profile["mahalanobis_sq_p99"])
    if md2 <= p95 and not warnings:
        grade = "in_distribution"
        regime = "default"
        multiplier = 1.0
        benchmark_use = True
    elif md2 <= p99 or not critical_tail:
        grade = "borderline"
        regime = "borderline_conservative"
        multiplier = float(profile.get("borderline_sigma_multiplier", DEFAULT_BORDERLINE_SIGMA_MULTIPLIER))
        benchmark_use = False
        if md2 > p99:
            warnings.append("mahalanobis_above_p99")
    else:
        grade = "ood_abstain"
        regime = "abstain"
        multiplier = None
        benchmark_use = False
        if md2 > p99:
            failed.append("mahalanobis_above_p99")
        if critical_tail:
            failed.append("critical_inference_tail_failure")

    return {
        "domain_score": md2,
        "mahalanobis_sq_p95": p95,
        "mahalanobis_sq_p99": p99,
        "domain_grade": grade,
        "failed_checks": failed,
        "warnings": warnings,
        "sigma_scale_regime": regime,
        "sigma_scale_multiplier": multiplier,
        "benchmark_use": benchmark_use,
        "profile_artifact": profile_artifact,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build Mode 1 ML domain profile")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--config", default=str(ROOT / "config" / "ml.yaml"))
    parser.add_argument("--output", default=str(DEFAULT_PROFILE))
    args = parser.parse_args(argv)
    profile = write_mode1_domain_profile(args.output, catalog_path=args.catalog, config_path=args.config)
    print(json.dumps({"output": args.output, "n_systems": profile.get("n_systems")}, indent=2))
    return profile


if __name__ == "__main__":
    main()
