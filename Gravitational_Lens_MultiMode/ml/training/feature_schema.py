"""Shared ParamEncoder feature schema for simulated and observed Mode inputs.

This module is the single source of truth for scalar features that feed the
shared ParamEncoder. Units: H0 [km/s/Mpc], sigma_v [km/s], theta_E [arcsec],
delays and cadence [days], photometric error [mag]. SIE 표준 근사 가정: these
features are public inference-side quantities only; truth-only full_numerical
keys are not accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

OPTIONAL_LENS_FIELDS = ("sigma_v", "theta_E", "q")
REQUIRED_DELAY_FIELDS = ("dt_lc", "dt_lc_sigma")
QUALITY_FIELDS = (
    "n_epochs_quality",
    "baseline_days",
    "median_cadence_days",
    "median_photometric_error",
)
MISSING_FLAG_FIELDS = ("missing_sigma_v", "missing_theta_E", "missing_q")
TRUTH_ONLY_KEYS = frozenset({"M200", "concentration", "kappa_ext", "nfw_offset"})
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_observed_feature_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "config" / "ml.yaml").open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return dict(cfg.get("data", {}).get("observed_features", {}))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _normalization_min(norm_cfg: Mapping[str, Any], key: str, fallback: float = 0.0) -> float:
    bounds = norm_cfg.get(key, {})
    if "min" in bounds:
        return float(bounds["min"])
    return float(fallback)


def _normalize_scalar(value: float, bounds: Mapping[str, Any]) -> float:
    lo = float(bounds.get("min", 0.0))
    hi = float(bounds.get("max", 1.0))
    if hi == lo:
        return 0.0
    transform = str(bounds.get("transform", "linear")).lower()
    if transform in {"log", "log1p"}:
        if lo < 0.0 or hi < 0.0 or value < 0.0:
            raise ValueError("log-normalized ParamEncoder features must be non-negative")
        lo_t = np.log1p(lo)
        hi_t = np.log1p(hi)
        if hi_t == lo_t:
            return 0.0
        return float((np.log1p(float(value)) - lo_t) / (hi_t - lo_t))
    if transform != "linear":
        raise ValueError(f"unsupported ParamEncoder normalization transform: {transform!r}")
    return float((float(value) - lo) / (hi - lo))


def normalize_param_features(raw: Mapping[str, float], norm_cfg: Mapping[str, Any]) -> np.ndarray:
    """Normalize scalar ParamEncoder features in config order.

    Each entry in ``norm_cfg`` may specify ``transform: linear`` or
    ``transform: log``/``log1p``. Missing numeric sentinels are represented by
    raw values equal to the configured minimum, which maps to normalized zero.
    """

    values: list[float] = []
    for key, bounds in norm_cfg.items():
        val = _finite_float(raw.get(key))
        if val is None:
            val = _normalization_min(norm_cfg, key)
        values.append(_normalize_scalar(float(val), bounds))
    return np.asarray(values, dtype=np.float32)


def dt_lc_sigma_from_sampler_config(
    dt_lc: float,
    sampler_cfg: Mapping[str, Any],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Sample or deterministically infer ``sigma_dt`` from config.

    Units: ``dt_lc`` and output sigma are [days]. The supported policy is
    ``relative_then_clip``: draw or infer a relative error, multiply by
    ``|dt_lc|``, then clip to configured absolute [days] bounds.
    """

    if str(sampler_cfg.get("mode", "")) != "relative_then_clip":
        raise ValueError("observed_features.dt_lc_sigma_sampler.mode must be relative_then_clip")
    rel_cfg = sampler_cfg.get("relative_error", {})
    if not isinstance(rel_cfg, Mapping):
        raise ValueError("observed_features.dt_lc_sigma_sampler.relative_error must be a mapping")
    if str(rel_cfg.get("distribution", "")) != "log_uniform":
        raise ValueError("dt_lc_sigma relative_error.distribution must be log_uniform")
    rel_min = float(rel_cfg["min"])
    rel_max = float(rel_cfg["max"])
    if not (0.0 < rel_min <= rel_max):
        raise ValueError("dt_lc_sigma relative_error min/max must be positive and ordered")
    if rng is None:
        rel = float(np.sqrt(rel_min * rel_max))
    else:
        rel = float(np.exp(rng.uniform(np.log(rel_min), np.log(rel_max))))
    floor = float(sampler_cfg["absolute_floor_days"])
    ceiling = float(sampler_cfg["absolute_ceiling_days"])
    if not (0.0 < floor <= ceiling):
        raise ValueError("dt_lc_sigma absolute floor/ceiling must be positive and ordered")
    sigma = float(np.clip(abs(float(dt_lc)) * rel, floor, ceiling))
    return sigma, rel


def legacy_delay_sigma_from_config(dt_lc: float, observed_feature_config: Mapping[str, Any]) -> float:
    """Config-driven deterministic fallback for legacy HDF5 without sigma_dt."""

    cfg = observed_feature_config or _default_observed_feature_config()
    sampler_cfg = cfg.get("dt_lc_sigma_sampler")
    if not isinstance(sampler_cfg, Mapping):
        raise ValueError("legacy HDF5 delay sigma fallback requires observed_features.dt_lc_sigma_sampler")
    sigma, _ = dt_lc_sigma_from_sampler_config(dt_lc, sampler_cfg, rng=None)
    return sigma


def _dataset_at(h5: Any, path: str, idx: int) -> float | None:
    if path not in h5:
        return None
    try:
        arr = np.asarray(h5[path][idx])
    except Exception:
        return None
    return _finite_float(arr)


def _first_dataset_at(h5: Any, paths: tuple[str, ...], idx: int) -> float | None:
    for path in paths:
        val = _dataset_at(h5, path, idx)
        if val is not None:
            return val
    return None


def compute_light_curve_quality(
    *,
    t_obs: np.ndarray | None,
    sigma_noise: np.ndarray | None,
    n_epochs: int,
) -> dict[str, float]:
    """Compute public light-curve quality metrics from observed cadence/noise.

    Units: ``t_obs`` [days], ``sigma_noise`` [mag or flux-error proxy]. SIE
    표준 근사 가정: this is observation-side metadata and contains no truth
    quantities. The output keys match ``config/ml.yaml``.
    """

    n_valid = max(int(n_epochs), 0)
    t = np.asarray([] if t_obs is None else t_obs, dtype=float).reshape(-1)[:n_valid]
    sig = np.asarray([] if sigma_noise is None else sigma_noise, dtype=float).reshape(-1)[:n_valid]
    finite_t = t[np.isfinite(t)]
    finite_sig = sig[np.isfinite(sig)]
    if finite_t.size >= 2:
        t_sorted = np.sort(finite_t)
        baseline = float(t_sorted[-1] - t_sorted[0])
        diffs = np.diff(t_sorted)
        cadence = float(np.median(diffs[diffs >= 0.0])) if diffs.size else 0.0
    else:
        baseline = 0.0
        cadence = 0.0
    phot_err = float(np.median(finite_sig)) if finite_sig.size else 0.0
    return {
        "n_epochs_quality": float(n_valid),
        "baseline_days": baseline,
        "median_cadence_days": cadence,
        "median_photometric_error": phot_err,
    }


def validate_no_truth_keys(mapping: Mapping[str, Any]) -> None:
    """Reject truth-only keys in public real-observation mappings."""

    present = sorted(key for key in TRUTH_ONLY_KEYS if key in mapping)
    if present:
        raise ValueError(f"truth-only keys are not allowed in observed inputs: {present}")


def build_raw_param_features(
    values: Mapping[str, Any],
    norm_cfg: Mapping[str, Any],
    *,
    observed_feature_config: Mapping[str, Any] | None = None,
    allow_legacy_delay_sigma: bool = False,
) -> dict[str, float]:
    """Return finite raw scalar features with explicit missing flags.

    ``dt_lc`` and ``dt_lc_sigma`` are required for new real-data entries. The
    legacy fallback is only for pre-existing simulated HDF5 files that did not
    store an observed delay uncertainty.
    """

    validate_no_truth_keys(values)
    observed_feature_config = observed_feature_config or {}
    missing_policy = str(observed_feature_config.get("missing_lens_value_policy", "normalized_zero"))
    if missing_policy != "normalized_zero":
        raise ValueError("observed_features.missing_lens_value_policy must be normalized_zero")
    raw: dict[str, float] = {}
    for key in ("H0_approx", "z_lens", "z_source"):
        val = _finite_float(values.get(key))
        if val is None:
            raise ValueError(f"required ParamEncoder feature {key!r} is missing or non-finite")
        raw[key] = val

    dt_lc = _finite_float(values.get("dt_lc", values.get("dt_approx")))
    if dt_lc is None or dt_lc <= 0.0:
        raise ValueError("required delay feature 'dt_lc' must be finite and positive")
    dt_sigma = _finite_float(values.get("dt_lc_sigma"))
    if dt_sigma is None and allow_legacy_delay_sigma:
        dt_sigma = legacy_delay_sigma_from_config(dt_lc, observed_feature_config or {})
    if dt_sigma is None or dt_sigma <= 0.0:
        raise ValueError("required delay feature 'dt_lc_sigma' must be finite and positive")
    raw["dt_lc"] = dt_lc
    raw["dt_lc_sigma"] = dt_sigma

    for key in OPTIONAL_LENS_FIELDS:
        val = _finite_float(values.get(key))
        missing = 1.0 if val is None else 0.0
        raw[key] = _normalization_min(norm_cfg, key) if val is None else float(val)
        raw[f"missing_{key}"] = missing

    quality = {
        "n_epochs_quality": values.get("n_epochs_quality", values.get("N_epochs")),
        "baseline_days": values.get("baseline_days"),
        "median_cadence_days": values.get("median_cadence_days"),
        "median_photometric_error": values.get("median_photometric_error"),
    }
    for key in QUALITY_FIELDS:
        raw[key] = float(_finite_float(quality.get(key)) or 0.0)
    return raw


def build_param_vector_from_features(
    values: Mapping[str, Any],
    norm_cfg: Mapping[str, Any],
    *,
    approx_level: int,
    target_mode: int,
    observed_feature_config: Mapping[str, Any] | None = None,
    allow_legacy_delay_sigma: bool = False,
) -> np.ndarray:
    """Build the complete ParamEncoder vector, including one-hot suffixes."""

    raw = build_raw_param_features(
        values,
        norm_cfg,
        observed_feature_config=observed_feature_config,
        allow_legacy_delay_sigma=allow_legacy_delay_sigma,
    )
    param_base = normalize_param_features(raw, norm_cfg)
    al_onehot = np.array(
        [float(int(approx_level) == 1), float(int(approx_level) == 2)],
        dtype=np.float32,
    )
    mode_oh = np.zeros(3, dtype=np.float32)
    mode_idx = int(target_mode) - 1
    if not 0 <= mode_idx < 3:
        raise ValueError(f"target_mode must be 1, 2, or 3; got {target_mode!r}")
    mode_oh[mode_idx] = 1.0
    return np.concatenate([param_base, al_onehot, mode_oh]).astype(np.float32)


def hdf5_feature_values(h5: Any, sys_idx: int) -> dict[str, float]:
    """Extract public scalar features from Phase 4 or legacy HDF5 schemas."""

    n_epochs = int(_first_dataset_at(h5, ("light_curves/n_epochs",), sys_idx) or 0)
    t_obs = np.asarray(h5["light_curves/t_obs"][sys_idx]) if "light_curves/t_obs" in h5 else None
    sigma_noise = (
        np.asarray(h5["light_curves/sigma_noise"][sys_idx])
        if "light_curves/sigma_noise" in h5
        else None
    )
    quality = {
        key: _first_dataset_at(
            h5,
            (f"observed_features/{key}", f"light_curve_quality/{key}"),
            sys_idx,
        )
        for key in QUALITY_FIELDS
    }
    if any(value is None for value in quality.values()):
        quality.update(compute_light_curve_quality(t_obs=t_obs, sigma_noise=sigma_noise, n_epochs=n_epochs))

    dt_lc = _first_dataset_at(
        h5,
        (
            "observed_features/dt_lc",
            "time_delay/dt_lc",
            "approx_outputs/dt_approx",
            # Legacy mock HDF5 fixture fallback only. Phase 4 catalogs and real
            # inputs must provide observed_features/time_delay values.
            "true_values/dt_true",
        ),
        sys_idx,
    )
    dt_lc_sigma = _first_dataset_at(
        h5,
        ("observed_features/dt_lc_sigma", "time_delay/dt_lc_sigma"),
        sys_idx,
    )
    h0_approx = _first_dataset_at(h5, ("approx_outputs/H0_approx", "params/H0"), sys_idx)
    theta_e = _first_dataset_at(h5, ("params/theta_E", "true_values/theta_E"), sys_idx)
    return {
        "H0_approx": h0_approx,
        "z_lens": _first_dataset_at(h5, ("params/z_lens",), sys_idx),
        "z_source": _first_dataset_at(h5, ("params/z_source",), sys_idx),
        "sigma_v": _first_dataset_at(h5, ("params/sigma_v",), sys_idx),
        "q": _first_dataset_at(h5, ("params/q",), sys_idx),
        "theta_E": theta_e,
        "dt_lc": dt_lc,
        "dt_lc_sigma": dt_lc_sigma,
        **quality,
    }
