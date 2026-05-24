"""YAML real-observation catalog adapter for Mode 1.

The YAML catalog is an external ingest format for Gaia GraL X + Bag+22 outputs.
Bag+22 is not called here: each entry must already contain ``dt_lc`` and
``dt_lc_sigma`` plus light-curve quality metrics. Units: delays/cadence [days],
redshifts dimensionless, sigma_v [km/s], theta_E [arcsec], q dimensionless.
SIE 표준 근사 가정: only public observation-side fields are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ml.training.feature_schema import TRUTH_ONLY_KEYS, validate_no_truth_keys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_observed_feature_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "config" / "ml.yaml").open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return dict(cfg.get("data", {}).get("observed_features", {}))


@dataclass(frozen=True)
class RealCatalogEntry:
    """Validated real-data Mode 1 entry with reserved Mode 2 fields."""

    name: str
    sources: dict[str, Any]
    z_lens: float
    z_source: float
    H0_approx: float | None
    sigma_v: float | None
    theta_E: float | None
    q: float | None
    dt_lc: float
    dt_lc_sigma: float
    n_epochs_quality: float
    baseline_days: float
    median_cadence_days: float
    median_photometric_error: float
    dphi_rad2: float | None
    image_pair_convention: str | None
    pair_order: dict[str, Any]
    conversion_log: tuple[str, ...]
    mode2_inputs: dict[str, Any]

    def to_feature_spec(self) -> dict[str, Any]:
        """Return a public feature spec accepted by ``build_corrector_inputs``."""

        spec = {
            "name": self.name,
            "H0_approx": 70.0 if self.H0_approx is None else self.H0_approx,
            "z_lens": self.z_lens,
            "z_source": self.z_source,
            "sigma_v": self.sigma_v,
            "theta_E": self.theta_E,
            "q": self.q,
            "dt_lc": self.dt_lc,
            "dt_lc_sigma": self.dt_lc_sigma,
            "n_epochs_quality": self.n_epochs_quality,
            "baseline_days": self.baseline_days,
            "median_cadence_days": self.median_cadence_days,
            "median_photometric_error": self.median_photometric_error,
            "image_pair_convention": self.image_pair_convention,
            "pair_order": dict(self.pair_order),
            "conversion_log": list(self.conversion_log),
            # Reserved for Mode 2; ignored by Mode 1 feature builder.
            "mode2_inputs": dict(self.mode2_inputs),
        }
        if self.dphi_rad2 is not None:
            spec["dphi_rad2"] = self.dphi_rad2
        return spec


def _finite_float(value: Any, field: str, *, required: bool = True) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    try:
        out = float(np.asarray(value).reshape(-1)[0])
    except Exception as exc:
        raise ValueError(f"{field} must be a finite scalar") from exc
    if not np.isfinite(out):
        if required:
            raise ValueError(f"{field} must be finite")
        return None
    return out


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _reject_truth_keys_recursive(mapping: Mapping[str, Any], prefix: str = "") -> None:
    validate_no_truth_keys(mapping)
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            _reject_truth_keys_recursive(value, f"{prefix}{key}.")
        elif key in TRUTH_ONLY_KEYS:
            raise ValueError(f"truth-only key {prefix}{key} is not allowed")


def _pair_order_from_mapping(time_delay: Mapping[str, Any]) -> dict[str, Any]:
    pair_order = time_delay.get("pair_order", {}) or {}
    if pair_order and not isinstance(pair_order, Mapping):
        raise ValueError("time_delay.pair_order must be a mapping when present")
    out = dict(pair_order)
    if "leading_image" in time_delay:
        out.setdefault("leading_image", time_delay["leading_image"])
    if "trailing_image" in time_delay:
        out.setdefault("trailing_image", time_delay["trailing_image"])
    return out


def _flip_pair_order(pair_order: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(pair_order)
    leading = out.get("leading_image")
    trailing = out.get("trailing_image")
    if leading is not None or trailing is not None:
        out["leading_image"] = trailing
        out["trailing_image"] = leading
    return out


def entry_from_mapping(
    mapping: Mapping[str, Any],
    index: int = 0,
    observed_feature_config: Mapping[str, Any] | None = None,
) -> RealCatalogEntry:
    """Validate one YAML catalog entry and normalize field names."""

    observed_feature_config = observed_feature_config or _default_observed_feature_config()
    _reject_truth_keys_recursive(mapping)
    name = str(mapping.get("name") or mapping.get("lens_id") or f"system_{index}")
    redshifts = _mapping(mapping.get("redshifts", {}), "redshifts")
    time_delay = _mapping(mapping.get("time_delay", {}), "time_delay")
    quality = _mapping(mapping.get("light_curve_quality", {}), "light_curve_quality")
    kinematics = mapping.get("kinematics", {}) or {}
    lens_model = mapping.get("lens_model", {}) or {}
    if not isinstance(kinematics, Mapping) or not isinstance(lens_model, Mapping):
        raise ValueError("kinematics and lens_model must be mappings when present")

    z_lens = _finite_float(redshifts.get("z_lens", mapping.get("z_lens")), "redshifts.z_lens")
    z_source = _finite_float(redshifts.get("z_source", mapping.get("z_source")), "redshifts.z_source")
    assert z_lens is not None and z_source is not None
    if not z_source > z_lens + 0.05:
        raise ValueError("redshifts.z_source must exceed redshifts.z_lens + 0.05")

    dt_lc = _finite_float(time_delay.get("dt_lc", mapping.get("dt_lc")), "time_delay.dt_lc")
    dt_lc_sigma = _finite_float(
        time_delay.get("dt_lc_sigma", mapping.get("dt_lc_sigma")),
        "time_delay.dt_lc_sigma",
    )
    assert dt_lc is not None and dt_lc_sigma is not None
    pair_order = _pair_order_from_mapping(time_delay)
    conversion_log: list[str] = []
    if dt_lc < 0.0:
        sign_policy = observed_feature_config.get("dt_sign_convention", {})
        if not isinstance(sign_policy, Mapping):
            raise ValueError("observed_features.dt_sign_convention must be a mapping")
        if sign_policy.get("negative_input_action") != "abs_and_flip_pair_order":
            raise ValueError("negative time_delay.dt_lc is not allowed by configured dt_sign_convention")
        dt_lc = abs(dt_lc)
        pair_order = _flip_pair_order(pair_order)
        conversion_log.append("negative time_delay.dt_lc converted to abs(dt_lc); pair_order flipped")
    if dt_lc <= 0.0 or dt_lc_sigma <= 0.0:
        raise ValueError("time_delay.dt_lc and dt_lc_sigma must be positive after sign normalization")

    mode2_inputs = mapping.get("mode2_inputs", {}) or {}
    if not isinstance(mode2_inputs, Mapping):
        raise ValueError("mode2_inputs must be a mapping when present")

    return RealCatalogEntry(
        name=name,
        sources=dict(mapping.get("sources", {}) or {}),
        z_lens=z_lens,
        z_source=z_source,
        H0_approx=_finite_float(lens_model.get("H0_approx"), "lens_model.H0_approx", required=False),
        sigma_v=_finite_float(kinematics.get("sigma_v", lens_model.get("sigma_v")), "sigma_v", required=False),
        theta_E=_finite_float(lens_model.get("theta_E"), "lens_model.theta_E", required=False),
        q=_finite_float(lens_model.get("q"), "lens_model.q", required=False),
        dt_lc=dt_lc,
        dt_lc_sigma=dt_lc_sigma,
        n_epochs_quality=_finite_float(
            quality.get("N_epochs", quality.get("n_epochs_quality")),
            "light_curve_quality.N_epochs",
        ) or 0.0,
        baseline_days=_finite_float(quality.get("baseline_days"), "light_curve_quality.baseline_days") or 0.0,
        median_cadence_days=_finite_float(
            quality.get("median_cadence_days"),
            "light_curve_quality.median_cadence_days",
        ) or 0.0,
        median_photometric_error=_finite_float(
            quality.get("median_photometric_error"),
            "light_curve_quality.median_photometric_error",
        ) or 0.0,
        dphi_rad2=_finite_float(lens_model.get("dphi_rad2"), "lens_model.dphi_rad2", required=False),
        image_pair_convention=(
            None
            if time_delay.get("image_pair_convention") is None
            else str(time_delay.get("image_pair_convention"))
        ),
        pair_order=pair_order,
        conversion_log=tuple(conversion_log),
        mode2_inputs=dict(mode2_inputs),
    )


def load_yaml_catalog(
    path: str | Path,
    observed_feature_config: Mapping[str, Any] | None = None,
) -> list[RealCatalogEntry]:
    """Load a YAML list of real observed systems."""

    with Path(path).open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    if not isinstance(loaded, list):
        raise ValueError("real observation catalog YAML must contain a top-level list")
    cfg = observed_feature_config or _default_observed_feature_config()
    return [entry_from_mapping(entry, idx, observed_feature_config=cfg) for idx, entry in enumerate(loaded)]
