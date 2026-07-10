from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from inversion.delay_extraction import extract_delay_from_observation
from inversion.observation_io import from_hdf5
from inversion.sie_fit import fit_sie_to_images
from ml.inference.domain_membership import (
    DEFAULT_CATALOG,
    FEATURE_NAMES,
    build_mode1_domain_features,
    build_mode1_domain_profile,
    score_mode1_domain_membership,
)
from ml.inference.mode1 import build_mode1_batch, load_cfg


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "ml.yaml"
J1226 = ROOT / "data" / "observations" / "sdss_j1226_observed.h5"
J1226_DELAY_CFG = ROOT / "data" / "observations" / "sdss_j1226_delay_config.json"


def _profile() -> dict:
    n = len(FEATURE_NAMES)
    return {
        "feature_names": list(FEATURE_NAMES),
        "center": [0.0] * n,
        "scale": [1.0] * n,
        "cov_inv": np.eye(n).tolist(),
        "mahalanobis_sq_p95": 50.0,
        "mahalanobis_sq_p99": 100.0,
        "inference_side_v0_2_thresholds": {
            "lc_flux_absmax_max": 3.408,
            "image_sum_max": 77.79,
            "dt_lc_max_days": 444.7,
            "mu_time_delay_abs_max": 0.9699,
            "image_separation_min_arcsec": 0.6598,
        },
        "borderline_sigma_multiplier": 2.0,
    }


def _features(vector: np.ndarray | None = None, *, use_image: bool = True) -> dict:
    if vector is None:
        vector = np.zeros(len(FEATURE_NAMES), dtype=float)
        vector[-3] = 1.0
        vector[-2] = 0.1
        vector[-1] = 1.0 if use_image else 0.0
    return {
        "feature_names": list(FEATURE_NAMES),
        "vector": np.asarray(vector, dtype=float),
        "n_valid_lc": 100,
        "use_image": use_image,
        "lc_flux_absmax": float(vector[-3]),
        "lc_noise_absmax": float(vector[-2]),
        "image_sum": float(vector[-1]),
    }


def _delay(mu: float = 0.5, dt: float = 10.0) -> dict:
    return {"mu": mu, "dt_obs_days": dt}


def _sie_fit(separation: float = 1.0) -> dict:
    return {"theta_1": [separation, 0.0], "theta_2": [0.0, 0.0]}


def _lc_norm(std: float = 1.0, n_valid: int = 100) -> dict:
    return {"flux_std": std, "n_valid": n_valid}


def test_domain_membership_in_distribution() -> None:
    got = score_mode1_domain_membership(
        features=_features(),
        profile=_profile(),
        profile_artifact="test_profile",
        delay=_delay(),
        sie_fit=_sie_fit(),
        lc_normalization=_lc_norm(),
    )
    assert got["domain_grade"] == "in_distribution"
    assert got["sigma_scale_regime"] == "default"
    assert got["benchmark_use"] is True


def test_domain_membership_missing_image_is_borderline_not_abstain() -> None:
    got = score_mode1_domain_membership(
        features=_features(use_image=False),
        profile=_profile(),
        profile_artifact="test_profile",
        delay=_delay(),
        sie_fit=_sie_fit(),
        lc_normalization=_lc_norm(),
    )
    assert got["domain_grade"] == "borderline"
    assert got["sigma_scale_regime"] == "borderline_conservative"
    assert got["sigma_scale_multiplier"] == 2.0
    assert "image_missing_borderline" in got["warnings"]


@pytest.mark.parametrize(
    "delay,lc_norm,expected",
    [
        (_delay(mu=1.0), _lc_norm(), "mu_time_delay_abs_ge_1"),
        (_delay(dt=0.0), _lc_norm(), "nonpositive_delay"),
        (_delay(), _lc_norm(std=0.0), "invalid_lc_normalization"),
    ],
)
def test_domain_membership_hard_abstain(delay: dict, lc_norm: dict, expected: str) -> None:
    got = score_mode1_domain_membership(
        features=_features(),
        profile=_profile(),
        profile_artifact="test_profile",
        delay=delay,
        sie_fit=_sie_fit(),
        lc_normalization=lc_norm,
    )
    assert got["domain_grade"] == "ood_abstain"
    assert got["sigma_scale_regime"] == "abstain"
    assert expected in got["failed_checks"]


def test_domain_membership_uses_no_truth_only_features() -> None:
    truth_only = {"mu_truth", "dphi_sie_over_truth", "mode1_H0_correction", "H0_true"}
    assert not truth_only.intersection(FEATURE_NAMES)


@pytest.mark.skipif(not DEFAULT_CATALOG.exists(), reason="Phase4 v0.4 catalog not available")
def test_phase4_v0_4_catalog_profile_scores_center_in_distribution() -> None:
    profile = build_mode1_domain_profile(DEFAULT_CATALOG, config_path=CONFIG)
    features = _features(np.asarray(profile["center"], dtype=float), use_image=True)
    got = score_mode1_domain_membership(
        features=features,
        profile=profile,
        profile_artifact="built_from_catalog",
        delay=_delay(),
        sie_fit=_sie_fit(),
        lc_normalization=_lc_norm(),
    )
    assert got["domain_grade"] == "in_distribution"
    assert profile["n_systems"] > 0


@pytest.mark.skipif(
    not (J1226.exists() and J1226_DELAY_CFG.exists()),
    reason="J1226 real observation artifact not available",
)
def test_j1226_domain_membership_not_abstained() -> None:
    cfg = load_cfg(CONFIG)
    observation = from_hdf5(J1226)
    delay_cfg = json.loads(J1226_DELAY_CFG.read_text(encoding="utf-8"))
    delay = extract_delay_from_observation(
        observation,
        delay_cfg,
        is_mock=False,
        return_grid=True,
    )
    sie_fit = fit_sie_to_images(
        observation.image_positions,
        observation.z_lens,
        observation.z_source,
        cosmology={"H0": 70.0},
    )
    batch, metadata = build_mode1_batch(
        observation=observation,
        input_path=J1226,
        system_index=0,
        delay=delay,
        sie_fit=sie_fit,
        h0_approx=58.9344,
        cfg=cfg,
        correction_approx_level=1,
    )
    profile = build_mode1_domain_profile(DEFAULT_CATALOG, config_path=CONFIG) if DEFAULT_CATALOG.exists() else _profile()
    features = build_mode1_domain_features(
        params_vector=batch["params"].numpy().reshape(-1),
        lc_tensor=batch["lc"].numpy().reshape(2, -1),
        n_valid_lc=int(metadata["n_valid_lc"]),
        image_tensor=batch["image"].numpy().reshape(batch["image"].shape[-2], batch["image"].shape[-1]),
        use_image=bool(metadata["use_image"]),
    )
    got = score_mode1_domain_membership(
        features=features,
        profile=profile,
        profile_artifact="test_profile",
        delay=delay,
        sie_fit=sie_fit,
        lc_normalization=metadata["lc_normalization"],
    )
    assert got["domain_grade"] != "ood_abstain"
    assert got["profile_artifact"] == "test_profile"
