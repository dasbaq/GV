from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inversion.real_catalog import entry_from_mapping, load_yaml_catalog
from inversion.obs_to_features import build_corrector_inputs


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "real_catalog"
COMPLETE = FIXTURE_DIR / "complete.yaml"
PARTIAL = FIXTURE_DIR / "partial_no_lens_model.yaml"
MINIMAL = FIXTURE_DIR / "minimal.yaml"
INVALID = FIXTURE_DIR / "invalid_examples.yaml"
CONFIG = ROOT / "config" / "ml.yaml"


def _param_norm() -> dict:
    with CONFIG.open() as fp:
        return yaml.safe_load(fp)["data"]["param_normalization"]


def _observed_feature_config() -> dict:
    with CONFIG.open() as fp:
        return yaml.safe_load(fp)["data"]["observed_features"]


def test_load_yaml_catalog_and_build_mode1_features() -> None:
    entries = load_yaml_catalog(COMPLETE)
    assert len(entries) == 1
    spec = entries[0].to_feature_spec()
    got = build_corrector_inputs(
        spec,
        param_norm=_param_norm(),
        target_mode=1,
        observed_feature_config=_observed_feature_config(),
    )
    assert got["params"].shape[-1] == len(_param_norm()) + 5
    assert got["lc"].shape[1] == 2
    # use_image deleted in v0.5 — key must not be present
    assert "use_image" not in got, "use_image was deleted in v0.5"
    assert spec["pair_order"] == {"leading_image": "A", "trailing_image": "B"}


def test_missing_lens_features_set_masks() -> None:
    entries = load_yaml_catalog(PARTIAL)
    spec = entries[0].to_feature_spec()
    got = build_corrector_inputs(
        spec,
        param_norm=_param_norm(),
        target_mode=1,
        observed_feature_config=_observed_feature_config(),
    )
    norm_keys = list(_param_norm())
    params = got["params"][0]
    assert float(params[norm_keys.index("missing_sigma_v")]) == 0.0
    assert float(params[norm_keys.index("missing_theta_E")]) == 1.0
    assert float(params[norm_keys.index("missing_q")]) == 1.0
    assert float(params[norm_keys.index("theta_E")]) == 0.0
    assert float(params[norm_keys.index("q")]) == 0.0


def test_minimal_entry_sets_all_lens_missing_masks() -> None:
    entries = load_yaml_catalog(MINIMAL)
    spec = entries[0].to_feature_spec()
    got = build_corrector_inputs(
        spec,
        param_norm=_param_norm(),
        target_mode=1,
        observed_feature_config=_observed_feature_config(),
    )
    norm_keys = list(_param_norm())
    params = got["params"][0]
    assert float(params[norm_keys.index("missing_sigma_v")]) == 1.0
    assert float(params[norm_keys.index("missing_theta_E")]) == 1.0
    assert float(params[norm_keys.index("missing_q")]) == 1.0
    assert float(params[norm_keys.index("sigma_v")]) == 0.0


def test_dt_lc_sigma_is_required() -> None:
    entry = yaml.safe_load(INVALID.read_text(encoding="utf-8"))[1]
    with pytest.raises(ValueError, match="dt_lc_sigma"):
        entry_from_mapping(entry)


def test_truth_only_keys_rejected() -> None:
    entry = yaml.safe_load(INVALID.read_text(encoding="utf-8"))[2]
    with pytest.raises(ValueError, match="truth-only"):
        entry_from_mapping(entry)


def test_negative_delay_is_abs_and_pair_order_is_flipped() -> None:
    entry = yaml.safe_load(INVALID.read_text(encoding="utf-8"))[0]
    got = entry_from_mapping(entry)
    assert got.dt_lc == pytest.approx(12.5)
    assert got.pair_order == {"leading_image": "B", "trailing_image": "A"}
    assert got.conversion_log
    assert "abs(dt_lc)" in got.conversion_log[0]


def test_run_mode1_accepts_yaml_catalog_without_raw_light_curve() -> None:
    from pipelines.run_mode1 import run_mode1

    result = run_mode1(COMPLETE, system_index=0, apply_correction=False)
    assert result["confidence_grade"] == "external_bag22"
    assert result["dt_obs_days"] == pytest.approx(24.3)
    assert result["H0"] == pytest.approx(result["H0_approx"])
