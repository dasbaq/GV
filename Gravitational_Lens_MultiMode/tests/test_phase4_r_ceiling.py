from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "phase4_v0_4_r_ceiling.py"
CATALOG = ROOT / "data" / "mock" / "phase4_v0_4.h5"


def _module():
    spec = importlib.util.spec_from_file_location("phase4_v0_4_r_ceiling", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_phase4_r_ceiling_leak_guard_paths_do_not_overlap():
    mod = _module()
    mod.validate_leak_guard()
    used = {path for paths in mod.USED_FEATURE_PATHS.values() for path in paths}
    assert used.isdisjoint(set(mod.FORBIDDEN_FEATURE_PATHS))


@pytest.mark.skipif(not CATALOG.exists(), reason="v0.4 catalog not present")
def test_phase4_r_ceiling_features_match_mode1_dataset_dimensions():
    mod = _module()
    cfg = mod.load_cfg(mod.DEFAULT_CONFIG)
    ids = mod.split_system_ids(CATALOG, "train", int(cfg["seed"]))[:2]
    x, dims = mod.extract_features(cfg, CATALOG, ids)
    assert x.shape == (2, dims["total"])
    assert dims["params"] == len(cfg["data"]["param_normalization"]) + 5
    assert dims["lc"] == 2 * int(cfg["data"]["max_lc_len"])
    assert dims["lc_mask"] == int(cfg["data"]["max_lc_len"])
    assert dims["sigma_curve"] == int(cfg["data"]["sigma_curve_size"])
    # image modality deleted in v0.5 — dims must NOT contain an "image" key
    assert "image" not in dims, "image modality was deleted in v0.5; dims should not contain 'image'"
    assert np.isfinite(x).all()
