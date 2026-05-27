"""
벤치마크: TDC1 Rung 0/1 — H₀ 역산 오차 < 3%.

실측 TDC1 없음 → mock으로 SKIP 마크.
합격 기준: CLAUDE.md 변경 불가.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pytest
import yaml

from inversion.delay_extraction import extract_delay_from_observation
from inversion.mode1_h0 import invert_h0, _d_delta_t
from inversion.observation_io import from_hdf5


ROOT = Path(__file__).parent.parent.parent
OBS_DIR = ROOT / "data" / "observations"
TDC1_RUNG0_H5 = OBS_DIR / "tdc1_rung0_observed.h5"
TDC1_RUNG0_SIDECAR = OBS_DIR / "tdc1_rung0_sidecar.yaml"
TDC1_RUNG0_DELAY_CFG = OBS_DIR / "tdc1_rung0_delay_config.json"


def _load_mapping(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _delay_cfg() -> dict:
    if TDC1_RUNG0_DELAY_CFG.exists():
        return _load_mapping(TDC1_RUNG0_DELAY_CFG)
    sidecar = _load_mapping(TDC1_RUNG0_SIDECAR)
    dt_ref = float(sidecar["dt_ref_days"])
    return {
        "grid": {
            "dt_try_range": [max(0.1, dt_ref - 10.0), dt_ref + 10.0],
            "dt_try_step": 0.05,
            "mu_try_range": [0.05, 0.95],
            "mu_try_step": 0.02,
        },
        "reconstruction": {
            "series_truncation_tol": 1.0e-5,
            "max_series_terms": 120,
            "interpolation": "cubic",
        },
        "fluctuation": {"diff_order": 1, "detrend": False},
        "selection": {
            "conservative": {"sigma_threshold": -1.0, "require_pair": False},
            "relaxed": {"sigma_threshold": -0.7, "depth_fraction": 0.5},
        },
    }


@pytest.mark.skipif(
    not (TDC1_RUNG0_H5.exists() and TDC1_RUNG0_SIDECAR.exists()),
    reason="⚠️ tdc1_rung0_observed.h5/sidecar 미존재 — MOCK 모드"
)
def test_tdc1_rung0_real():
    """TDC1 Rung 0 real light curves: extracted Δt must match reference."""
    sidecar = _load_mapping(TDC1_RUNG0_SIDECAR)
    dt_ref = float(sidecar["dt_ref_days"])
    result = extract_delay_from_observation(
        from_hdf5(TDC1_RUNG0_H5),
        _delay_cfg(),
        is_mock=False,
        return_grid=True,
    )

    assert result["confidence_grade"] != "rejected"
    assert abs(result["mu"]) < 1.0
    assert np.isfinite(result["grid"]["sigma_map"]).all()
    assert abs(result["dt_obs_days"] - dt_ref) < 0.15


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "data" / "tdc1_rung1.h5").exists(),
    reason="⚠️ tdc1_rung1.h5 미존재 — MOCK 모드"
)
def test_tdc1_rung1_real():
    pass


@pytest.mark.parametrize("H0_true,z_lens,z_source", [
    (67.4, 0.3, 1.5),
    (73.0, 0.5, 2.0),
    (70.0, 0.4, 1.8),
])
def test_tdc1_mock_h0_error_lt3pct(H0_true, z_lens, z_source):
    """
    Mock: 알려진 H₀로 Δt 합성 → 역산 → 상대 오차 < 3%.
    합격 기준: CLAUDE.md 변경 불가.
    """
    import yaml
    phys = yaml.safe_load(
        open(Path(__file__).parent.parent.parent / "config" / "physics.yaml")
    )
    c, days_s, Mpc_km = phys["c_km_s"], phys["days_s"], phys["Mpc_km"]

    ddt = _d_delta_t(H0_true, z_lens, z_source, approx_level=1)
    fermat = np.array([0.6, 1.0, 1.4])
    dt_obs = ((1 + z_lens) * ddt * Mpc_km / c) * fermat / days_s

    result = invert_h0(dt_obs, fermat, z_lens, z_source,
                       approx_level=1, n_bootstrap=50)
    rel_err = abs(result["H0"] - H0_true) / H0_true

    assert rel_err < 0.03, (
        f"⚠️ TDC1 mock H₀ 상대 오차 {rel_err:.4f} ≥ 3%  "
        f"(H0_true={H0_true}, predicted={result['H0']:.2f})"
    )
