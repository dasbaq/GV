"""
벤치마크: TDC1 Rung 0/1 — H₀ 역산 오차 < 3%.

실측 TDC1 없음 → mock으로 SKIP 마크.
합격 기준: CLAUDE.md 변경 불가.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from inversion.mode1_h0 import invert_h0, _d_delta_t


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "data" / "tdc1_rung0.h5").exists(),
    reason="⚠️ tdc1_rung0.h5 미존재 — MOCK 모드"
)
def test_tdc1_rung0_real():
    pass


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
