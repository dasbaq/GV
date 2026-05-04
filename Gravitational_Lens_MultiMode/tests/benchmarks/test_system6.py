"""
벤치마크: system 6 — Mode 1 입력 Δt 오차 < 0.15일.

실측 데이터 없음 → mock 기반 SKIP 마크 적용.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from inversion.mode1_h0 import invert_h0, _d_delta_t, _h0_from_dt

SYSTEM6_DT_TRUE = 24.14   # days


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "data" / "system6.h5").exists(),
    reason="⚠️ system6.h5 미존재 — MOCK 모드 (실측 검증 대기)"
)
def test_system6_real():
    """실측 system 6 데이터로 Δt 재구성 오차 < 0.15일."""
    pass  # 실측 데이터 있을 때 구현


def test_system6_mock():
    """
    Mock: H0=70, z_lens=0.3, z_source=1.5, fermat=0.6 → Δt ≈ 24.14일 재구성.
    역산 오차 < 0.15일 확인.
    """
    import yaml
    phys = yaml.safe_load(
        open(Path(__file__).parent.parent.parent / "config" / "physics.yaml")
    )
    c, days_s, Mpc_km = phys["c_km_s"], phys["days_s"], phys["Mpc_km"]
    z_lens, z_source, H0 = 0.3, 1.5, 70.0

    # fermat potential을 Δt=24.14일을 생성하도록 역산
    ddt = _d_delta_t(H0, z_lens, z_source, approx_level=1)
    fermat = SYSTEM6_DT_TRUE * days_s * c / ((1 + z_lens) * ddt * Mpc_km)
    fermat = np.array([fermat])

    # dt_obs에 약간의 노이즈 추가
    rng = np.random.default_rng(6)
    dt_obs = np.array([SYSTEM6_DT_TRUE]) + rng.normal(0, 0.05, 1)

    result = invert_h0(dt_obs, fermat, z_lens, z_source, approx_level=1, n_bootstrap=50)

    # 역산된 H0로 Δt 재계산
    ddt_pred = _d_delta_t(result["H0"], z_lens, z_source, approx_level=1)
    dt_pred  = ((1 + z_lens) * ddt_pred * Mpc_km / c) * fermat[0] / days_s

    err = abs(dt_pred - SYSTEM6_DT_TRUE)
    assert err < 0.15, f"⚠️ system6 Δt 오차 {err:.4f}일 ≥ 0.15일"
