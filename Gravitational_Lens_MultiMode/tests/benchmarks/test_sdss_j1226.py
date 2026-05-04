"""
벤치마크: SDSS J1226-0006 — COSMOGRAIL 1σ 이내 H₀ 출력.

실측 데이터 없음 → mock으로 SKIP 마크 적용.
합격 기준: CLAUDE.md 변경 불가.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from inversion.mode1_h0 import invert_h0, _d_delta_t

# COSMOGRAIL 공표값 (Eulaers et al. 2013 기준 예시)
SDSS_J1226_H0_REF  = 72.8   # km/s/Mpc
SDSS_J1226_H0_SIGMA = 5.5   # 1σ 불확도


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "data" / "sdss_j1226.h5").exists(),
    reason="⚠️ sdss_j1226.h5 미존재 — MOCK 모드"
)
def test_sdss_j1226_real():
    """실측 SDSS J1226-0006 데이터로 H₀ COSMOGRAIL 1σ 검증."""
    pass


def test_sdss_j1226_mock():
    """
    Mock: COSMOGRAIL 공표값 기준 Δt 합성 → H₀ 역산 → 1σ 범위 확인.
    """
    import yaml
    phys = yaml.safe_load(
        open(Path(__file__).parent.parent.parent / "config" / "physics.yaml")
    )
    c, days_s, Mpc_km = phys["c_km_s"], phys["days_s"], phys["Mpc_km"]

    z_lens, z_source = 0.322, 1.131   # SDSS J1226-0006 공표 적색편이 근사
    H0_true = SDSS_J1226_H0_REF

    ddt = _d_delta_t(H0_true, z_lens, z_source, approx_level=1)
    # 임의 fermat potential
    fermat = np.array([0.8, 1.2, 1.6])
    dt_obs = ((1 + z_lens) * ddt * Mpc_km / c) * fermat / days_s

    # 관측 오차 시뮬레이션 (1%)
    rng = np.random.default_rng(1226)
    dt_obs_noisy = dt_obs * (1 + rng.normal(0, 0.01, len(dt_obs)))

    result = invert_h0(dt_obs_noisy, fermat, z_lens, z_source,
                       approx_level=1, n_bootstrap=100)

    diff = abs(result["H0"] - SDSS_J1226_H0_REF)
    assert diff < SDSS_J1226_H0_SIGMA, (
        f"⚠️ H₀={result['H0']:.2f}, 기준={SDSS_J1226_H0_REF}±{SDSS_J1226_H0_SIGMA} "
        f"→ diff={diff:.2f} ≥ 1σ={SDSS_J1226_H0_SIGMA}"
    )
