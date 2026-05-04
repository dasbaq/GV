"""
벤치마크: DM 회복 정확도 (Mode 2) — mock 라벨 기준 상대 오차 < 10%.

합격 기준: CLAUDE.md 변경 불가.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from inversion.mode2_dm import invert_dm, _sis_einstein_radius


@pytest.mark.parametrize("sigma_v_true,z_lens,z_source", [
    (200.0, 0.3, 1.5),
    (280.0, 0.5, 2.0),
    (320.0, 0.4, 1.8),
])
def test_dm_recovery_sis_mock(sigma_v_true, z_lens, z_source):
    """
    SIS 렌즈 σ_v 회복 — 상대 오차 < 10%.
    합격 기준: CLAUDE.md 변경 불가.
    """
    theta_E = _sis_einstein_radius(sigma_v_true, z_lens, z_source, H0=70.0)

    # 두 상 (SIS: 두 상이 아인슈타인 반경 양쪽)
    theta_obs = np.array([
        [theta_E * 1.6, 0.0],
        [-theta_E * 0.85, 0.2],
    ])
    r = np.linalg.norm(theta_obs, axis=1)
    mu_obs = np.abs(r / (r - theta_E + 1e-9))
    dt_obs = np.array([15.0])

    result = invert_dm(
        dt_obs, theta_obs, mu_obs,
        H0=70.0, z_lens=z_lens, z_source=z_source,
        lens_model="SIS", approx_level=1, n_bootstrap=30,
    )

    recovered  = result["dm_params"][0]
    rel_err    = abs(recovered - sigma_v_true) / sigma_v_true

    assert rel_err < 0.10, (
        f"⚠️ DM 회복 상대 오차 {rel_err:.3f} ≥ 10%  "
        f"(σ_v_true={sigma_v_true:.1f}, predicted={recovered:.1f})"
    )


def test_dm_recovery_nfw_mock():
    """NFW 파라미터 회복 — 최적화 수렴 및 범위 확인."""
    theta_obs = np.array([[1.2, 0.3], [-0.9, -0.2]])
    mu_obs    = np.array([2.5, 1.8])
    dt_obs    = np.array([20.0])

    result = invert_dm(
        dt_obs, theta_obs, mu_obs,
        H0=70.0, z_lens=0.3, z_source=1.5,
        lens_model="NFW", approx_level=1, n_bootstrap=20,
    )

    log10_M200, c_nfw = result["dm_params"]
    assert 12.0 <= log10_M200 <= 14.0, f"log10_M200={log10_M200:.2f} 범위 초과"
    assert 3.0  <= c_nfw      <= 15.0, f"c_nfw={c_nfw:.2f} 범위 초과"
