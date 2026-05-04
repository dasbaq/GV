"""Mode 2 DM 역산 — SIS σ_v 회복 테스트."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from inversion.mode2_dm import invert_dm, _sis_einstein_radius, LENS_PARAM_DIM


def _make_sis_obs(sigma_v: float, z_lens: float, z_source: float,
                   H0: float = 70.0) -> tuple:
    """SIS 기반 더미 관측량 생성."""
    theta_E = _sis_einstein_radius(sigma_v, z_lens, z_source, H0)
    theta_obs = np.array([[theta_E * 1.5, 0.0],
                           [-theta_E * 0.8, 0.3]])
    r = np.linalg.norm(theta_obs, axis=1)
    mu_obs = np.abs(r / (r - theta_E + 1e-9))
    dt_obs = np.array([20.0])
    return theta_obs, mu_obs, dt_obs


def test_sis_sigma_v_recovery():
    """SIS σ_v=250 km/s 회복 — 상대 오차 < 15%."""
    sigma_true = 250.0
    theta_obs, mu_obs, dt_obs = _make_sis_obs(sigma_true, 0.3, 1.5)
    result = invert_dm(
        dt_obs, theta_obs, mu_obs,
        H0=70.0, z_lens=0.3, z_source=1.5,
        lens_model="SIS", approx_level=1, n_bootstrap=20,
    )
    recovered = result["dm_params"][0]
    rel_err   = abs(recovered - sigma_true) / sigma_true
    assert rel_err < 0.15, f"σ_v 상대 오차 {rel_err:.3f} ≥ 15%"


def test_output_keys():
    theta_obs = np.array([[1.0, 0.0], [-0.8, 0.2]])
    mu_obs    = np.array([3.0, 2.0])
    result    = invert_dm(np.array([25.0]), theta_obs, mu_obs,
                           H0=70.0, z_lens=0.3, z_source=1.5,
                           lens_model="SIS", approx_level=1, n_bootstrap=5)
    for key in ["dm_params", "dm_uncertainty", "lens_model", "approx_level", "param_names"]:
        assert key in result


def test_nfw_model():
    """NFW 모델도 오류 없이 실행."""
    theta_obs = np.array([[1.5, 0.2], [-1.2, 0.1]])
    mu_obs    = np.array([2.5, 1.8])
    result    = invert_dm(np.array([30.0]), theta_obs, mu_obs,
                           H0=70.0, z_lens=0.3, z_source=1.5,
                           lens_model="NFW", approx_level=1, n_bootstrap=5)
    assert result["dm_params"].shape == (2,)


def test_param_dim():
    """lens_model별 파라미터 차원 검증."""
    assert LENS_PARAM_DIM["SIS"]   == 1
    assert LENS_PARAM_DIM["NFW"]   == 2
    assert LENS_PARAM_DIM["SIE"]   == 2
    assert LENS_PARAM_DIM["POINT"] == 1
