"""Mode 1 H₀ 역산 known-answer 테스트."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from inversion.mode1_h0 import invert_h0, _d_delta_t, _h0_from_dt


def test_round_trip_h0():
    """H₀=70 → Δt 합성 → 역산 → H₀ 복원 (오차 < 1 km/s/Mpc)."""
    H0_true   = 70.0
    z_lens    = 0.3
    z_source  = 1.5
    fermat    = np.array([0.5, 1.0, 1.5])

    import yaml
    phys = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "physics.yaml"))
    c, days_s, Mpc_km = phys["c_km_s"], phys["days_s"], phys["Mpc_km"]

    ddt = _d_delta_t(H0_true, z_lens, z_source, approx_level=1)
    dt_obs = ((1 + z_lens) * ddt * Mpc_km / c) * fermat / days_s

    result = invert_h0(dt_obs, fermat, z_lens, z_source, approx_level=1, n_bootstrap=20)

    assert abs(result["H0"] - H0_true) < 1.0, f"H0 오차 {result['H0']-H0_true:.3f}"
    assert result["H0_uncertainty"] >= 0.0
    assert result["approx_level"] == 1
    assert result["n_pairs"] == 3


def test_h0_output_range():
    """H₀ 결과가 [50, 90] 범위 내."""
    dt   = np.array([20.0])
    phi  = np.array([0.8])
    res  = invert_h0(dt, phi, 0.5, 2.0, approx_level=1, n_bootstrap=10)
    assert 50.0 <= res["H0"] <= 90.0


def test_approx_level2():
    """approx_level=2에서도 정상 실행."""
    dt  = np.array([30.0, 45.0])
    phi = np.array([1.0, 1.5])
    res = invert_h0(dt, phi, 0.4, 1.8, approx_level=2, n_bootstrap=10)
    assert "H0" in res
    assert np.isfinite(res["H0"])


def test_z_assertion():
    """z_source <= z_lens+0.05 이면 AssertionError."""
    with pytest.raises(AssertionError):
        invert_h0(np.array([10.0]), np.array([1.0]), 0.5, 0.5, approx_level=1)
