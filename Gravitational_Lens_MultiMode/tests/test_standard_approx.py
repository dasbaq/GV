from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.physics.standard_approx import solve_standard_approx


def _public_system() -> dict:
    return {
        "H0": 70.0,
        "z_lens": 0.5,
        "z_source": 2.0,
        "sigma_v": 230.0,
        "q": 0.8,
        "position_angle": 0.2,
        "source_pos_xy": np.array([0.12, 0.04], dtype=np.float32),
        "image_size": 64,
        "pixel_scale": 0.1,
    }


def test_solve_standard_approx_signature_and_docstring() -> None:
    sig = inspect.signature(solve_standard_approx)
    assert not any(
        banned in name
        for name in sig.parameters
        for banned in ("approximation", "profile", "level")
    )
    doc = inspect.getdoc(solve_standard_approx) or ""
    assert "SIE" in doc
    assert "표준 근사" in doc


def test_solve_standard_approx_outputs_are_finite_and_mu_converges() -> None:
    out = solve_standard_approx(_public_system())
    assert np.isfinite(out.dt_approx)
    assert np.isfinite(out.H0_approx)
    assert out.dt_approx > 0.0
    assert out.H0_approx == pytest.approx(70.0)
    assert abs(out.mu_approx) < 1.0
    assert out.S_approx is not None
    assert out.S_approx.shape == (64, 64)
    assert out.dm_params_approx.shape == (4,)


def test_solve_standard_approx_rejects_truth_only_keys() -> None:
    system = _public_system()
    system["kappa_ext"] = 0.05
    with pytest.raises(ValueError, match="truth-only"):
        solve_standard_approx(system)
