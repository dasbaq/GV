from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from inversion.delay_extraction import (
    _reconstruct_grid_vectorized,
    extract_delay_from_observation,
)
from inversion.observation_io import ObservedLensSystem, ObservedLightCurves
from tests.benchmarks._mock_data import make_system6_synthetic


def _cfg() -> dict:
    return {
        "grid": {
            "dt_try_range": [23.0, 25.2],
            "dt_try_step": 0.05,
            "mu_try_range": [0.1, 0.9],
            "mu_try_step": 0.05,
        },
        "reconstruction": {
            "series_truncation_tol": 1.0e-5,
            "max_series_terms": 100,
            "interpolation": "cubic",
        },
        "fluctuation": {"diff_order": 1, "detrend": False},
        "selection": {
            "conservative": {"sigma_threshold": -0.7, "require_pair": False},
            "relaxed": {"sigma_threshold": -0.5, "depth_fraction": 0.5},
        },
    }


def _synthetic_observation() -> tuple[ObservedLensSystem, float]:
    data = make_system6_synthetic(
        dt_true=24.14,
        mu_true=0.7,
        n_epochs=180,
        cadence=1.0,
        snr=1000.0,
        seed=11,
    )
    lc = ObservedLightCurves(
        F=np.asarray(data["F"], dtype=np.float32)[None, :],
        t_obs=np.asarray(data["t_obs"], dtype=np.float32)[None, :],
        sigma_noise=np.asarray(data["sigma_noise"], dtype=np.float32)[None, :],
    )
    obs = ObservedLensSystem(
        image_positions=np.array([[0.8, 0.2], [-0.6, 0.1]], dtype=np.float32),
        light_curves=lc,
        z_lens=0.45,
        z_source=1.8,
        name="MOCK-system6-format",
    )
    return obs, float(data["dt_true"])


def test_extract_delay_from_observation_recovers_injected_mock_delay() -> None:
    obs, dt_true = _synthetic_observation()
    result = extract_delay_from_observation(obs, _cfg(), is_mock=True, return_grid=True)

    assert result["mock"] is True
    assert result["confidence_grade"] != "rejected"
    assert abs(result["dt_obs_days"] - dt_true) < 0.15
    assert np.isfinite(result["dt_uncertainty_days"])
    assert abs(result["mu"]) < 1.0
    assert result["grid"]["sigma_map"].shape == (
        result["grid"]["dt_grid"].size,
        result["grid"]["mu_grid"].size,
    )


def test_extract_delay_from_observation_rejects_nonconvergent_mu_grid() -> None:
    obs, _ = _synthetic_observation()
    bad_cfg = _cfg()
    bad_cfg["grid"] = {**bad_cfg["grid"], "mu_try_range": [0.9, 1.0]}

    with pytest.raises(ValueError, match=r"\\|mu_try\\| < 1"):
        extract_delay_from_observation(obs, bad_cfg, is_mock=True)


def test_reconstruction_grid_has_no_dt_mu_loop() -> None:
    tree = ast.parse(inspect.getsource(_reconstruct_grid_vectorized))
    loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
    assert loops == []
