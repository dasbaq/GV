import numpy as np

from core.light_curve.time_delay import extract_time_delay
from tests.benchmarks._mock_data import make_system6_synthetic


def test_extract_time_delay_returns_grid_on_synthetic_system6():
    data = make_system6_synthetic(n_epochs=150, cadence=1.0, snr=1000.0, seed=5)
    cfg = {
        "grid": {"dt_try_range": [22.0, 26.0], "dt_try_step": 0.1, "mu_try_range": [-0.8, 0.8], "mu_try_step": 0.1},
        "reconstruction": {"series_truncation_tol": 1e-5, "max_series_terms": 60, "interpolation": "cubic"},
        "smoothing": {"method": "none"},
        "fluctuation": {"diff_order": 1, "detrend": False},
        "selection": {"conservative": {"sigma_threshold": -0.7, "require_pair": False}, "relaxed": {"sigma_threshold": -0.5}},
        "uncertainty": {"bootstrap_n": 0},
        "seed": 42,
    }
    result = extract_time_delay(data["F"], data["t_obs"], data["sigma_noise"], cfg, return_grid=True)
    assert "sigma_map" in result["grid"]
    assert result["confidence_grade"] != "rejected"
    assert abs(result["dt"] - data["dt_true"]) < 0.3
