import numpy as np
import pytest

from core.light_curve.reconstruction import reconstruct_f1, reconstruct_grid


def test_reconstruct_f1_self_consistency():
    t = np.linspace(0, 120, 241)
    dt, mu = 6.0, 0.45
    source = lambda x: 1.0 + 0.2 * np.sin(0.2 * x) + 0.05 * np.cos(0.07 * x)
    f1 = source(t)
    F = f1 + mu * source(t - dt)
    result = reconstruct_f1(F, t, dt, mu, series_tol=1e-10, max_terms=80)
    interior = t > 35
    rmse = np.sqrt(np.mean((result["f1_rec"][interior] - f1[interior]) ** 2))
    assert rmse / np.std(f1[interior]) < 0.01


def test_reconstruct_rejects_abs_mu_ge_one():
    with pytest.raises(AssertionError):
        reconstruct_f1(np.ones(5), np.arange(5), 1.0, 1.0)


def test_reconstruct_grid_matches_scalar():
    t = np.linspace(0, 30, 61)
    F = np.sin(t / 4) + 1.0
    dt_grid = np.array([2.0, 3.0])
    mu_grid = np.array([-0.4, 0.4])
    cfg = {"reconstruction": {"series_truncation_tol": 1e-8, "max_series_terms": 50}}
    cube = reconstruct_grid(F, t, dt_grid, mu_grid, cfg)
    scalar = reconstruct_f1(F, t, dt_grid[1], mu_grid[0], series_tol=1e-8, max_terms=50)
    np.testing.assert_allclose(cube[1, 0], scalar["f1_rec"], atol=1e-10)
