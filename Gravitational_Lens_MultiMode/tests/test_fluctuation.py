import numpy as np

from core.light_curve.fluctuation import compute_epsilon, compute_sigma_curve, find_minima, select_best_minimum


def test_epsilon_and_sigma_normalization():
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(5, 7, 20))
    epsilon = compute_epsilon(cube)
    sigma = compute_sigma_curve(epsilon)
    assert epsilon.shape == (5, 7)
    assert abs(float(np.mean(sigma))) < 1e-12
    assert abs(float(np.std(sigma)) - 1.0) < 1e-12


def test_find_minima_requires_opposite_mu_pair():
    dt_grid = np.array([10.0, 20.0, 30.0])
    mu_grid = np.array([-0.5, 0.0, 0.5])
    sigma = np.zeros((3, 3))
    sigma[1, 0] = -3.0
    sigma[1, 2] = -2.8
    minima = find_minima(sigma, dt_grid, mu_grid, sigma_threshold=-2.0, require_pair=True)
    assert len(minima) == 2
    best = select_best_minimum(minima, {"conservative": {"sigma_threshold": -2.0, "require_pair": True}})
    assert best["dt"] == 20.0
    assert best["confidence_grade"] == "conservative"
