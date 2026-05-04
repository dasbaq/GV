import numpy as np

from core.light_curve.smoothing import shafieloo_smooth


def test_shafieloo_smooth_recovers_sine():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 20, 80)
    truth = np.sin(t / 3)
    sigma = np.full_like(t, 0.03)
    y = truth + rng.normal(0, sigma)
    result = shafieloo_smooth(t, y, sigma, n_iter=12, bandwidth=1.5)
    rmse = np.sqrt(np.mean((result["f_smooth"] - truth) ** 2))
    assert rmse < 0.05


def test_silverman_bandwidth_is_clamped_positive():
    t = np.linspace(0, 10, 20)
    result = shafieloo_smooth(t, np.ones_like(t), np.ones_like(t) * 0.1, n_iter=1)
    assert 1.0 <= result["bandwidth"] <= 50.0
