import numpy as np

from core.physics.lens_models import SISLens
from core.physics.ray_tracing import (
    integrate_optical_path,
    thin_lens_equation,
    thin_lens_time_delay,
    time_delay_between_paths,
    trace_ray_bundle,
    trace_ray_in_refractive_field,
    travel_time_from_path,
)


def _lens():
    return SISLens(sigma_v=220.0, z_lens=0.3, z_source=1.5, softening_radius_m=1.0e18)


def test_trace_ray_shape_and_finite_values():
    lens = _lens()
    path = trace_ray_in_refractive_field(
        np.array([1.0e20, 0.0, -1.0e20]),
        np.array([0.0, 0.0, 1.0]),
        lens,
        step_size_m=1.0e18,
        n_steps=8,
    )
    assert path.shape == (9, 3)
    assert np.all(np.isfinite(path))


def test_optical_path_and_travel_time_are_positive():
    lens = _lens()
    path = np.array([[1.0e20, 0.0, 0.0], [1.0e20, 0.0, 1.0e18], [1.0e20, 0.0, 2.0e18]])
    assert integrate_optical_path(path, lens) > 0
    assert travel_time_from_path(path, lens) > 0


def test_time_delay_between_paths_is_finite_days():
    lens = _lens()
    path_a = np.array([[1.0e20, 0.0, 0.0], [1.0e20, 0.0, 1.0e18]])
    path_b = np.array([[1.1e20, 0.0, 0.0], [1.1e20, 0.0, 1.0e18]])
    dt = time_delay_between_paths(path_a, path_b, lens)
    assert np.isfinite(dt)


def test_trace_ray_bundle_shape():
    lens = _lens()
    paths = trace_ray_bundle(
        np.array([[1.0e20, 0.0, -1.0e20], [1.1e20, 0.0, -1.0e20]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        lens,
        step_size_m=1.0e18,
        n_steps=4,
    )
    assert paths.shape == (2, 5, 3)


def test_thin_lens_helpers():
    lens = _lens()
    theta = np.array([[1.0, 0.2], [-1.0, 0.2]])
    beta = thin_lens_equation(theta, lens)
    assert beta.shape == theta.shape
    dt = thin_lens_time_delay(theta[0], theta[1], np.array([0.1, 0.0]), lens, 0.3, 1.5)
    assert np.isfinite(dt)
