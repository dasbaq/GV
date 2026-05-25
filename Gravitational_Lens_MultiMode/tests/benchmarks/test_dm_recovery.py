"""Benchmark smoke: Mode 2 SIE DM recovery on synthetic MOCK labels."""

from __future__ import annotations

import numpy as np

from core.physics.lens_models import SIELens
from core.physics.ray_tracing import find_images_thin_lens
from core.physics.standard_approx import _refine_sie_images
from inversion.mode2_dm import _time_delay_from_dphi, invert_dm


def _synthetic_quad(sigma_v: float, q: float, position_angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_lens = 0.45
    z_source = 1.8
    beta = np.array([0.02, 0.03], dtype=float)
    lens = SIELens(
        sigma_v=sigma_v,
        q=q,
        position_angle=position_angle,
        z_lens=z_lens,
        z_source=z_source,
        cosmology={"H0": 70.0},
    )
    theta = _refine_sie_images(
        find_images_thin_lens(beta, lens, search_box_arcsec=3.5, grid_size=120),
        lens,
        beta,
    )
    phi = lens.fermat_potential(theta, beta)
    dt = _time_delay_from_dphi(np.array([abs(phi[0] - phi[1])]), 70.0, z_lens, z_source)
    truth = np.array([lens.einstein_radius(), q, position_angle, sigma_v], dtype=float)
    return theta, dt, truth


def test_dm_recovery_sie_mock_theta_dt_only() -> None:
    """SIE parameters recover from θ_i + Δt without μ input on MOCK data."""

    theta_obs, dt_obs, truth = _synthetic_quad(280.0, 0.65, 0.45)
    result = invert_dm(
        dt_obs,
        theta_obs,
        mu_obs=None,
        H0=70.0,
        z_lens=0.45,
        z_source=1.8,
        lens_model="SIE",
        approx_level=0,
        n_bootstrap=3,
        dt_sigma=np.array([dt_obs[0] * 0.01], dtype=float),
    )

    pred = np.asarray(result["dm_params"], dtype=float)
    rel = np.abs(pred[[0, 1, 3]] - truth[[0, 1, 3]]) / np.abs(truth[[0, 1, 3]])
    angle_err = abs(np.arctan2(np.sin(pred[2] - truth[2]), np.cos(pred[2] - truth[2])))
    assert np.all(rel < np.array([0.03, 0.06, 0.03]))
    assert angle_err < 0.03
    assert result["input_audit"]["uses_truth_mu_true"] is False

