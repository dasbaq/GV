from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.physics.lens_models import SIELens
from core.physics.ray_tracing import find_images_thin_lens
from core.physics.standard_approx import _refine_sie_images
from inversion.mode2_dm import LENS_PARAM_DIM, PARAM_NAMES, _time_delay_from_dphi, invert_dm


def _make_sie_observation(
    sigma_v: float = 280.0,
    q: float = 0.65,
    position_angle: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_lens = 0.45
    z_source = 1.8
    H0 = 70.0
    beta = np.array([0.02, 0.03], dtype=float)
    lens = SIELens(
        sigma_v=sigma_v,
        q=q,
        position_angle=position_angle,
        z_lens=z_lens,
        z_source=z_source,
        cosmology={"H0": H0},
    )
    candidates = find_images_thin_lens(beta, lens, search_box_arcsec=3.5, grid_size=120)
    theta_obs = _refine_sie_images(candidates, lens, beta)
    assert theta_obs.shape[0] >= 4
    phi = lens.fermat_potential(theta_obs, beta)
    dt_obs = _time_delay_from_dphi(np.array([abs(phi[0] - phi[1])]), H0, z_lens, z_source)
    truth = np.array([lens.einstein_radius(), q, position_angle, sigma_v], dtype=float)
    return theta_obs, dt_obs, np.array([dt_obs[0] * 0.01], dtype=float), truth


def test_mode2_docstring_and_fixed_sie_surface() -> None:
    sig = inspect.signature(invert_dm)
    assert "approximation_profile" not in sig.parameters
    doc = inspect.getdoc(invert_dm) or ""
    assert "SIE" in doc
    assert "표준 근사" in doc
    assert LENS_PARAM_DIM == {"SIE": 4}
    assert PARAM_NAMES == ["theta_E", "q", "position_angle", "sigma_v"]


def test_sie_theta_dt_recovery_mu_free() -> None:
    theta_obs, dt_obs, dt_sigma, truth = _make_sie_observation()
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
        dt_sigma=dt_sigma,
    )

    got = result["dm_params"]
    rel = np.abs(got[[0, 1, 3]] - truth[[0, 1, 3]]) / np.abs(truth[[0, 1, 3]])
    angle_err = abs(np.arctan2(np.sin(got[2] - truth[2]), np.cos(got[2] - truth[2])))
    assert np.all(rel < np.array([0.02, 0.05, 0.02]))
    assert angle_err < 0.02
    assert result["position_residual_rms_arcsec"] < 2.0e-3
    assert abs(result["dt_model_days"] - float(dt_obs[0])) < 1.0e-4
    assert result["input_audit"]["uses_mu_obs"] is False
    assert result["input_audit"]["uses_truth_mu_true"] is False


def test_rejects_non_sie_placeholder_models() -> None:
    theta_obs, dt_obs, dt_sigma, _ = _make_sie_observation()
    with pytest.raises(ValueError, match="only the fixed SIE"):
        invert_dm(
            dt_obs,
            theta_obs,
            H0=70.0,
            z_lens=0.45,
            z_source=1.8,
            lens_model="NFW",
            dt_sigma=dt_sigma,
            n_bootstrap=0,
        )


def test_rejects_approx_level_switch() -> None:
    theta_obs, dt_obs, dt_sigma, _ = _make_sie_observation()
    with pytest.raises(ValueError, match="approx_level=0"):
        invert_dm(
            dt_obs,
            theta_obs,
            H0=70.0,
            z_lens=0.45,
            z_source=1.8,
            lens_model="SIE",
            approx_level=1,
            dt_sigma=dt_sigma,
            n_bootstrap=0,
        )

