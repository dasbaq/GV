from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.physics.standard_approx import solve_standard_approx
from inversion.sie_fit import fit_sie_to_images


def _forward_system() -> dict:
    return {
        "H0": 70.0,
        "z_lens": 0.45,
        "z_source": 1.8,
        "sigma_v": 220.0,
        "q": 0.8,
        "position_angle": 0.0,
        "source_pos_xy": np.array([0.08, 0.03], dtype=np.float32),
        "image_grid_size": 120,
    }


def test_fit_sie_to_images_signature_and_docstring() -> None:
    sig = inspect.signature(fit_sie_to_images)
    assert not any("approximation" in name for name in sig.parameters)
    doc = inspect.getdoc(fit_sie_to_images) or ""
    assert "dphi_rad2" in doc
    assert "radian^2" in doc
    assert "SIE" in doc
    assert "표준 근사" in doc


def test_fit_sie_to_images_self_consistency_from_standard_approx() -> None:
    system = _forward_system()
    forward = solve_standard_approx(system)
    image_positions = np.stack([forward.theta_1, forward.theta_2], axis=0)

    fitted = fit_sie_to_images(
        image_positions,
        z_lens=system["z_lens"],
        z_source=system["z_source"],
        cosmology={"H0": system["H0"]},
    )

    sigma_error = abs(fitted["sigma_v"] - system["sigma_v"])
    q_error = abs(fitted["q"] - system["q"])
    dphi_error = abs(fitted["dphi_rad2"] - forward.fermat_potential)

    assert fitted["success"] is True
    assert fitted["n_images"] == 2
    assert fitted["residual_rms_arcsec"] < 1.0e-6
    assert abs(fitted["mu_fit"]) < 1.0
    assert sigma_error < 0.2
    assert q_error < 0.02
    assert dphi_error / forward.fermat_potential < 1.0e-3


def test_fit_sie_to_images_rejects_single_image() -> None:
    with pytest.raises(ValueError, match="at least two image"):
        fit_sie_to_images(np.array([[0.5, 0.1]]), 0.4, 1.5, {"H0": 70.0})


def test_fit_sie_to_images_rejects_bad_redshift_order() -> None:
    with pytest.raises(ValueError, match="z_source"):
        fit_sie_to_images(np.array([[0.5, 0.1], [-0.5, 0.0]]), 0.6, 0.62, {"H0": 70.0})
