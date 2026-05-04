from __future__ import annotations

import numpy as np

from core.physics.lens_models import NFWLens, SIELens
from core.physics.standard_approx import solve_standard_approx
from ml.data.error_catalog import (
    DeflectionAdditiveTruthLens,
    TRUTH_IMAGE_DEDUPE_ARCSEC,
    TRUTH_ROOT_RESIDUAL_ARCSEC,
    solve_truth_images_from_sie,
)


def test_truth_root_solver_converges_and_moves_sie_images() -> None:
    public = {
        "H0": 70.0,
        "z_lens": 0.5,
        "z_source": 2.0,
        "sigma_v": 260.0,
        "q": 0.8,
        "position_angle": 0.2,
        "source_pos_xy": np.array([0.1, 0.05], dtype=np.float32),
        "image_size": 64,
        "pixel_scale": 0.1,
    }
    approx = solve_standard_approx(public)
    sie_images = np.stack([approx.theta_1, approx.theta_2])
    sie = SIELens(
        sigma_v=260.0,
        q=0.8,
        position_angle=0.2,
        z_lens=0.5,
        z_source=2.0,
        cosmology={"H0": 70.0},
    )
    nfw = NFWLens(M200=1.0e14, concentration=12.0, z_lens=0.5, z_source=2.0, cosmology={"H0": 70.0})
    truth = DeflectionAdditiveTruthLens(sie, nfw, kappa_ext=0.08)

    result = solve_truth_images_from_sie(sie_images, truth, public["source_pos_xy"])

    assert result["success"]
    assert result["theta"].shape[0] >= 2
    assert result["residual_norm_max"] < TRUTH_ROOT_RESIDUAL_ARCSEC
    shift = np.linalg.norm(result["theta"][:2] - sie_images[:2], axis=1)
    assert float(np.max(shift)) > TRUTH_IMAGE_DEDUPE_ARCSEC


def test_truth_solver_dedupes_duplicate_roots() -> None:
    public = {
        "H0": 70.0,
        "z_lens": 0.5,
        "z_source": 2.0,
        "sigma_v": 230.0,
        "q": 0.85,
        "position_angle": 0.1,
        "source_pos_xy": np.array([0.12, 0.02], dtype=np.float32),
    }
    approx = solve_standard_approx(public)
    duplicated = np.stack([approx.theta_1, approx.theta_1 + np.array([1.0e-4, 0.0])])
    sie = SIELens(
        sigma_v=230.0,
        q=0.85,
        position_angle=0.1,
        z_lens=0.5,
        z_source=2.0,
        cosmology={"H0": 70.0},
    )
    truth = DeflectionAdditiveTruthLens(sie, None, kappa_ext=0.0)

    result = solve_truth_images_from_sie(duplicated, truth, public["source_pos_xy"])

    assert not result["success"]
    assert result["reason"] == "dedupe_lt2"
    assert result["dedupe_count"] == 1
