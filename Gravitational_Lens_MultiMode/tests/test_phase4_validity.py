from __future__ import annotations

import h5py
import numpy as np

from ml.data.error_catalog import (
    TRUTH_DPHI_RATIO_RANGE,
    TRUTH_H0_APPROX_RANGE,
    TRUTH_MIN_IMAGE_SEPARATION_ARCSEC,
    TRUTH_MU_MAX,
    TRUTH_ROOT_RESIDUAL_ARCSEC,
    CatalogConfig,
    build_phase4_catalog,
)


def test_phase4_catalog_contains_only_valid_systems(tmp_path) -> None:
    path = tmp_path / "phase4_valid.h5"
    summary = build_phase4_catalog(
        path,
        CatalogConfig(
            n_systems=8,
            seed=123,
            log_path=tmp_path / "labels.json",
            reject_log_path=tmp_path / "rejects.json",
            diagnosis_log_path=tmp_path / "diagnosis.json",
        ),
    )

    assert summary["off_off_sanity"]["max_abs"] < 1.0e-6
    with h5py.File(path, "r") as f:
        theta_1 = f["ray_paths/theta_1"][:]
        theta_2 = f["ray_paths/theta_2"][:]
        sep = np.linalg.norm(theta_1 - theta_2, axis=1)
        assert np.all(sep >= TRUTH_MIN_IMAGE_SEPARATION_ARCSEC)
        assert np.all(np.abs(f["true_values/mu_true"][:]) < TRUTH_MU_MAX)
        assert np.all(f["true_values/dt_true"][:] > 0.0)
        assert np.all(f["approx_outputs/H0_approx"][:] >= TRUTH_H0_APPROX_RANGE[0])
        assert np.all(f["approx_outputs/H0_approx"][:] <= TRUTH_H0_APPROX_RANGE[1])
        ratio = f["ray_paths/dphi_sie_over_truth"][:]
        assert np.all(ratio >= TRUTH_DPHI_RATIO_RANGE[0])
        assert np.all(ratio <= TRUTH_DPHI_RATIO_RANGE[1])
        assert np.all(f["ray_paths/root_residual_norm_max"][:] < TRUTH_ROOT_RESIDUAL_ARCSEC)
        corr = f["correction_targets/mode1_H0_correction"][:]
        expected = f["true_values/H0_true"][:] - f["approx_outputs/H0_approx"][:]
        np.testing.assert_allclose(corr, expected, rtol=1.0e-6, atol=2.0e-6)


def test_phase4_metadata_records_sie_anchored_search(tmp_path) -> None:
    path = tmp_path / "phase4_meta.h5"
    build_phase4_catalog(
        path,
        CatalogConfig(
            n_systems=3,
            seed=456,
            log_path=tmp_path / "labels.json",
            reject_log_path=tmp_path / "rejects.json",
            diagnosis_log_path=tmp_path / "diagnosis.json",
        ),
    )
    with h5py.File(path, "r") as f:
        assert "SIE-anchored" in f["metadata"].attrs["truth_image_search_mode"]
        assert float(f["metadata"].attrs["truth_image_dedupe_arcsec"]) == 0.01
        assert f["metadata"].attrs["correction_sign"] == "true_minus_approx"
