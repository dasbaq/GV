from __future__ import annotations

import h5py
import numpy as np

from ml.data.error_catalog import CatalogConfig, build_phase4_catalog


def test_phase4_catalog_schema_and_correction_sign(tmp_path) -> None:
    path = tmp_path / "phase4_test.h5"
    summary = build_phase4_catalog(path, CatalogConfig(n_systems=4, seed=11))
    assert path.exists()
    assert summary["correction_sign"] == "true_minus_approx"

    with h5py.File(path, "r") as f:
        for group in ("true_values", "approx_outputs", "correction_targets", "simplification_errors"):
            assert group in f
        assert "observed_features" in f
        assert "light_curve_quality" in f
        for key in ("dt_lc", "dt_lc_sigma", "n_epochs_quality", "baseline_days", "median_cadence_days", "median_photometric_error"):
            assert key in f["observed_features"]
            assert np.all(np.isfinite(f["observed_features"][key][:]))
        assert np.all(f["observed_features/dt_lc"][:] > 0.0)
        assert np.all(f["observed_features/dt_lc_sigma"][:] > 0.0)
        assert np.all(f["observed_features/dt_lc_sigma"][:] <= 20.0)
        assert f["observed_features"].attrs["dt_lc_sigma_model"] == "relative_then_clip"
        assert f["observed_features"].attrs["dt_lc_sigma_relative_distribution"] == "log_uniform"
        corr = f["correction_targets/mode1_H0_correction"][:]
        expected = f["true_values/H0_true"][:] - f["approx_outputs/H0_approx"][:]
        np.testing.assert_allclose(corr, expected, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(f["simplification_errors/mode1_H0_error"][:], corr)
        assert f["correction_targets/mode3_source_correction"].shape == (4, 64, 64)
        np.testing.assert_allclose(f["images/pixel_scale"][:], np.full(4, 0.1), rtol=0.0, atol=1.0e-7)
        assert bool(f["metadata"].attrs["full_truth_available"])
        assert f["metadata"].attrs["correction_sign"] == "true_minus_approx"
        assert bool(f["metadata"].attrs["incompatible_with_v2_scalers_checkpoints"])


def test_phase4_off_off_sanity_is_zero(tmp_path) -> None:
    path = tmp_path / "phase4_off.h5"
    summary = build_phase4_catalog(
        path,
        CatalogConfig(n_systems=3, seed=13, include_nfw=False, include_kappa_ext=False),
    )
    sanity = summary["off_off_sanity"]
    assert sanity["abs_mean"] < 1.0e-8
    assert sanity["max_abs"] < 1.0e-8
    assert summary["variance_decomposition"]["off_off_var"] < 1.0e-16


def test_phase4_distribution_has_variance_decomposition(tmp_path) -> None:
    path = tmp_path / "phase4_var.h5"
    summary = build_phase4_catalog(path, CatalogConfig(n_systems=5, seed=17))
    variance = summary["variance_decomposition"]
    assert set(variance) == {
        "off_off_var",
        "on_off_var",
        "off_on_var",
        "on_on_var",
        "cross_term",
    }
    assert variance["off_off_var"] < 1.0e-16
    assert variance["on_on_var"] > 0.0
    assert "v2_6_baseline" in summary
