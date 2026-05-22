from __future__ import annotations

import h5py
import numpy as np

from ml.data.error_catalog import (
    TRUTH_DPHI_RATIO_RANGE,
    TRUTH_H0_APPROX_RANGE,
    TRUTH_MIN_IMAGE_SEPARATION_ARCSEC,
    TRUTH_MU_MAX,
    TRUTH_ROOT_RESIDUAL_ARCSEC,
    TRUTH_V0_2_DPHI_RATIO_RANGE,
    TRUTH_V0_2_DT_APPROX_MAX_DAYS,
    TRUTH_V0_2_F_JOINT_ABSMAX_MAX,
    TRUTH_V0_2_I_OBS_SUM_MAX,
    TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC,
    TRUTH_V0_2_MODE1_CORRECTION_ABSMAX,
    TRUTH_V0_2_MU_MAX,
    TRUTH_V0_3_H0_STRATIFIED_BINS,
    CatalogConfig,
    _validity_reject_reason,
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


def _validity_kwargs(**overrides):
    kwargs = {
        "dt_true": 10.0,
        "dt_approx": 10.0,
        "h0_true": 70.0,
        "h0_approx": 60.0,
        "phi_truth": 1.0e-11,
        "phi_sie": 8.0e-12,
        "mu_truth": 0.5,
        "separation_truth": 2.0,
        "dphi_ratio": 0.8,
        "i_obs_sum": 10.0,
        "f_joint_absmax": 2.0,
    }
    kwargs.update(overrides)
    return kwargs


def test_phase4_v0_2_threshold_boundaries_pass() -> None:
    assert _validity_reject_reason(**_validity_kwargs(mu_truth=TRUTH_V0_2_MU_MAX)) is None
    assert (
        _validity_reject_reason(
            **_validity_kwargs(dphi_ratio=TRUTH_V0_2_DPHI_RATIO_RANGE[0])
        )
        is None
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(dphi_ratio=TRUTH_V0_2_DPHI_RATIO_RANGE[1])
        )
        is None
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(separation_truth=TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC)
        )
        is None
    )
    assert _validity_reject_reason(**_validity_kwargs(dt_approx=TRUTH_V0_2_DT_APPROX_MAX_DAYS)) is None
    assert _validity_reject_reason(**_validity_kwargs(i_obs_sum=TRUTH_V0_2_I_OBS_SUM_MAX)) is None
    assert _validity_reject_reason(**_validity_kwargs(f_joint_absmax=TRUTH_V0_2_F_JOINT_ABSMAX_MAX)) is None
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                h0_true=TRUTH_H0_APPROX_RANGE[0] + TRUTH_V0_2_MODE1_CORRECTION_ABSMAX,
                h0_approx=TRUTH_H0_APPROX_RANGE[0],
            )
        )
        is None
    )


def test_phase4_v0_2_threshold_boundaries_reject() -> None:
    assert (
        _validity_reject_reason(**_validity_kwargs(mu_truth=TRUTH_V0_2_MU_MAX + 1.0e-4))
        == "mu_truth_gt_v0_2_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(dphi_ratio=TRUTH_V0_2_DPHI_RATIO_RANGE[0] - 1.0e-4)
        )
        == "dphi_ratio_outside_v0_2_p01_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(dphi_ratio=TRUTH_V0_2_DPHI_RATIO_RANGE[1] + 1.0e-4)
        )
        == "dphi_ratio_outside_v0_2_p01_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(separation_truth=TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC - 1.0e-4)
        )
        == "image_separation_lt_v0_2_p01"
    )
    assert (
        _validity_reject_reason(**_validity_kwargs(dt_approx=TRUTH_V0_2_DT_APPROX_MAX_DAYS + 1.0e-4))
        == "dt_approx_gt_v0_2_p99"
    )
    assert (
        _validity_reject_reason(**_validity_kwargs(i_obs_sum=TRUTH_V0_2_I_OBS_SUM_MAX + 1.0e-4))
        == "image_sum_gt_v0_2_p99"
    )
    assert (
        _validity_reject_reason(**_validity_kwargs(f_joint_absmax=TRUTH_V0_2_F_JOINT_ABSMAX_MAX + 1.0e-4))
        == "lc_absmax_gt_v0_2_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                h0_true=TRUTH_H0_APPROX_RANGE[0] + TRUTH_V0_2_MODE1_CORRECTION_ABSMAX + 1.0e-4,
                h0_approx=TRUTH_H0_APPROX_RANGE[0],
            )
        )
        == "mode1_correction_abs_gt_v0_2_p99"
    )


def test_phase4_v0_3_removes_label_dependent_gates() -> None:
    assert (
        _validity_reject_reason(
            **_validity_kwargs(h0_approx=10.0, dphi_ratio=0.1, validity_filter="v0_3")
        )
        is None
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                h0_true=120.0,
                h0_approx=10.0,
                validity_filter="v0_3",
            )
        )
        is None
    )
    assert (
        _validity_reject_reason(**_validity_kwargs(dt_true=-1.0, validity_filter="v0_3"))
        == "dt_true_nonpositive"
    )
    assert (
        _validity_reject_reason(**_validity_kwargs(mu_truth=0.981, validity_filter="v0_3"))
        == "mu_truth_ge_0p98"
    )


def test_phase4_v0_3_1_restores_only_input_tail_gates() -> None:
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                h0_true=120.0,
                h0_approx=10.0,
                validity_filter="v0_3_1",
            )
        )
        is None
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(mu_truth=TRUTH_V0_2_MU_MAX + 1.0e-4, validity_filter="v0_3_1")
        )
        == "mu_truth_gt_v0_2_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                dphi_ratio=TRUTH_V0_2_DPHI_RATIO_RANGE[1] + 1.0e-4,
                validity_filter="v0_3_1",
            )
        )
        == "dphi_ratio_outside_v0_2_p01_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                dt_approx=TRUTH_V0_2_DT_APPROX_MAX_DAYS + 1.0e-4,
                validity_filter="v0_3_1",
            )
        )
        == "dt_approx_gt_v0_2_p99"
    )
    assert (
        _validity_reject_reason(
            **_validity_kwargs(
                h0_true=TRUTH_H0_APPROX_RANGE[0] + TRUTH_V0_2_MODE1_CORRECTION_ABSMAX + 1.0e-4,
                h0_approx=TRUTH_H0_APPROX_RANGE[0],
                validity_filter="v0_3_1",
            )
        )
        is None
    )


def test_phase4_v0_3_catalog_uses_h0_stratified_quota(tmp_path) -> None:
    path = tmp_path / "phase4_v0_3_valid.h5"
    summary = build_phase4_catalog(
        path,
        CatalogConfig(
            n_systems=20,
            seed=789,
            log_path=tmp_path / "labels.json",
            reject_log_path=tmp_path / "rejects.json",
            diagnosis_log_path=tmp_path / "diagnosis.json",
            validity_filter="v0_3",
            resample_budget=80,
        ),
    )
    counts = summary["resample"]["h0_stratified_counts"]
    assert len(counts) == TRUTH_V0_3_H0_STRATIFIED_BINS
    assert sum(counts) == 20
    assert set(counts) == {2}
    with h5py.File(path, "r") as f:
        assert f["metadata"].attrs["validity_filter"] == "v0_3"
        assert bool(f["metadata"].attrs["v0_3_h0_neutral_filter"])
        assert np.all(f["true_values/dt_true"][:] > 0.0)
        assert np.all(np.abs(f["true_values/mu_true"][:]) < TRUTH_MU_MAX)


def test_phase4_v0_3_1_catalog_uses_h0_quota_and_tail_gates(tmp_path) -> None:
    path = tmp_path / "phase4_v0_3_1_valid.h5"
    summary = build_phase4_catalog(
        path,
        CatalogConfig(
            n_systems=20,
            seed=789,
            log_path=tmp_path / "labels.json",
            reject_log_path=tmp_path / "rejects.json",
            diagnosis_log_path=tmp_path / "diagnosis.json",
            validity_filter="v0_3_1",
            resample_budget=120,
        ),
    )
    counts = summary["resample"]["h0_stratified_counts"]
    assert len(counts) == TRUTH_V0_3_H0_STRATIFIED_BINS
    assert sum(counts) == 20
    assert set(counts) == {2}
    with h5py.File(path, "r") as f:
        assert f["metadata"].attrs["validity_filter"] == "v0_3_1"
        assert bool(f["metadata"].attrs["v0_3_h0_neutral_filter"])
        assert bool(f["metadata"].attrs["v0_3_1_restores_input_tail_gates"])
        assert np.all(np.abs(f["true_values/mu_true"][:]) <= TRUTH_V0_2_MU_MAX)
        ratio = f["ray_paths/dphi_sie_over_truth"][:]
        assert np.all(ratio >= TRUTH_V0_2_DPHI_RATIO_RANGE[0])
        assert np.all(ratio <= TRUTH_V0_2_DPHI_RATIO_RANGE[1])
        assert np.all(f["approx_outputs/dt_approx"][:] <= TRUTH_V0_2_DT_APPROX_MAX_DAYS)
