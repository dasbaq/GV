from __future__ import annotations

import numpy as np

from core.physics.config import constants
from core.physics.standard_approx import invert_h0_from_delay_sie
from inversion.mode1_h0 import _d_delta_t, invert_h0


def _dt_from_h0(H0: float, dphi_rad2: float, z_lens: float, z_source: float) -> float:
    d_reduced = _d_delta_t(H0, z_lens, z_source, approx_level=0)
    return float(
        (1.0 + z_lens)
        * d_reduced
        * (constants()["Mpc_m"] / 1000.0)
        * dphi_rad2
        / constants()["c_km_s"]
        / constants()["day_s"]
    )


def test_mode1_exact_matches_standard_approx_closed_form() -> None:
    H0_true = 70.0
    z_lens = 0.45
    z_source = 1.8
    dphi_rad2 = 1.2e-12
    dt_obs = _dt_from_h0(H0_true, dphi_rad2, z_lens, z_source)

    exact = invert_h0(
        np.array([dt_obs]),
        np.array([dphi_rad2]),
        z_lens,
        z_source,
        approx_level=0,
        n_bootstrap=0,
    )["H0"]
    closed = invert_h0_from_delay_sie(dt_obs, dphi_rad2, z_lens, z_source)

    np.testing.assert_allclose([exact], [closed], rtol=1.0e-3)


def test_mode1_rejects_arcsec2_scale_as_unit_regression() -> None:
    H0_true = 70.0
    z_lens = 0.45
    z_source = 1.8
    dphi_rad2 = 1.2e-12
    dt_obs = _dt_from_h0(H0_true, dphi_rad2, z_lens, z_source)

    wrong_arcsec2_value = dphi_rad2 / constants()["arcsec_to_rad"] ** 2
    wrong = invert_h0(
        np.array([dt_obs]),
        np.array([wrong_arcsec2_value]),
        z_lens,
        z_source,
        approx_level=0,
        n_bootstrap=0,
    )["H0"]

    assert abs(wrong - H0_true) / H0_true > 0.1


def test_mode1_distance_levels_agree_within_turbo_bound() -> None:
    H0_true = 70.0
    z_lens = 0.45
    z_source = 1.8
    dphi_rad2 = 1.2e-12
    dt_obs = _dt_from_h0(H0_true, dphi_rad2, z_lens, z_source)

    h0_by_level = [
        invert_h0(
            np.array([dt_obs]),
            np.array([dphi_rad2]),
            z_lens,
            z_source,
            approx_level=level,
            n_bootstrap=0,
        )["H0"]
        for level in (0, 1, 2)
    ]

    assert max(abs(h0 - h0_by_level[0]) / h0_by_level[0] for h0 in h0_by_level) < 0.05
