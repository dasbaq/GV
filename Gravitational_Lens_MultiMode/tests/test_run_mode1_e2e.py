from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from core.physics.config import constants
from core.physics.lens_models import SIELens
from inversion.mode1_h0 import _d_delta_t
from pipelines.run_mode1 import run_mode1
from tests.benchmarks._mock_data import make_system6_synthetic


def _mode1_delay_cfg(dt_true: float) -> dict:
    return {
        "grid": {
            "dt_try_range": [float(dt_true - 1.0), float(dt_true + 1.0)],
            "dt_try_step": 0.05,
            "mu_try_range": [0.1, 0.9],
            "mu_try_step": 0.05,
        },
        "reconstruction": {
            "series_truncation_tol": 1.0e-5,
            "max_series_terms": 100,
            "interpolation": "cubic",
        },
        "fluctuation": {"diff_order": 1, "detrend": False},
        "selection": {
            "conservative": {"sigma_threshold": -0.7, "require_pair": False},
            "relaxed": {"sigma_threshold": -0.5, "depth_fraction": 0.5},
        },
    }


def _write_forward_observation_h5(path: Path, H0_true: float = 70.0) -> tuple[Path, float]:
    z_lens = 0.45
    z_source = 1.8
    lens = SIELens(
        sigma_v=300.0,
        q=1.0,
        position_angle=0.0,
        z_lens=z_lens,
        z_source=z_source,
        cosmology={"H0": H0_true},
    )
    theta_e = lens.einstein_radius()
    beta = np.array([0.25 * theta_e, 0.0], dtype=float)
    theta_1 = np.array([theta_e + beta[0], 0.0], dtype=float)
    theta_2 = np.array([-theta_e + beta[0], 0.0], dtype=float)
    dphi_rad2 = abs(float(lens.fermat_potential(theta_1, beta) - lens.fermat_potential(theta_2, beta)))
    dt_true = (
        (1.0 + z_lens)
        * _d_delta_t(H0_true, z_lens, z_source, approx_level=0)
        * (constants()["Mpc_m"] / 1000.0)
        * dphi_rad2
        / constants()["c_km_s"]
        / constants()["day_s"]
    )
    lc = make_system6_synthetic(
        dt_true=dt_true,
        mu_true=0.5,
        n_epochs=220,
        cadence=1.0,
        snr=1000.0,
        seed=11,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["mock"] = True
        meta.attrs["source"] = "synthetic_mode1_e2e"
        meta.attrs["H0_true"] = H0_true
        meta.attrs["dt_true_days"] = dt_true
        params = h5.create_group("params")
        params.create_dataset("z_lens", data=np.array([z_lens], dtype=np.float32))
        params.create_dataset("z_source", data=np.array([z_source], dtype=np.float32))
        rays = h5.create_group("ray_paths")
        rays.create_dataset("theta_1", data=np.asarray([theta_1], dtype=np.float32))
        rays.create_dataset("theta_2", data=np.asarray([theta_2], dtype=np.float32))
        lc_group = h5.create_group("light_curves")
        lc_group.create_dataset("F_joint", data=np.asarray(lc["F"], dtype=np.float32)[None, :])
        lc_group.create_dataset("t_obs", data=np.asarray(lc["t_obs"], dtype=np.float32)[None, :])
        lc_group.create_dataset("sigma_noise", data=np.asarray(lc["sigma_noise"], dtype=np.float32)[None, :])
        lc_group.create_dataset("n_epochs", data=np.array([len(lc["t_obs"])], dtype=np.int32))
    return path, float(dt_true)


def test_run_mode1_e2e_recovers_forward_h0(tmp_path: Path) -> None:
    input_path, dt_true = _write_forward_observation_h5(tmp_path / "mode1_obs.h5")
    result = run_mode1(
        input_path,
        system_index=0,
        approx_level=0,
        apply_correction=True,
        delay_config=_mode1_delay_cfg(dt_true),
    )

    assert result["mock"] is True
    assert result["ml_correction"]["applied"] is False
    assert "correction skipped" in result["ml_correction"]["reason"]
    assert abs(result["dt_obs_days"] - dt_true) < 0.15
    assert abs(result["H0"] - 70.0) < 0.2
    assert result["dphi_rad2"] > 0.0
    assert result["sie_fit"]["residual_rms_arcsec"] < 1.0e-6


def test_run_mode1_cli_writes_json(tmp_path: Path) -> None:
    input_path, dt_true = _write_forward_observation_h5(tmp_path / "mode1_obs.h5")
    cfg_path = tmp_path / "delay_cfg.json"
    output_path = tmp_path / "mode1_result.json"
    cfg_path.write_text(json.dumps(_mode1_delay_cfg(dt_true)), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.run_mode1",
            "--input",
            str(input_path),
            "--system-index",
            "0",
            "--approx-level",
            "0",
            "--apply-correction",
            "--delay-config",
            str(cfg_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["mock"] is True
    assert abs(result["H0"] - 70.0) < 0.2
    assert abs(result["dt_obs_days"] - dt_true) < 0.15
