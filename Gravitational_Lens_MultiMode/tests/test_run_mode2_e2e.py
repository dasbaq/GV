from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from core.physics.lens_models import SIELens
from core.physics.ray_tracing import find_images_thin_lens
from core.physics.standard_approx import _refine_sie_images
from inversion.mode2_dm import _time_delay_from_dphi
from pipelines.run_mode2 import run_mode2


def _write_mode2_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    H0 = 70.0
    z_lens = 0.45
    z_source = 1.8
    sigma_v = 280.0
    q = 0.65
    pa = 0.45
    beta = np.array([0.02, 0.03], dtype=float)
    lens = SIELens(
        sigma_v=sigma_v,
        q=q,
        position_angle=pa,
        z_lens=z_lens,
        z_source=z_source,
        cosmology={"H0": H0},
    )
    theta = _refine_sie_images(
        find_images_thin_lens(beta, lens, search_box_arcsec=3.5, grid_size=120),
        lens,
        beta,
    )
    phi = lens.fermat_potential(theta, beta)
    dt = _time_delay_from_dphi(np.array([abs(phi[0] - phi[1])]), H0, z_lens, z_source)
    truth = np.array([lens.einstein_radius(), q, pa, sigma_v], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["mock"] = True
        meta.attrs["source"] = "synthetic_mode2_e2e"
        params = h5.create_group("params")
        params.create_dataset("H0", data=np.array([H0], dtype=np.float32))
        params.create_dataset("z_lens", data=np.array([z_lens], dtype=np.float32))
        params.create_dataset("z_source", data=np.array([z_source], dtype=np.float32))
        obs = h5.create_group("observed_features")
        obs.create_dataset("dt_lc", data=dt.astype(np.float32))
        obs.create_dataset("dt_lc_sigma", data=np.array([dt[0] * 0.01], dtype=np.float32))
        rays = h5.create_group("ray_paths")
        rays.create_dataset("theta_1", data=theta[:1].astype(np.float32))
        rays.create_dataset("theta_2", data=theta[1:2].astype(np.float32))
        rays.create_dataset("theta_all", data=theta[None, :, :].astype(np.float32))
        truth_g = h5.create_group("true_values")
        truth_g.create_dataset("dm_params_true", data=truth[None, :])
    return truth, dt


def test_run_mode2_e2e_mu_free_hdf5(tmp_path: Path) -> None:
    input_path = tmp_path / "mode2_obs.h5"
    truth, _ = _write_mode2_h5(input_path)
    result = run_mode2(input_path, system_index=0, n_bootstrap=3, eval_truth=False)

    pred = np.asarray(result["dm_params"], dtype=float)
    rel = np.abs(pred[[0, 1, 3]] - truth[[0, 1, 3]]) / np.abs(truth[[0, 1, 3]])
    angle_err = abs(np.arctan2(np.sin(pred[2] - truth[2]), np.cos(pred[2] - truth[2])))
    assert result["mock"] is True
    assert "truth_eval" not in result
    assert result["input_audit"]["uses_mu_obs"] is False
    assert result["input_audit"]["uses_truth_mu_true"] is False
    assert np.all(rel < np.array([0.03, 0.06, 0.03]))
    assert angle_err < 0.03
    assert result["position_residual_rms_arcsec"] < 2.0e-3


def test_run_mode2_cli_writes_json_without_truth(tmp_path: Path) -> None:
    input_path = tmp_path / "mode2_obs.h5"
    output_path = tmp_path / "mode2_result.json"
    _write_mode2_h5(input_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.run_mode2",
            "--input",
            str(input_path),
            "--system-index",
            "0",
            "--n-bootstrap",
            "2",
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
    assert "truth_eval" not in result
    assert result["input_audit"]["uses_truth_mu_true"] is False
    assert result["q"] < 0.72


def test_run_mode2_eval_truth_is_explicit(tmp_path: Path) -> None:
    input_path = tmp_path / "mode2_obs.h5"
    _write_mode2_h5(input_path)
    result = run_mode2(input_path, n_bootstrap=2, eval_truth=True)
    assert result["summary"]["mock"] is True
    assert result["summary"]["input_audit"]["truth_dm_params_read_only_for_eval"] is True
    assert result["summary"]["input_audit"]["uses_truth_mu_true"] is False
    assert result["systems"][0]["truth_eval"]["truth_source"] == "true_values/dm_params_true"

