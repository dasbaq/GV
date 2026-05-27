from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from inversion.delay_extraction import extract_delay_from_observation
from inversion.observation_io import from_hdf5
from ml.data_adapters.observed_mode1 import (
    build_light_curve_arrays,
    ingest_observed_mode1,
    magnitude_to_flux,
    read_light_curve_table,
)
from pipelines.run_mode1 import run_mode1
from tests.benchmarks._mock_data import make_system6_synthetic


def _manifest(tmp_path: Path, columns: dict, *, z_lens: float = 0.45, z_source: float = 1.8) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "synthetic-observed-mode1",
                "source": "unit-test",
                "z_lens": z_lens,
                "z_source": z_source,
                "image_positions": [[0.8, 0.0], [-0.5, 0.0]],
                "columns": columns,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _delay_cfg(dt_true: float) -> dict:
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


def _write_flux_light_curve(path: Path, dt_true: float = 24.14) -> tuple[Path, float]:
    data = make_system6_synthetic(
        dt_true=dt_true,
        mu_true=0.7,
        n_epochs=180,
        cadence=1.0,
        snr=1000.0,
        seed=11,
    )
    sigma = np.asarray(data["sigma_noise"], dtype=float)
    df = pd.DataFrame(
        {
            "mjd": np.asarray(data["t_obs"], dtype=float),
            "A_flux": np.asarray(data["F"], dtype=float),
            "A_flux_err": sigma,
            "B_flux": np.zeros_like(sigma),
            "B_flux_err": np.full_like(sigma, float(np.median(sigma))),
        }
    )
    if path.suffix.lower() == ".rdb":
        path.write_text(
            " ".join(df.columns) + "\n"
            + " ".join("-" * len(col) for col in df.columns) + "\n"
            + "\n".join(" ".join(str(value) for value in row) for row in df.to_numpy())
            + "\n",
            encoding="utf-8",
        )
    else:
        df.to_csv(path, index=False)
    return path, float(data["dt_true"])


def test_magnitude_to_flux_propagates_magerr() -> None:
    mag = np.array([20.0, 21.0])
    mag_err = np.array([0.01, 0.02])
    flux, sigma = magnitude_to_flux(mag, mag_err)

    np.testing.assert_allclose(flux, 10.0 ** (-0.4 * mag))
    np.testing.assert_allclose(sigma, 0.4 * np.log(10.0) * flux * mag_err)
    assert np.all(sigma > 0.0)


def test_read_light_curve_table_accepts_rdb(tmp_path: Path) -> None:
    path = tmp_path / "lc.rdb"
    path.write_text(
        "mjd A_mag A_mag_err B_mag B_mag_err\n"
        "--- ----- --------- ----- ---------\n"
        "0.0 20.0 0.01 20.3 0.02\n"
        "1.0 20.1 0.01 20.4 0.02\n",
        encoding="utf-8",
    )

    df = read_light_curve_table(path)
    assert list(df.columns) == ["mjd", "A_mag", "A_mag_err", "B_mag", "B_mag_err"]
    assert len(df) == 2


def test_build_light_curve_arrays_rejects_time_grid_mismatch(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "t_A": [0.0, 1.0, 2.0],
            "t_B": [0.0, 1.0, 2.1],
            "A_flux": [1.0, 1.1, 1.2],
            "A_flux_err": [0.01, 0.01, 0.01],
            "B_flux": [0.5, 0.4, 0.3],
            "B_flux_err": [0.01, 0.01, 0.01],
        }
    )
    manifest = {
        "columns": {
            "time": "t_A",
            "series": [
                {"name": "A", "unit": "flux", "time": "t_A", "flux": "A_flux", "flux_err": "A_flux_err"},
                {"name": "B", "unit": "flux", "time": "t_B", "flux": "B_flux", "flux_err": "B_flux_err"},
            ],
        }
    }

    with pytest.raises(ValueError, match="same t_obs grid"):
        build_light_curve_arrays(df, manifest)


def test_ingest_observed_mode1_requires_manifest_fields(tmp_path: Path) -> None:
    lc, _ = _write_flux_light_curve(tmp_path / "lc.csv")
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(yaml.safe_dump({"z_lens": 0.3}), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest missing required fields"):
        ingest_observed_mode1(lc, manifest, tmp_path / "out.h5")


def test_ingest_observed_mode1_writes_observation_hdf5(tmp_path: Path) -> None:
    lc, _ = _write_flux_light_curve(tmp_path / "lc.csv")
    manifest = _manifest(
        tmp_path,
        {
            "time": "mjd",
            "series": [
                {"name": "A", "unit": "flux", "flux": "A_flux", "flux_err": "A_flux_err"},
                {"name": "B", "unit": "flux", "flux": "B_flux", "flux_err": "B_flux_err"},
            ],
        },
    )
    out = ingest_observed_mode1(lc, manifest, tmp_path / "observed.h5")

    obs = from_hdf5(out)
    assert obs.light_curves.F.shape[0] == 2
    assert obs.image_positions.shape == (2, 2)
    with h5py.File(out, "r") as h5:
        assert "true_values" not in h5
        assert h5["metadata"].attrs["has_ground_truth"] == False
        assert h5["metadata"].attrs["mock"] == False


def test_ingest_observation_cli_writes_report_without_hdf5_leak(tmp_path: Path) -> None:
    lc, _ = _write_flux_light_curve(tmp_path / "lc.csv")
    manifest = _manifest(
        tmp_path,
        {
            "time": "mjd",
            "series": [
                {"name": "A", "unit": "flux", "flux": "A_flux", "flux_err": "A_flux_err"},
                {"name": "B", "unit": "flux", "flux": "B_flux", "flux_err": "B_flux_err"},
            ],
        },
    )
    sidecar = tmp_path / "refs.yaml"
    sidecar.write_text(yaml.safe_dump({"dt_ref_days": 24.14, "H0_ref": 72.8}), encoding="utf-8")
    out = tmp_path / "observed.h5"
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.ingest_observation",
            "--light-curves",
            str(lc),
            "--manifest",
            str(manifest),
            "--sidecar",
            str(sidecar),
            "--output",
            str(out),
            "--report-output",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["references"]["dt_ref_days"] == 24.14
    with h5py.File(out, "r") as h5:
        assert "dt_ref_days" not in h5["metadata"].attrs
        assert "H0_ref" not in h5["metadata"].attrs


def test_tdc1_style_ingested_delay_matches_reference(tmp_path: Path) -> None:
    lc, dt_true = _write_flux_light_curve(tmp_path / "tdc1_rung0.csv")
    manifest = _manifest(
        tmp_path,
        {
            "time": "mjd",
            "series": [
                {"name": "A", "unit": "flux", "flux": "A_flux", "flux_err": "A_flux_err"},
                {"name": "B", "unit": "flux", "flux": "B_flux", "flux_err": "B_flux_err"},
            ],
        },
    )
    out = ingest_observed_mode1(lc, manifest, tmp_path / "tdc1_rung0_observed.h5")

    result = extract_delay_from_observation(from_hdf5(out), _delay_cfg(dt_true), return_grid=True)

    assert result["confidence_grade"] != "rejected"
    assert abs(result["dt_obs_days"] - dt_true) < 0.15
    assert abs(result["mu"]) < 1.0
    assert np.isfinite(result["grid"]["sigma_map"]).all()


def test_sdss_style_ingested_mode1_e2e_returns_finite_h0(tmp_path: Path) -> None:
    lc, dt_true = _write_flux_light_curve(tmp_path / "sdss_j1226.rdb")
    manifest = _manifest(
        tmp_path,
        {
            "time": "mjd",
            "series": [
                {"name": "A", "unit": "flux", "flux": "A_flux", "flux_err": "A_flux_err"},
                {"name": "B", "unit": "flux", "flux": "B_flux", "flux_err": "B_flux_err"},
            ],
        },
        z_lens=0.322,
        z_source=1.131,
    )
    out = ingest_observed_mode1(lc, manifest, tmp_path / "sdss_j1226_observed.h5")

    result = run_mode1(out, approx_level=0, delay_config=_delay_cfg(dt_true))

    assert result["mock"] is False
    assert np.isfinite(result["H0"])
    assert result["dphi_rad2"] > 0.0
    assert result["confidence_grade"] != "rejected"
