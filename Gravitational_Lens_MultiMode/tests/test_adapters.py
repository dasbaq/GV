"""
데이터 어댑터 — mock CSV → HDF5 스키마 검증.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import h5py
import pytest


REQUIRED_GROUPS = [
    "metadata", "params", "light_curves", "images",
    "true_values", "ray_paths", "simplification_errors",
]
REQUIRED_TV = ["dt_true", "mu_true", "theta_E", "H0_true",
               "dm_params_true", "dm_dim", "D_delta_t"]


def _check_schema(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        for g in REQUIRED_GROUPS:
            assert g in f, f"그룹 누락: {g}"
        for k in REQUIRED_TV:
            assert k in f["true_values"], f"true_values/{k} 누락"
        assert "has_ground_truth" in f["metadata"].attrs
        n = int(f["metadata"].attrs["n_systems"])
        assert f["params/H0"].shape[0] == n
        assert f["light_curves/F_joint"].shape[0] == n


# ---- COSMOGRAIL ----
@pytest.fixture
def cosmograil_csv(tmp_path):
    df = pd.DataFrame({
        "lens_name": ["Q2237", "RXJ1131"],
        "dt_AB":     [1.5, 91.4],
        "dt_AB_err": [0.5, 1.1],
        "H0_ref":    [73.0, 71.0],
    })
    p = tmp_path / "cosmo.csv"
    df.to_csv(p, index=False)
    return p


def test_cosmograil_schema(cosmograil_csv, tmp_path):
    from ml.data_adapters.cosmograil import cosmograil_to_h5
    out = cosmograil_to_h5(cosmograil_csv, tmp_path / "cosmo.h5")
    _check_schema(out)

    with h5py.File(out, "r") as f:
        assert f["metadata"].attrs["has_ground_truth"] == True
        np.testing.assert_allclose(
            f["true_values/dt_true"][:], [1.5, 91.4], atol=1e-5
        )


def test_cosmograil_missing_col(tmp_path):
    from ml.data_adapters.cosmograil import load_cosmograil_csv
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_cosmograil_csv(p)


# ---- TDC1 ----
@pytest.fixture
def tdc1_csv(tmp_path):
    df = pd.DataFrame({
        "lens_id": [1, 2, 3],
        "dt":      [10.0, 25.0, 42.0],
        "dt_err":  [0.5, 1.0, 1.5],
    })
    p = tmp_path / "tdc1.csv"
    df.to_csv(p, index=False)
    return p


def test_tdc1_schema(tdc1_csv, tmp_path):
    from ml.data_adapters.tdc1 import tdc1_to_h5
    out = tdc1_to_h5(tdc1_csv, tmp_path / "tdc1.h5", rung=0)
    _check_schema(out)

    with h5py.File(out, "r") as f:
        assert f["metadata"].attrs["source"] == "TDC1_rung0"
        assert f["metadata"].attrs["n_systems"] == 3


# ---- ZTF ----
@pytest.fixture
def ztf_csv(tmp_path):
    n = 50
    df = pd.DataFrame({
        "mjd":    np.linspace(58000, 59000, n),
        "mag":    np.random.randn(n) + 20.0,
        "magerr": np.abs(np.random.randn(n)) * 0.05 + 0.01,
        "filter": ["r"] * n,
    })
    p = tmp_path / "ztf.csv"
    df.to_csv(p, index=False)
    return p


def test_ztf_schema(ztf_csv, tmp_path):
    from ml.data_adapters.ztf import ztf_to_h5
    out = ztf_to_h5(ztf_csv, tmp_path / "ztf.h5", lens_id="TEST")
    _check_schema(out)

    with h5py.File(out, "r") as f:
        assert f["metadata"].attrs["has_ground_truth"] == False
        assert f["metadata"].attrs["n_systems"] == 1
        ne = int(f["light_curves/n_epochs"][0])
        assert ne == 50
