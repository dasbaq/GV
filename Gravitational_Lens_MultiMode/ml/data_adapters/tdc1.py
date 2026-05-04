"""
TDC1 어댑터 — Mode 1 검증용.

TDC1 데이터셋: http://timedelaychallenge.org  (다운로드 코드 금지, URL만)
형식: rung별 CSV  (lens_id, dt, dt_err, ...)

has_ground_truth = True (Rung 0/1 검증 가능)
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd


def load_tdc1_csv(csv_path: Union[str, Path],
                  rung: int = 0) -> pd.DataFrame:
    """
    Parameters
    ----------
    csv_path : TDC1 CSV 경로
    rung     : 0 또는 1

    Returns
    -------
    DataFrame with columns: lens_id, dt, dt_err (최소)
    """
    df = pd.read_csv(csv_path)
    required = ["lens_id", "dt", "dt_err"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"TDC1 CSV 필수 컬럼 없음: {missing}")
    df["rung"] = rung
    return df


def tdc1_to_h5(
    csv_path: Union[str, Path],
    out_path: Union[str, Path],
    rung: int = 0,
    image_size: int = 128,
    max_lc_len: int = 1024,
    mode2_max_dm_dim: int = 4,
) -> Path:
    """TDC1 CSV → ARCHITECTURE.md 스키마 HDF5."""
    df = load_tdc1_csv(csv_path, rung)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(df)

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["source"]           = f"TDC1_rung{rung}"
        meta.attrs["n_systems"]        = n
        meta.attrs["has_ground_truth"] = True
        meta.attrs["approx_level"]     = -1

        params_g = f.create_group("params")
        for key, val in [("H0", 70.0), ("z_lens", 0.3), ("z_source", 1.5),
                          ("sigma_v", 250.0), ("concentration", 7.0)]:
            params_g.create_dataset(key, data=np.full(n, val, np.float32))
        params_g.create_dataset("M200", data=np.full(n, 1e13, np.float32))
        params_g.create_dataset("lens_model", data=np.array([b"SIS"] * n))

        lc_g = f.create_group("light_curves")
        lc_g.create_dataset("F_joint",    data=np.zeros((n, max_lc_len), np.float32))
        lc_g.create_dataset("sigma_noise",data=np.zeros((n, max_lc_len), np.float32))
        lc_g.create_dataset("t_obs",      data=np.zeros((n, max_lc_len), np.float32))
        lc_g.create_dataset("n_epochs",   data=np.zeros(n, np.int32))

        img_g = f.create_group("images")
        img_g.create_dataset("I_obs",      data=np.zeros((n, image_size, image_size), np.float32))
        img_g.create_dataset("S_true",     data=np.zeros((n, image_size, image_size), np.float32))
        img_g.create_dataset("psf",        data=np.zeros((n, 11, 11), np.float32))
        img_g.create_dataset("pixel_scale",data=np.full(n, 0.05, np.float32))

        tv_g = f.create_group("true_values")
        tv_g.create_dataset("dt_true",       data=df["dt"].values.astype(np.float32))
        tv_g.create_dataset("mu_true",       data=np.ones(n, np.float32))
        tv_g.create_dataset("theta_E",       data=np.zeros(n, np.float32))
        tv_g.create_dataset("H0_true",       data=np.full(n, 70.0, np.float32))
        tv_g.create_dataset("dm_params_true",data=np.zeros((n, mode2_max_dm_dim), np.float32))
        tv_g.create_dataset("dm_dim",        data=np.ones(n, np.int32))
        tv_g.create_dataset("D_delta_t",     data=np.zeros(n, np.float32))

        rp_g = f.create_group("ray_paths")
        rp_g.create_dataset("theta_1",         data=np.zeros((n, 2), np.float32))
        rp_g.create_dataset("theta_2",         data=np.zeros((n, 2), np.float32))
        rp_g.create_dataset("fermat_potential", data=np.zeros(n, np.float32))

        se_g = f.create_group("simplification_errors")
        se_g.create_dataset("mode1_H0_error",       data=np.zeros(n, np.float32))
        se_g.create_dataset("mode2_dm_error",       data=np.zeros((n, mode2_max_dm_dim), np.float32))
        se_g.create_dataset("mode3_source_residual",data=np.zeros((n, image_size, image_size), np.float32))
        se_g.create_dataset("approx_level_used",    data=np.full(n, -1, np.int32))

    return out_path
