"""
ZTF 어댑터 — Mode 1 추론 전용 (라벨 없음).

ZTF 공개 데이터: https://www.ztf.caltech.edu  (다운로드 코드 금지, URL만)
형식: CSV  (lens_id, mjd, mag, magerr, filter)

has_ground_truth = False
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd


def load_ztf_lightcurve(csv_path: Union[str, Path],
                         filter_band: str = "r") -> pd.DataFrame:
    """
    Parameters
    ----------
    csv_path    : ZTF 광도곡선 CSV
    filter_band : 사용할 필터 (기본 'r')

    Returns
    -------
    DataFrame sorted by mjd: columns [mjd, mag, magerr]
    """
    df = pd.read_csv(csv_path)
    if "filter" in df.columns:
        df = df[df["filter"] == filter_band]
    required = ["mjd", "mag", "magerr"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ZTF CSV 필수 컬럼 없음: {missing}")
    return df.sort_values("mjd").reset_index(drop=True)


def ztf_to_h5(
    csv_path: Union[str, Path],
    out_path: Union[str, Path],
    lens_id: str = "unknown",
    max_lc_len: int = 1024,
    image_size: int = 128,
    mode2_max_dm_dim: int = 4,
    filter_band: str = "r",
) -> Path:
    """ZTF 광도곡선 CSV → ARCHITECTURE.md 스키마 HDF5.  라벨 없음."""
    df = load_ztf_lightcurve(csv_path, filter_band)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 1  # ZTF는 lens별 단일 시스템

    flux    = df["mag"].values.astype(np.float32)
    err     = df["magerr"].values.astype(np.float32)
    t       = (df["mjd"].values - df["mjd"].values[0]).astype(np.float32)
    ne      = min(len(flux), max_lc_len)

    F_joint    = np.zeros((1, max_lc_len), np.float32)
    sig_noise  = np.zeros((1, max_lc_len), np.float32)
    t_obs      = np.zeros((1, max_lc_len), np.float32)
    F_joint[0, :ne]    = flux[:ne]
    sig_noise[0, :ne]  = err[:ne]
    t_obs[0, :ne]      = t[:ne]

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["source"]           = f"ZTF_{lens_id}"
        meta.attrs["n_systems"]        = n
        meta.attrs["has_ground_truth"] = False
        meta.attrs["approx_level"]     = -1

        params_g = f.create_group("params")
        for key, val in [("H0", 70.0), ("z_lens", 0.3), ("z_source", 1.5),
                          ("sigma_v", 250.0), ("concentration", 7.0)]:
            params_g.create_dataset(key, data=np.full(n, val, np.float32))
        params_g.create_dataset("M200",       data=np.full(n, 1e13, np.float32))
        params_g.create_dataset("lens_model", data=np.array([b"SIS"] * n))

        lc_g = f.create_group("light_curves")
        lc_g.create_dataset("F_joint",    data=F_joint)
        lc_g.create_dataset("sigma_noise",data=sig_noise)
        lc_g.create_dataset("t_obs",      data=t_obs)
        lc_g.create_dataset("n_epochs",   data=np.array([ne], np.int32))

        img_g = f.create_group("images")
        img_g.create_dataset("I_obs",      data=np.zeros((n, image_size, image_size), np.float32))
        img_g.create_dataset("S_true",     data=np.zeros((n, image_size, image_size), np.float32))
        img_g.create_dataset("psf",        data=np.zeros((n, 11, 11), np.float32))
        img_g.create_dataset("pixel_scale",data=np.full(n, 0.05, np.float32))

        tv_g = f.create_group("true_values")
        tv_g.create_dataset("dt_true",       data=np.zeros(n, np.float32))
        tv_g.create_dataset("mu_true",       data=np.ones(n, np.float32))
        tv_g.create_dataset("theta_E",       data=np.zeros(n, np.float32))
        tv_g.create_dataset("H0_true",       data=np.zeros(n, np.float32))
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
