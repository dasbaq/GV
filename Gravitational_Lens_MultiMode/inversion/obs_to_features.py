"""관측/스펙 → MultiModalErrorCorrector 입력 텐서 어댑터 (Mode 1).

`ml/training/dataset.py::LensCorrectionDataset.__getitem__`의 Mode 1 입력 구성을
그대로 재현한다(라벨 제외). 같은 정규화·키·shape를 보장해야 학습된 corrector에
관측을 그대로 먹일 수 있다.

단위: H0 [km/s/Mpc], 각도 [arcsec], 지연 [days], z 무차원. SIE 표준 근사 가정
(단일 평면, κ_ext=0, smooth profile, isotropic). 이 모듈은 truth-side 키
(dt_true, mu_true, kappa_ext 등)를 절대 읽지 않는다.

Image 입력 모달리티는 삭제됨 (DECISIONS.md [2026-05-25] 참조).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ml.training.feature_schema import (
    build_param_vector_from_features,
    compute_light_curve_quality,
    hdf5_feature_values,
)
from ml.training.dataset import _compute_sigma_curve_fallback
from ml.utils.mask import make_lc_mask


def build_corrector_inputs(
    spec: Mapping[str, Any],
    *,
    param_norm: Mapping[str, Any],
    max_len: int = 1024,
    sigma_curve_size: int = 512,
    approx_level: int = 1,
    target_mode: int = 1,
    observed_feature_config: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """스펙 dict → batch=1 corrector 입력 텐서 dict (Mode 1).

    `dataset.__getitem__`의 Mode 1 경로와 byte 단위로 동일한 텐서를 만든다.
    필요한 scalar ``spec`` 키: ``H0_approx``, ``z_lens``, ``z_source``,
    ``dt_lc``/``dt_lc_sigma`` 또는 legacy ``dt_approx``. ``sigma_v``, ``q``,
    ``theta_E``는 누락 가능하며 missing flag가 함께 들어간다. Raw
    light-curve tensors are optional for real YAML catalogs; missing
    modalities are represented by zero tensors. approx_level은 eval 기본값 1.
    """

    n_valid = min(int(spec.get("n_epochs", spec.get("n_epochs_quality", 0)) or 0), max_len)
    if spec.get("F_joint") is None:
        F_raw = np.zeros(max(n_valid, 1), dtype=np.float32)
    else:
        F_raw = np.asarray(spec["F_joint"], dtype=np.float32).reshape(-1)
    if spec.get("sigma_noise") is None:
        sigma_noise = np.zeros(max(n_valid, 1), dtype=np.float32)
    else:
        sigma_noise = np.asarray(spec["sigma_noise"], dtype=np.float32).reshape(-1)

    lc = np.zeros((2, max_len), dtype=np.float32)
    lc[0, :n_valid] = F_raw[:n_valid]
    lc[1, :n_valid] = sigma_noise[:n_valid]
    lc_mask = make_lc_mask(n_valid, max_len)

    raw_params = dict(spec)
    if not all(key in raw_params for key in ("n_epochs_quality", "baseline_days", "median_cadence_days", "median_photometric_error")):
        raw_params.update(
            compute_light_curve_quality(
                t_obs=np.asarray(spec.get("t_obs"), dtype=np.float32) if spec.get("t_obs") is not None else None,
                sigma_noise=sigma_noise,
                n_epochs=n_valid,
            )
        )
    params_vec = build_param_vector_from_features(
        raw_params,
        dict(param_norm),
        approx_level=approx_level,
        target_mode=target_mode,
        observed_feature_config=observed_feature_config or {},
        allow_legacy_delay_sigma=True,
    )

    if spec.get("sigma_curve") is not None:
        sigma_curve = np.asarray(spec["sigma_curve"], dtype=np.float32)[:sigma_curve_size]
        if len(sigma_curve) < sigma_curve_size:
            sigma_curve = np.concatenate(
                [sigma_curve, np.zeros(sigma_curve_size - len(sigma_curve), dtype=np.float32)]
            )
    else:
        sigma_curve = _compute_sigma_curve_fallback(F_raw, n_valid, sigma_curve_size)

    return {
        "lc": torch.from_numpy(lc).unsqueeze(0),
        "lc_mask": lc_mask.unsqueeze(0),
        "params": torch.from_numpy(params_vec).unsqueeze(0),
        "sigma_curve": torch.from_numpy(sigma_curve[np.newaxis]).unsqueeze(0),
        "target_mode": torch.tensor([int(target_mode)]),
    }


def system_spec_from_hdf5(path: str | Path, sys_idx: int = 0) -> dict[str, Any]:
    """Phase 4 HDF5 한 시스템에서 dataset가 읽는 inference-side 필드를 그대로 추출.

    self-consistency 테스트 및 HDF5-관측 추론용. truth-side 키는 읽지 않는다.
    Image 필드(images/I_obs, approx_outputs/S_approx)는 삭제됨.
    """

    import h5py

    with h5py.File(str(path), "r") as f:
        features = hdf5_feature_values(f, sys_idx)
        return {
            "F_joint": np.asarray(f["light_curves/F_joint"][sys_idx], dtype=np.float32),
            "sigma_noise": np.asarray(f["light_curves/sigma_noise"][sys_idx], dtype=np.float32),
            "t_obs": np.asarray(f["light_curves/t_obs"][sys_idx], dtype=np.float32),
            "n_epochs": int(f["light_curves/n_epochs"][sys_idx]),
            **features,
            "sigma_curve": (
                np.asarray(f["sigma_curve"][sys_idx], dtype=np.float32)
                if "sigma_curve" in f
                else None
            ),
        }


def load_corrector(checkpoint: str | Path, config_path: str | Path):
    """config/ml.yaml + checkpoint로 MultiModalErrorCorrector를 로드(eval 모드).

    round build_model과 동일하게 ``param_in_dim = len(param_normalization) + 5``.
    """

    import yaml

    from ml.models.error_corrector import MultiModalErrorCorrector

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    param_norm = cfg["data"]["param_normalization"]
    model_cfg = dict(cfg["model"])
    model_cfg["param_in_dim"] = len(param_norm) + 5
    model = MultiModalErrorCorrector(model_cfg)
    state = torch.load(str(checkpoint), map_location="cpu")
    first_weight_key = "par_enc.net.0.weight"
    if first_weight_key in state:
        current = model.state_dict()[first_weight_key]
        loaded = state[first_weight_key]
        if loaded.shape != current.shape and loaded.shape[0] == current.shape[0]:
            patched = current.clone()
            n_cols = min(int(loaded.shape[1]), int(current.shape[1]))
            patched[:, :n_cols] = loaded[:, :n_cols]
            state[first_weight_key] = patched
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_target_scaler(path: str | Path) -> dict:
    with open(str(path), "rb") as f:
        return pickle.load(f)
