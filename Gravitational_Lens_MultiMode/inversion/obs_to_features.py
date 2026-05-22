"""관측/스펙 → MultiModalErrorCorrector 입력 텐서 어댑터 (Mode 1).

`ml/training/dataset.py::LensCorrectionDataset.__getitem__`의 Mode 1 입력 구성을
그대로 재현한다(라벨 제외). 같은 정규화·키·shape를 보장해야 학습된 corrector에
관측을 그대로 먹일 수 있다.

단위: H0 [km/s/Mpc], 각도 [arcsec], 지연 [days], z 무차원. SIE 표준 근사 가정
(단일 평면, κ_ext=0, smooth profile, isotropic). 이 모듈은 truth-side 키
(dt_true, mu_true, kappa_ext 등)를 절대 읽지 않는다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ml.training.dataset import _compute_sigma_curve_fallback
from ml.utils.mask import make_lc_mask
from ml.utils.normalize import build_param_vector

_DT_LC_REL_SIGMA_EXPECTED = 0.045  # dataset.py와 동일


def _resize_image(arr: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == image_size and arr.shape[1] == image_size:
        return arr
    from skimage.transform import resize

    return resize(
        arr, (image_size, image_size), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)


def build_corrector_inputs(
    spec: Mapping[str, Any],
    *,
    param_norm: Mapping[str, Any],
    image_size: int = 128,
    max_len: int = 1024,
    sigma_curve_size: int = 512,
    approx_level: int = 1,
    target_mode: int = 1,
) -> dict[str, torch.Tensor]:
    """스펙 dict → batch=1 corrector 입력 텐서 dict (Mode 1).

    `dataset.__getitem__`의 Mode 1 경로와 byte 단위로 동일한 텐서를 만든다.
    필요한 ``spec`` 키: ``F_joint``[max_epochs], ``sigma_noise``[max_epochs],
    ``n_epochs``, ``H0_approx``, ``z_lens``, ``z_source``, ``sigma_v``, ``q``,
    ``theta_E``, ``dt_approx``, ``I_obs``[H,W], ``S_approx``[H,W]. 선택:
    ``sigma_curve``[S] (없으면 F_joint 기반 fallback). approx_level은 eval 기본값 1.
    """

    F_raw = np.asarray(spec["F_joint"], dtype=np.float32)
    sigma_noise = np.asarray(spec["sigma_noise"], dtype=np.float32)
    n_valid = min(int(spec["n_epochs"]), max_len)

    lc = np.zeros((2, max_len), dtype=np.float32)
    lc[0, :n_valid] = F_raw[:n_valid]
    lc[1, :n_valid] = sigma_noise[:n_valid]
    lc_mask = make_lc_mask(n_valid, max_len)

    dt_lc = float(spec["dt_approx"])
    dt_lc_sigma = max(dt_lc * _DT_LC_REL_SIGMA_EXPECTED, 1.0e-6)
    raw_params = {
        "H0_approx": float(spec["H0_approx"]),
        "z_lens": float(spec["z_lens"]),
        "z_source": float(spec["z_source"]),
        "sigma_v": float(spec["sigma_v"]),
        "q": float(spec["q"]),
        "theta_E": float(spec["theta_E"]),
        "dt_lc": dt_lc,
        "dt_lc_sigma": dt_lc_sigma,
    }
    param_base = build_param_vector(raw_params, dict(param_norm))
    al_onehot = np.array(
        [float(approx_level == 1), float(approx_level == 2)], dtype=np.float32
    )
    mode_oh = np.zeros(3, dtype=np.float32)
    mode_oh[target_mode - 1] = 1.0
    params_vec = np.concatenate([param_base, al_onehot, mode_oh])

    if spec.get("sigma_curve") is not None:
        sigma_curve = np.asarray(spec["sigma_curve"], dtype=np.float32)[:sigma_curve_size]
        if len(sigma_curve) < sigma_curve_size:
            sigma_curve = np.concatenate(
                [sigma_curve, np.zeros(sigma_curve_size - len(sigma_curve), dtype=np.float32)]
            )
    else:
        sigma_curve = _compute_sigma_curve_fallback(F_raw, n_valid, sigma_curve_size)

    use_image = bool(int(target_mode) in (1, 3))
    if use_image:
        img_raw = _resize_image(spec["I_obs"], image_size)
        approx_raw = _resize_image(spec["S_approx"], image_size)
        image = np.stack([img_raw, img_raw - approx_raw], axis=0)
    else:
        image = np.zeros((2, image_size, image_size), dtype=np.float32)

    return {
        "lc": torch.from_numpy(lc).unsqueeze(0),
        "lc_mask": lc_mask.unsqueeze(0),
        "params": torch.from_numpy(params_vec).unsqueeze(0),
        "sigma_curve": torch.from_numpy(sigma_curve[np.newaxis]).unsqueeze(0),
        "image": torch.from_numpy(image).unsqueeze(0),
        "use_image": torch.tensor([use_image]),
        "target_mode": torch.tensor([int(target_mode)]),
    }


def system_spec_from_hdf5(path: str | Path, sys_idx: int = 0) -> dict[str, Any]:
    """Phase 4 HDF5 한 시스템에서 dataset가 읽는 inference-side 필드를 그대로 추출.

    self-consistency 테스트 및 HDF5-관측 추론용. truth-side 키는 읽지 않는다.
    """

    import h5py

    with h5py.File(str(path), "r") as f:
        return {
            "F_joint": np.asarray(f["light_curves/F_joint"][sys_idx], dtype=np.float32),
            "sigma_noise": np.asarray(f["light_curves/sigma_noise"][sys_idx], dtype=np.float32),
            "n_epochs": int(f["light_curves/n_epochs"][sys_idx]),
            "H0_approx": float(f["approx_outputs/H0_approx"][sys_idx]),
            "z_lens": float(f["params/z_lens"][sys_idx]),
            "z_source": float(f["params/z_source"][sys_idx]),
            "sigma_v": float(f["params/sigma_v"][sys_idx]),
            "q": float(f["params/q"][sys_idx]),
            "theta_E": float(f["params/theta_E"][sys_idx]),
            "dt_approx": float(f["approx_outputs/dt_approx"][sys_idx]),
            "I_obs": np.asarray(f["images/I_obs"][sys_idx], dtype=np.float32),
            "S_approx": np.asarray(f["approx_outputs/S_approx"][sys_idx], dtype=np.float32),
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
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_target_scaler(path: str | Path) -> dict:
    with open(str(path), "rb") as f:
        return pickle.load(f)
