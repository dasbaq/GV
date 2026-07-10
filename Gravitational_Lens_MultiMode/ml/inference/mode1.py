"""Mode 1 ML correction inference utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from inversion.observation_io import ObservedLensSystem
from ml.inference.domain_membership import (
    DEFAULT_PROFILE,
    build_mode1_domain_features,
    load_or_build_mode1_domain_profile,
    score_mode1_domain_membership,
)
from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.dataset import _DT_LC_REL_SIGMA_EXPECTED
from ml.utils.mask import make_lc_mask
from ml.utils.normalize import build_param_vector


ROOT = Path(__file__).resolve().parents[2]
SCALE_SOURCE = "filtered_val_abs_residual_over_pred_sigma_mean"


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this process")
    return torch.device(name)


def load_cfg(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_target_scaler(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_model(cfg: dict[str, Any]) -> MultiModalErrorCorrector:
    param_norm = cfg["data"]["param_normalization"]
    model_cfg = dict(cfg["model"])
    model_cfg["param_in_dim"] = len(param_norm) + 5
    model_cfg["image_size"] = cfg["data"]["image_size"]
    return MultiModalErrorCorrector(model_cfg)


def _load_state_dict(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = loaded.get(key)
            if isinstance(value, dict):
                return value
    return loaded


def _compatibility_payload(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    config_path: str | Path,
    state: dict[str, torch.Tensor],
    model: MultiModalErrorCorrector,
) -> dict[str, Any]:
    ckpt_param = state.get("par_enc.net.0.weight")
    model_param = model.state_dict().get("par_enc.net.0.weight")
    ckpt_img = state.get("img_enc.enc1.net.0.weight")
    model_img = model.state_dict().get("img_enc.enc1.net.0.weight")
    return {
        "expected_round": "phase4_v0_6",
        "expected_scaler": "target_scaler_phase4_v0_4.pkl",
        "checkpoint": str(Path(checkpoint_path)),
        "scaler": str(Path(scaler_path)),
        "config": str(Path(config_path)),
        "checkpoint_param_in_dim": None if ckpt_param is None else int(ckpt_param.shape[1]),
        "config_param_in_dim": None if model_param is None else int(model_param.shape[1]),
        "checkpoint_image_channels": None if ckpt_img is None else int(ckpt_img.shape[1]),
        "config_image_channels": None if model_img is None else int(model_img.shape[1]),
    }


def _validate_checkpoint_compatible(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    config_path: str | Path,
    state: dict[str, torch.Tensor],
    model: MultiModalErrorCorrector,
) -> dict[str, Any]:
    payload = _compatibility_payload(checkpoint_path, scaler_path, config_path, state, model)
    errors: list[str] = []
    if payload["checkpoint_param_in_dim"] != payload["config_param_in_dim"]:
        errors.append(
            "ParamEncoder input dim mismatch "
            f"checkpoint={payload['checkpoint_param_in_dim']} config={payload['config_param_in_dim']}"
        )
    if payload["checkpoint_image_channels"] != payload["config_image_channels"]:
        errors.append(
            "ImageEncoder channel mismatch "
            f"checkpoint={payload['checkpoint_image_channels']} config={payload['config_image_channels']}"
        )
    if any(key.startswith("head3.") for key in state):
        errors.append("checkpoint contains legacy Mode 3 head weights")
    if errors:
        details = "; ".join(errors)
        raise RuntimeError(
            "Mode 1 ML correction artifact compatibility error: "
            f"{details}. Current config/ml.yaml expects phase4_v0_6 checkpoint "
            "with 20 ParamEncoder inputs and 1-channel I_obs image input; use "
            "data/checkpoints/phase4_v0_6_imgres_best.pt with "
            "data/target_scaler_phase4_v0_4.pkl, or pass a matching --ml-config."
        )
    return payload


def _as_2d(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("light curve arrays must be 1D or 2D")
    return arr


def _joint_light_curve(observation: ObservedLensSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    light_curves = observation.light_curves
    F = _as_2d(light_curves.F)
    t = _as_2d(light_curves.t_obs)
    sigma = _as_2d(light_curves.sigma_noise)
    if t.shape[0] == 1 and F.shape[0] > 1:
        t = np.repeat(t, F.shape[0], axis=0)
    if sigma.shape[0] == 1 and F.shape[0] > 1:
        sigma = np.repeat(sigma, F.shape[0], axis=0)
    if F.shape != t.shape or F.shape != sigma.shape:
        raise ValueError("F, t_obs, and sigma_noise must have matching shapes")

    order = np.argsort(t[0])
    t_ref = t[0, order]
    if F.shape[0] == 1:
        return t_ref, F[0, order], sigma[0, order]
    if not np.allclose(t[:, order], t_ref[None, :], rtol=0.0, atol=1.0e-8):
        raise ValueError("multi-image light curves must share the same t_obs grid")
    return t_ref, np.sum(F[:, order], axis=0), np.sqrt(np.sum(sigma[:, order] ** 2, axis=0))


def _finite_or_default(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _normalize_lc_for_inference(
    F_joint: np.ndarray,
    sigma_joint: np.ndarray,
    n_valid: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Normalize observed LC to the Phase 4 training scale.

    Units: flux and flux uncertainty are arbitrary linear units. SIE 표준 근사
    가정: this is observation-side ML preprocessing only and does not alter
    the lens model or the extracted delay.
    """

    valid_flux = np.asarray(F_joint[:n_valid], dtype=np.float32)
    if n_valid <= 1 or not np.isfinite(valid_flux).all():
        raise ValueError("valid light-curve flux must contain at least two finite epochs")
    mean = float(np.mean(valid_flux))
    std = float(np.std(valid_flux, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("valid light-curve flux standard deviation must be finite and positive")
    flux_norm = (np.asarray(F_joint[:n_valid], dtype=np.float32) - mean) / std
    sigma_norm = np.asarray(sigma_joint[:n_valid], dtype=np.float32) / std
    if not np.isfinite(flux_norm).all() or not np.isfinite(sigma_norm).all():
        raise ValueError("normalized light-curve channels contain non-finite values")
    metadata = {
        "method": "valid_epoch_standard_score",
        "flux_mean": mean,
        "flux_std": std,
        "n_valid": int(n_valid),
    }
    return flux_norm.astype(np.float32), sigma_norm.astype(np.float32), metadata


def _resample_1d(values: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == size:
        return arr
    if arr.size == 0:
        return np.zeros(size, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, size)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _sigma_curve_from_delay(delay: dict[str, Any], sigma_curve_size: int) -> np.ndarray:
    grid = delay.get("grid") if isinstance(delay, dict) else None
    if isinstance(grid, dict) and "sigma_map" in grid:
        sigma_map = np.asarray(grid["sigma_map"], dtype=np.float32)
        if sigma_map.ndim == 2 and sigma_map.size:
            finite = np.where(np.isfinite(sigma_map), sigma_map, np.inf)
            profile = np.min(finite, axis=1)
            profile = np.where(np.isfinite(profile), profile, 0.0)
            return _resample_1d(profile, sigma_curve_size)
    return np.zeros(sigma_curve_size, dtype=np.float32)


def _maybe_resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape == (image_size, image_size):
        return arr
    from skimage.transform import resize

    return resize(
        arr,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _optional_image(
    input_path: str | Path,
    system_index: int,
    image_size: int,
) -> tuple[np.ndarray, bool]:
    image = np.zeros((1, image_size, image_size), dtype=np.float32)
    with h5py.File(input_path, "r") as h5:
        has_image = "images" in h5 and "I_obs" in h5["images"]
        if not has_image:
            return image, False
        img_raw = _maybe_resize_image(h5["images/I_obs"][system_index], image_size)
    return img_raw[np.newaxis, ...].astype(np.float32), True


def build_mode1_batch(
    *,
    observation: ObservedLensSystem,
    input_path: str | Path,
    system_index: int,
    delay: dict[str, Any],
    sie_fit: dict[str, Any],
    h0_approx: float,
    cfg: dict[str, Any],
    correction_approx_level: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    data_cfg = cfg["data"]
    max_len = int(data_cfg["max_lc_len"])
    sigma_curve_size = int(data_cfg["sigma_curve_size"])
    image_size = int(data_cfg["image_size"])

    _, F_joint, sigma_joint = _joint_light_curve(observation)
    n_valid = min(int(F_joint.shape[-1]), max_len)
    flux_norm, sigma_norm, lc_norm_meta = _normalize_lc_for_inference(
        F_joint,
        sigma_joint,
        n_valid,
    )
    lc = np.zeros((2, max_len), dtype=np.float32)
    lc[0, :n_valid] = flux_norm
    lc[1, :n_valid] = sigma_norm

    dt_lc = _finite_or_default(delay.get("dt_obs_days"), 0.0)
    dt_unc = _finite_or_default(delay.get("dt_uncertainty_days"), dt_lc * _DT_LC_REL_SIGMA_EXPECTED)
    dt_lc_sigma = max(dt_unc, dt_lc * _DT_LC_REL_SIGMA_EXPECTED, 1.0e-6)
    raw_params = {
        "H0_approx": float(h0_approx),
        "z_lens": float(observation.z_lens),
        "z_source": float(observation.z_source),
        "sigma_v": float(sie_fit["sigma_v"]),
        "q": float(sie_fit["q"]),
        "theta_E": float(sie_fit["theta_E"]),
        "dt_lc": dt_lc,
        "dt_lc_sigma": dt_lc_sigma,
    }
    param_base = build_param_vector(raw_params, data_cfg["param_normalization"])
    al_onehot = np.array(
        [float(correction_approx_level == 1), float(correction_approx_level == 2)],
        dtype=np.float32,
    )
    mode_oh = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    params = np.concatenate([param_base, al_onehot, mode_oh]).astype(np.float32)
    sigma_curve = _sigma_curve_from_delay(delay, sigma_curve_size)
    image, use_image = _optional_image(input_path, system_index, image_size)

    batch = {
        "lc": torch.from_numpy(lc).unsqueeze(0),
        "lc_mask": make_lc_mask(n_valid, max_len).unsqueeze(0),
        "params": torch.from_numpy(params).unsqueeze(0),
        "sigma_curve": torch.from_numpy(sigma_curve[np.newaxis, np.newaxis]),
        "image": torch.from_numpy(image).unsqueeze(0),
        "use_image": torch.tensor([use_image], dtype=torch.bool),
        "target_mode": torch.tensor([1], dtype=torch.long),
    }
    metadata = {
        "n_valid_lc": int(n_valid),
        "use_image": bool(use_image),
        "correction_approx_level": int(correction_approx_level),
        "lc_normalization": lc_norm_meta,
        "raw_params": raw_params,
    }
    return batch, metadata


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_mode1_correction(
    *,
    observation: ObservedLensSystem,
    input_path: str | Path,
    system_index: int,
    delay: dict[str, Any],
    sie_fit: dict[str, Any],
    h0_approx: float,
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    config_path: str | Path,
    mode1_sigma_scale: float,
    domain_profile_path: str | Path | None = DEFAULT_PROFILE,
    correction_approx_level: int = 1,
    device_name: str = "auto",
) -> dict[str, Any]:
    if correction_approx_level not in {1, 2}:
        raise ValueError("correction_approx_level must be 1 or 2")
    if not np.isfinite(mode1_sigma_scale) or mode1_sigma_scale <= 0.0:
        raise ValueError("mode1_sigma_scale must be finite and positive")

    cfg = load_cfg(config_path)
    scaler = load_target_scaler(scaler_path)
    device = select_device(device_name)
    model = build_model(cfg).to(device)
    state = _load_state_dict(checkpoint_path, device)
    compatibility = _validate_checkpoint_compatible(
        checkpoint_path,
        scaler_path,
        config_path,
        state,
        model,
    )
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Mode 1 ML correction checkpoint failed to load after shape "
            "compatibility checks. Use a checkpoint generated with the same "
            "config/ml.yaml model definition."
        ) from exc
    model.eval()

    batch, input_metadata = build_mode1_batch(
        observation=observation,
        input_path=input_path,
        system_index=system_index,
        delay=delay,
        sie_fit=sie_fit,
        h0_approx=h0_approx,
        cfg=cfg,
        correction_approx_level=correction_approx_level,
    )
    profile, profile_artifact = load_or_build_mode1_domain_profile(
        domain_profile_path,
        config_path=config_path,
    )
    domain_features = build_mode1_domain_features(
        params_vector=batch["params"].numpy().reshape(-1),
        lc_tensor=batch["lc"].numpy().reshape(2, -1),
        n_valid_lc=int(input_metadata["n_valid_lc"]),
        image_tensor=batch["image"].numpy().reshape(batch["image"].shape[-2], batch["image"].shape[-1]),
        use_image=bool(input_metadata["use_image"]),
    )
    domain = score_mode1_domain_membership(
        features=domain_features,
        profile=profile,
        profile_artifact=profile_artifact,
        delay=delay,
        sie_fit=sie_fit,
        lc_normalization=input_metadata["lc_normalization"],
    )
    if domain["domain_grade"] == "ood_abstain":
        return {
            "applied": False,
            "reason": "correction abstained: OOD domain membership",
            "h0_correction": 0.0,
            "h0_correction_scaled_raw": None,
            "posthoc_sigma_scale": float(mode1_sigma_scale),
            "scale_source": SCALE_SOURCE,
            "checkpoint": str(Path(checkpoint_path)),
            "scaler": str(Path(scaler_path)),
            "config": str(Path(config_path)),
            "device": device.type,
            "correction_approx_level": int(correction_approx_level),
            "artifact_compatibility": compatibility,
            "input_adapter": input_metadata,
            "domain_membership": domain,
        }

    batch = _move_batch(batch, device)
    with torch.no_grad():
        out = model(
            lc=batch["lc"],
            lc_mask=batch["lc_mask"],
            params=batch["params"],
            sigma_curve=batch["sigma_curve"],
            image=batch["image"],
            target_mode=batch["target_mode"],
        )
    if out["mode1"] is None:
        raise RuntimeError("Mode 1 head did not run")

    pred_scaled = float(out["mode1"]["h0_correction"].detach().cpu().reshape(-1)[0])
    log_sigma = float(out["mode1"]["log_sigma"].detach().cpu().reshape(-1)[0])
    mode1_scaler = scaler["mode1"]
    target_mean = float(mode1_scaler["mean"])
    target_scale = float(mode1_scaler["scale"])
    pred_corr = pred_scaled * target_scale + target_mean
    sigma_raw = float(np.exp(log_sigma) * target_scale)
    domain_multiplier = domain.get("sigma_scale_multiplier")
    if domain_multiplier is None:
        domain_multiplier = 1.0
    sigma_scaled = sigma_raw * float(mode1_sigma_scale) * float(domain_multiplier)

    return {
        "applied": True,
        "h0_correction": float(pred_corr),
        "h0_correction_scaled_raw": float(pred_scaled),
        "log_sigma": float(log_sigma),
        "sigma_H0_raw": sigma_raw,
        "sigma_H0_scaled": sigma_scaled,
        "posthoc_sigma_scale": float(mode1_sigma_scale),
        "domain_sigma_scale_multiplier": float(domain_multiplier),
        "scale_source": SCALE_SOURCE,
        "checkpoint": str(Path(checkpoint_path)),
        "scaler": str(Path(scaler_path)),
        "config": str(Path(config_path)),
        "device": device.type,
        "correction_approx_level": int(correction_approx_level),
        "artifact_compatibility": compatibility,
        "input_adapter": input_metadata,
        "domain_membership": domain,
    }
