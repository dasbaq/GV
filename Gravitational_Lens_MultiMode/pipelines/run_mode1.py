"""CLI entrypoint for observation-to-H0 Mode 1 inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from inversion.delay_extraction import extract_delay_from_observation
from inversion.mode1_h0 import invert_h0
from inversion.observation_io import ObservedLensSystem, from_hdf5
from inversion.sie_fit import fit_sie_to_images


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("delay config JSON must contain an object")
    return loaded


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _mock_flag(path: Path, observation: ObservedLensSystem) -> bool:
    with h5py.File(path, "r") as h5:
        if "metadata" in h5:
            meta = h5["metadata"].attrs
            for key in ("mock", "is_mock", "synthetic"):
                if key in meta:
                    return bool(meta[key])
            source = str(meta.get("source", "")).lower()
            if "mock" in source or "synthetic" in source:
                return True
        return bool(observation.name and "mock" in observation.name.lower())


_DEFAULT_CORRECTION_CONFIG = Path(__file__).resolve().parent.parent / "config" / "ml.yaml"
_DEFAULT_CORRECTION_SCALER = (
    Path(__file__).resolve().parent.parent / "data" / "target_scaler_phase4_v0_4.pkl"
)


def _apply_ml_correction(
    h0_approx: float,
    *,
    apply_correction: bool,
    checkpoint: str | Path | None,
    feature_spec: dict[str, Any] | None,
    scaler_path: str | Path | None,
    config_path: str | Path | None,
) -> tuple[float, dict[str, Any]]:
    """Apply the Phase 4/5 Mode 1 corrector: H0_corrected = H0_approx + correction.

    Loads the trained MultiModalErrorCorrector and target scaler, builds the
    Mode 1 input tensors from ``feature_spec`` (same construction as the training
    dataset), runs a target_mode=1 forward, and inverse-transforms the scaled
    prediction. Units: H0 [km/s/Mpc]. SIE 표준 근사 가정은 corrector 학습 시와 동일.
    """

    if not apply_correction:
        return h0_approx, {"applied": False, "reason": "correction disabled"}
    if checkpoint is None:
        return h0_approx, {"applied": False, "reason": "correction skipped: checkpoint not provided"}
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return h0_approx, {
            "applied": False,
            "reason": f"correction skipped: checkpoint not found ({ckpt_path})",
        }
    if feature_spec is None:
        return h0_approx, {
            "applied": False,
            "checkpoint": str(ckpt_path),
            "reason": "correction skipped: image/sigma_curve feature inputs unavailable in input",
        }

    import torch

    from inversion.obs_to_features import (
        build_corrector_inputs,
        load_corrector,
        load_target_scaler,
    )

    cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CORRECTION_CONFIG
    scl_path = Path(scaler_path) if scaler_path is not None else _DEFAULT_CORRECTION_SCALER
    if not scl_path.exists():
        return h0_approx, {
            "applied": False,
            "checkpoint": str(ckpt_path),
            "reason": f"correction skipped: scaler not found ({scl_path})",
        }

    model, cfg = load_corrector(ckpt_path, cfg_path)
    scaler = load_target_scaler(scl_path)
    inputs = build_corrector_inputs(
        feature_spec,
        param_norm=cfg["data"]["param_normalization"],
        image_size=int(cfg["data"]["image_size"]),
        max_len=int(cfg["data"]["max_lc_len"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
        approx_level=1,
        target_mode=1,
    )
    with torch.no_grad():
        out = model(**inputs)
    m1 = out["mode1"]
    s = scaler["mode1"]
    scale = float(s["scale"])
    mean = float(s["mean"])
    correction = float(m1["h0_correction"].item()) * scale + mean
    sigma = float(torch.exp(m1["log_sigma"]).item()) * scale
    h0_corrected = float(h0_approx) + correction
    return h0_corrected, {
        "applied": True,
        "checkpoint": str(ckpt_path),
        "scaler": str(scl_path),
        "correction": correction,
        "sigma": sigma,
    }


def _feature_spec_from_phase4_hdf5(
    path: Path,
    system_index: int,
    *,
    h0_approx: float,
    dt_approx: float,
    sie_fit: dict[str, Any],
    z_lens: float,
    z_source: float,
) -> dict[str, Any] | None:
    """Build a corrector feature spec from a Phase 4-schema HDF5 system.

    Image/LC/sigma_curve inputs are read from the HDF5; the inference-side
    parameter fields are overridden with the analytic Mode 1 outputs that this
    run produced (H0_approx, dt_approx, fitted SIE σ_v/q/θ_E). Returns ``None``
    if the input lacks the Phase 4 inference groups (e.g. a minimal observation
    file), so correction degrades gracefully.
    """

    with h5py.File(path, "r") as h5:
        needed = ("images/I_obs", "approx_outputs/S_approx", "light_curves/F_joint")
        if not all(key in h5 for key in needed):
            return None
        spec = {
            "F_joint": np.asarray(h5["light_curves/F_joint"][system_index], dtype=np.float32),
            "sigma_noise": np.asarray(h5["light_curves/sigma_noise"][system_index], dtype=np.float32),
            "n_epochs": int(h5["light_curves/n_epochs"][system_index]),
            "I_obs": np.asarray(h5["images/I_obs"][system_index], dtype=np.float32),
            "S_approx": np.asarray(h5["approx_outputs/S_approx"][system_index], dtype=np.float32),
            "sigma_curve": (
                np.asarray(h5["sigma_curve"][system_index], dtype=np.float32)
                if "sigma_curve" in h5
                else None
            ),
        }
    spec.update(
        {
            "H0_approx": float(h0_approx),
            "dt_approx": float(dt_approx),
            "sigma_v": float(sie_fit["sigma_v"]),
            "q": float(sie_fit["q"]),
            "theta_E": float(sie_fit["theta_E"]),
            "z_lens": float(z_lens),
            "z_source": float(z_source),
        }
    )
    return spec


def run_mode1(
    input_path: str | Path,
    *,
    system_index: int = 0,
    approx_level: int = 0,
    apply_correction: bool = False,
    correction_checkpoint: str | Path | None = None,
    correction_scaler: str | Path | None = None,
    correction_config: str | Path | None = None,
    delay_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run observation-to-H0 Mode 1 inversion.

    Units: observed light-curve times and ``dt_obs_days`` are [days],
    image positions are [arcsec], fitted ``dphi_rad2`` is [radian^2], and H0 is
    [km/s/Mpc]. SIE 표준 근사 가정: single lens plane, κ_ext=0, smooth SIE
    mass profile, isotropic velocity dispersion. ML correction is an optional
    placeholder and is disabled by default.
    """

    path = Path(input_path)
    observation = from_hdf5(path, system_index=system_index)
    is_mock = _mock_flag(path, observation)

    delay = extract_delay_from_observation(
        observation,
        delay_config,
        is_mock=is_mock,
        return_grid=False,
    )
    if delay["confidence_grade"] == "rejected" or not np.isfinite(delay["dt_obs_days"]):
        raise RuntimeError("time-delay extraction rejected the observation")

    sie_fit = fit_sie_to_images(
        observation.image_positions,
        observation.z_lens,
        observation.z_source,
        cosmology={"H0": 70.0},
    )
    h0 = invert_h0(
        np.array([delay["dt_obs_days"]], dtype=float),
        np.array([sie_fit["dphi_rad2"]], dtype=float),
        observation.z_lens,
        observation.z_source,
        approx_level=approx_level,
        n_bootstrap=200,
    )
    h0_approx = float(h0["H0"])
    feature_spec = None
    if apply_correction and correction_checkpoint is not None and Path(correction_checkpoint).exists():
        feature_spec = _feature_spec_from_phase4_hdf5(
            path,
            system_index,
            h0_approx=h0_approx,
            dt_approx=float(delay["dt_obs_days"]),
            sie_fit=sie_fit,
            z_lens=observation.z_lens,
            z_source=observation.z_source,
        )
    h0_corrected, correction = _apply_ml_correction(
        h0_approx,
        apply_correction=apply_correction,
        checkpoint=correction_checkpoint,
        feature_spec=feature_spec,
        scaler_path=correction_scaler,
        config_path=correction_config,
    )

    result = {
        "input": str(path),
        "system_index": int(system_index),
        "mock": bool(is_mock),
        "approx_level": int(approx_level),
        "H0_approx": h0_approx,
        "H0": float(h0_corrected),
        "H0_uncertainty": float(h0["H0_uncertainty"]),
        "dt_obs_days": float(delay["dt_obs_days"]),
        "dt_uncertainty_days": float(delay["dt_uncertainty_days"]),
        "mu_time_delay": float(delay["mu"]),
        "sigma_min": float(delay["sigma_min"]),
        "confidence_grade": delay["confidence_grade"],
        "dphi_rad2": float(sie_fit["dphi_rad2"]),
        "sie_fit": {
            "sigma_v": sie_fit["sigma_v"],
            "q": sie_fit["q"],
            "position_angle": sie_fit["position_angle"],
            "source_pos_xy": sie_fit["source_pos_xy"],
            "theta_E": sie_fit["theta_E"],
            "theta_1": sie_fit["theta_1"],
            "theta_2": sie_fit["theta_2"],
            "mu_fit": sie_fit["mu_fit"],
            "residual_rms_arcsec": sie_fit["residual_rms_arcsec"],
            "max_residual_arcsec": sie_fit["max_residual_arcsec"],
        },
        "ml_correction": correction,
    }
    return _jsonable(result)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run Mode 1 observation-to-H0 inversion")
    parser.add_argument("--input", required=True, help="Observation HDF5 path")
    parser.add_argument("--system-index", type=int, default=0)
    parser.add_argument("--approx-level", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--apply-correction", action="store_true")
    parser.add_argument("--correction-checkpoint")
    parser.add_argument("--correction-scaler", help="target scaler pkl (default v0.4)")
    parser.add_argument("--correction-config", help="ML config yaml (default config/ml.yaml)")
    parser.add_argument("--delay-config", help="Optional JSON config for Phase 1 delay extraction")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run_mode1(
        args.input,
        system_index=args.system_index,
        approx_level=args.approx_level,
        apply_correction=args.apply_correction,
        correction_checkpoint=args.correction_checkpoint,
        correction_scaler=args.correction_scaler,
        correction_config=args.correction_config,
        delay_config=_load_json(args.delay_config),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
