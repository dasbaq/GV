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
from ml.inference.domain_membership import DEFAULT_PROFILE
from ml.inference.mode1 import run_mode1_correction
from ml.inference.fermat_ratio import run_fermat_ratio_posterior


ROOT = Path(__file__).resolve().parents[1]


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


def _apply_ml_correction(
    h0_approx: float,
    *,
    apply_correction: bool,
    checkpoint: str | Path | None,
    scaler: str | Path,
    config: str | Path,
    device: str,
    correction_approx_level: int,
    mode1_sigma_scale: float,
    observation: ObservedLensSystem,
    input_path: str | Path,
    system_index: int,
    delay: dict[str, Any],
    sie_fit: dict[str, Any],
    domain_profile: str | Path | None = DEFAULT_PROFILE,
) -> tuple[float, dict[str, Any]]:
    if not np.isfinite(mode1_sigma_scale) or mode1_sigma_scale <= 0.0:
        raise ValueError("mode1_sigma_scale must be finite and positive")
    metadata = {
        "posthoc_sigma_scale": float(mode1_sigma_scale),
        "scale_source": "filtered_val_abs_residual_over_pred_sigma_mean",
        "scale_note": "reserved for ML-predicted Mode 1 sigma; H0 point estimate is unchanged",
    }
    if not apply_correction:
        return h0_approx, {"applied": False, "reason": "correction disabled", **metadata}
    if checkpoint is None:
        return h0_approx, {
            "applied": False,
            "reason": "correction skipped: checkpoint not provided",
            **metadata,
        }
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return h0_approx, {
            "applied": False,
            "reason": f"correction skipped: checkpoint not found ({ckpt_path})",
            **metadata,
        }
    scaler_path = Path(scaler)
    if not scaler_path.exists():
        return h0_approx, {
            "applied": False,
            "checkpoint": str(ckpt_path),
            "reason": f"correction skipped: scaler not found ({scaler_path})",
            **metadata,
        }
    config_path = Path(config)
    if not config_path.exists():
        return h0_approx, {
            "applied": False,
            "checkpoint": str(ckpt_path),
            "scaler": str(scaler_path),
            "reason": f"correction skipped: ML config not found ({config_path})",
            **metadata,
        }
    correction = run_mode1_correction(
        observation=observation,
        input_path=input_path,
        system_index=system_index,
        delay=delay,
        sie_fit=sie_fit,
        h0_approx=h0_approx,
        checkpoint_path=ckpt_path,
        scaler_path=scaler_path,
        config_path=config_path,
        mode1_sigma_scale=mode1_sigma_scale,
        domain_profile_path=domain_profile,
        correction_approx_level=correction_approx_level,
        device_name=device,
    )
    if not correction.get("applied", False):
        return h0_approx, correction
    return h0_approx + float(correction["h0_correction"]), correction


def run_mode1(
    input_path: str | Path,
    *,
    system_index: int = 0,
    approx_level: int = 0,
    apply_correction: bool = False,
    correction_checkpoint: str | Path | None = None,
    correction_scaler: str | Path = ROOT / "data" / "target_scaler_phase4_v0_4.pkl",
    ml_config: str | Path = ROOT / "config" / "ml.yaml",
    correction_device: str = "auto",
    correction_approx_level: int = 1,
    mode1_sigma_scale: float = 1.0,
    domain_profile: str | Path | None = DEFAULT_PROFILE,
    apply_phi_correction: bool = False,
    phi_checkpoint: str | Path | None = None,
    phi_device: str = "auto",
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
        return_grid=bool(apply_correction),
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
    h0_corrected, correction = _apply_ml_correction(
        h0_approx,
        apply_correction=apply_correction,
        checkpoint=correction_checkpoint,
        scaler=correction_scaler,
        config=ml_config,
        device=correction_device,
        correction_approx_level=correction_approx_level,
        mode1_sigma_scale=mode1_sigma_scale,
        domain_profile=domain_profile,
        observation=observation,
        input_path=path,
        system_index=system_index,
        delay=delay,
        sie_fit=sie_fit,
    )
    phi_posterior: dict[str, Any] = {"applied": False, "reason": "phi correction disabled"}
    if apply_phi_correction:
        if phi_checkpoint is None or not Path(phi_checkpoint).exists():
            phi_posterior = {"applied": False, "reason": "phi correction skipped: checkpoint not provided or not found"}
        else:
            phi_posterior = run_fermat_ratio_posterior(input_path=path, system_index=system_index, observation=observation, sie_fit=sie_fit, checkpoint_path=phi_checkpoint, device_name=phi_device)
            samples = np.asarray(phi_posterior["log_dphi_truth_over_sie"]["samples"], dtype=float)
            phi_posterior["dphi_corrected_posterior"] = (float(sie_fit["dphi_rad2"]) * np.exp(samples)).tolist()
            # Downstream-only diagnostic: this value was not an ML input or target.
            phi_posterior["H0_downstream_diagnostic_posterior"] = (h0_approx * np.exp(samples)).tolist()

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
        "phi_ratio_posterior": phi_posterior,
    }
    return _jsonable(result)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run Mode 1 observation-to-H0 inversion")
    parser.add_argument("--input", required=True, help="Observation HDF5 path")
    parser.add_argument("--system-index", type=int, default=0)
    parser.add_argument("--approx-level", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--apply-correction", action="store_true")
    parser.add_argument("--apply-phi-correction", action="store_true", help="Run the separate H0-blind Fermat-ratio diagnostic.")
    parser.add_argument("--phi-checkpoint", help="H0-blind Fermat-ratio checkpoint")
    parser.add_argument("--phi-device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--correction-checkpoint")
    parser.add_argument(
        "--correction-scaler",
        default=str(ROOT / "data" / "target_scaler_phase4_v0_4.pkl"),
    )
    parser.add_argument("--ml-config", default=str(ROOT / "config" / "ml.yaml"))
    parser.add_argument("--correction-device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--correction-approx-level", type=int, default=1, choices=(1, 2))
    parser.add_argument("--mode1-sigma-scale", type=float, default=1.0,
                        help="Post-hoc multiplier reserved for Mode 1 ML sigma.")
    parser.add_argument(
        "--domain-profile",
        default=str(DEFAULT_PROFILE),
        help="Mode 1 ML domain-membership profile JSON. If missing, the Phase4 v0.4 catalog is used when available.",
    )
    parser.add_argument("--delay-config", help="Optional JSON config for Phase 1 delay extraction")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run_mode1(
        args.input,
        system_index=args.system_index,
        approx_level=args.approx_level,
        apply_correction=args.apply_correction,
        apply_phi_correction=args.apply_phi_correction,
        phi_checkpoint=args.phi_checkpoint,
        phi_device=args.phi_device,
        correction_checkpoint=args.correction_checkpoint,
        correction_scaler=args.correction_scaler,
        ml_config=args.ml_config,
        correction_device=args.correction_device,
        correction_approx_level=args.correction_approx_level,
        mode1_sigma_scale=args.mode1_sigma_scale,
        domain_profile=args.domain_profile,
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
