"""CLI entrypoint for observation-to-DM Mode 2 SIE inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from inversion.mode2_dm import PARAM_NAMES, invert_dm


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


def _read_indexed(group: h5py.Group, key: str, system_index: int) -> np.ndarray:
    if key not in group:
        raise KeyError(f"missing HDF5 dataset {group.name}/{key}")
    arr = np.asarray(group[key][...])
    if arr.ndim == 0:
        return arr
    return np.asarray(arr[int(system_index)])


def _read_theta_obs(h5: h5py.File, system_index: int) -> np.ndarray:
    """Read public image positions [arcsec] from Phase 4 HDF5.

    SIE 표준 근사 가정: image positions are observation-side inputs; truth-only
    DM labels and ``true_values/mu_true`` are deliberately not read here.
    """

    rays = h5["ray_paths"]
    for key in ("theta_all", "image_positions", "theta_extra"):
        if key in rays:
            arr = np.asarray(rays[key][int(system_index)], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 2:
                return arr
    theta_1 = _read_indexed(rays, "theta_1", system_index)
    theta_2 = _read_indexed(rays, "theta_2", system_index)
    return np.stack([theta_1, theta_2], axis=0).astype(float)


def _mock_flag(h5: h5py.File) -> bool:
    if "metadata" not in h5:
        return False
    meta = h5["metadata"].attrs
    for key in ("mock", "is_mock", "synthetic", "full_truth_available"):
        if key in meta:
            return bool(meta[key])
    text = " ".join(str(meta.get(key, "")) for key in ("source", "generator_version", "truth_model")).lower()
    return "mock" in text or "synthetic" in text or "phase4" in text


def _load_public_system(path: Path, system_index: int, H0_override: float | None) -> dict[str, Any]:
    """Load only public Mode 2 inference inputs from HDF5.

    Units: positions [arcsec], delay/sigma [days], H0 [km/s/Mpc], redshifts
    dimensionless. SIE 표준 근사 가정: no truth-side key is accessed in this
    loader, including ``true_values/mu_true`` and ``true_values/dm_params_true``.
    """

    with h5py.File(path, "r") as h5:
        params = h5["params"]
        obs = h5["observed_features"]
        theta = _read_theta_obs(h5, system_index)
        h0 = float(H0_override) if H0_override is not None else float(_read_indexed(params, "H0", system_index))
        return {
            "theta_obs": theta,
            "dt_obs": np.array([float(_read_indexed(obs, "dt_lc", system_index))], dtype=float),
            "dt_sigma": np.array([float(_read_indexed(obs, "dt_lc_sigma", system_index))], dtype=float),
            "H0": h0,
            "z_lens": float(_read_indexed(params, "z_lens", system_index)),
            "z_source": float(_read_indexed(params, "z_source", system_index)),
            "mock": _mock_flag(h5),
        }


def _angle_delta(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    delta = np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)
    return np.arctan2(np.sin(delta), np.cos(delta))


def _relative_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    delta = pred - truth
    if delta.shape[-1] >= 3:
        delta = delta.copy()
        delta[..., 2] = _angle_delta(pred[..., 2], truth[..., 2])
    denom = np.maximum(np.abs(truth), 1.0e-8)
    if delta.shape[-1] >= 3:
        denom = denom.copy()
        denom[..., 2] = 1.0
    return np.abs(delta) / denom


def _truth_eval_for_system(path: Path, system_index: int, prediction: dict[str, Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        truth = np.asarray(h5["true_values/dm_params_true"][int(system_index)], dtype=float)
    pred = np.asarray(prediction["dm_params"], dtype=float)
    rel = _relative_error(pred, truth)
    unc = np.asarray(prediction["dm_uncertainty"], dtype=float)
    delta = pred - truth
    if delta.shape[0] >= 3:
        delta[2] = _angle_delta(pred[2], truth[2])
    return {
        "mock": True,
        "truth_source": "true_values/dm_params_true",
        "param_names": list(PARAM_NAMES),
        "truth": truth,
        "relative_error": rel,
        "absolute_error": np.abs(delta),
        "within_1sigma": np.abs(delta) <= np.maximum(unc, 1.0e-12),
    }


def _summarize_truth_eval(preds: np.ndarray, truths: np.ndarray, uncs: np.ndarray) -> dict[str, Any]:
    rel = _relative_error(preds, truths)
    delta = preds - truths
    delta[:, 2] = _angle_delta(preds[:, 2], truths[:, 2])
    finite = np.isfinite(rel) & np.isfinite(delta)
    summary: dict[str, Any] = {
        "mock": True,
        "mock_reason": "Phase 4 synthetic truth catalog; not a real observation benchmark",
        "param_names": list(PARAM_NAMES),
        "n_systems": int(preds.shape[0]),
        "truth_source": "true_values/dm_params_true",
        "input_audit": {
            "uses_theta_obs": True,
            "uses_dt_obs": True,
            "uses_mu_obs": False,
            "uses_truth_mu_true": False,
            "truth_dm_params_read_only_for_eval": True,
        },
        "per_param": {},
    }
    for j, name in enumerate(PARAM_NAMES):
        mask = finite[:, j]
        values = rel[mask, j]
        bias = delta[mask, j]
        coverage = np.abs(delta[mask, j]) <= np.maximum(uncs[mask, j], 1.0e-12)
        summary["per_param"][name] = {
            "finite_count": int(mask.sum()),
            "relative_error_median": float(np.median(values)) if values.size else None,
            "relative_error_p90": float(np.percentile(values, 90)) if values.size else None,
            "bias_mean": float(np.mean(bias)) if bias.size else None,
            "bias_median": float(np.median(bias)) if bias.size else None,
            "coverage_1sigma": float(np.mean(coverage)) if coverage.size else None,
        }
    return summary


def run_mode2(
    input_path: str | Path,
    *,
    system_index: int = 0,
    H0: float | None = None,
    n_bootstrap: int = 0,
    eval_truth: bool = False,
    eval_max_systems: int | None = None,
) -> dict[str, Any]:
    """Run observation-to-DM Mode 2 SIE inversion.

    Units: image positions [arcsec], observed delay [days], H0 [km/s/Mpc],
    returned Fermat difference [rad²]. SIE 표준 근사 가정: single lens plane,
    κ_ext=0, smooth SIE mass profile, isotropic velocity dispersion. Truth
    labels are read only when ``eval_truth=True``.
    """

    path = Path(input_path)
    if not eval_truth:
        public = _load_public_system(path, system_index, H0)
        result = invert_dm(
            public["dt_obs"],
            public["theta_obs"],
            mu_obs=None,
            H0=public["H0"],
            z_lens=public["z_lens"],
            z_source=public["z_source"],
            lens_model="SIE",
            approx_level=0,
            n_bootstrap=n_bootstrap,
            dt_sigma=public["dt_sigma"],
        )
        out = {
            "input": str(path),
            "system_index": int(system_index),
            "mock": bool(public["mock"]),
            **result,
        }
        return _jsonable(out)

    with h5py.File(path, "r") as h5:
        n_total = int(h5["params/z_lens"].shape[0])
    # Phase 4 catalogs can be hundreds of systems and each Mode 2 solve is a
    # nonlinear fit. By default the MOCK evaluation records a fixed small
    # subset; callers can pass --eval-max-systems to widen or shrink it.
    n_eval = min(n_total, 25) if eval_max_systems is None else min(n_total, int(eval_max_systems))
    results: list[dict[str, Any]] = []
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    uncs: list[np.ndarray] = []
    for idx in range(n_eval):
        public = _load_public_system(path, idx, H0)
        result = invert_dm(
            public["dt_obs"],
            public["theta_obs"],
            mu_obs=None,
            H0=public["H0"],
            z_lens=public["z_lens"],
            z_source=public["z_source"],
            lens_model="SIE",
            approx_level=0,
            n_bootstrap=n_bootstrap,
            dt_sigma=public["dt_sigma"],
            rng_seed=42 + idx,
        )
        result = {"input": str(path), "system_index": int(idx), "mock": bool(public["mock"]), **result}
        result["truth_eval"] = _truth_eval_for_system(path, idx, result)
        results.append(_jsonable(result))
        preds.append(np.asarray(result["dm_params"], dtype=float))
        truths.append(np.asarray(result["truth_eval"]["truth"], dtype=float))
        uncs.append(np.asarray(result["dm_uncertainty"], dtype=float))

    summary = _summarize_truth_eval(np.stack(preds), np.stack(truths), np.stack(uncs))
    return _jsonable(
        {
            "input": str(path),
            "mode": "Mode 2 SIE DM recovery",
            "summary": summary,
            "systems": results,
        }
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run Mode 2 observation-to-DM SIE inversion")
    parser.add_argument("--input", required=True, help="Phase 4 HDF5 path")
    parser.add_argument("--system-index", type=int, default=0)
    parser.add_argument("--H0", type=float, help="Optional fixed H0 override [km/s/Mpc]")
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--eval-truth", action="store_true")
    parser.add_argument("--eval-max-systems", type=int, help="Optional cap for MOCK truth evaluation")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run_mode2(
        args.input,
        system_index=args.system_index,
        H0=args.H0,
        n_bootstrap=args.n_bootstrap,
        eval_truth=args.eval_truth,
        eval_max_systems=args.eval_max_systems,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
