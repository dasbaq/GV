#!/usr/bin/env python3
"""Estimate the Phase 4 v0.4 inputs-conditioned achievable-r ceiling.

The oracle is fit in correction space with the same Mode 1 inputs seen by the
corrector, then evaluated in H0 space:

    r = corr(H0_true, H0_approx + E[correction | inputs])

Labels and H0_true are read only after feature extraction, for evaluation.
Truth-side simulation keys are never used as features.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.training.dataset import LensCorrectionDataset  # noqa: E402

ROUND = "phase4_v0_4"
DEFAULT_TRAIN = ROOT / "data" / "mock" / "phase4_v0_4.h5"
DEFAULT_UNFILTERED = ROOT / "data" / "mock" / "phase4_v0_4_eval_unfiltered.h5"
DEFAULT_SCALER = ROOT / "data" / "target_scaler_phase4_v0_4.pkl"
DEFAULT_CONFIG = ROOT / "config" / "ml.yaml"
DEFAULT_FILTERED_EVAL = ROOT / "data" / "logs" / "phase4_v0_4_imgres_h0_eval.json"
DEFAULT_UNFILTERED_EVAL = ROOT / "data" / "logs" / "phase4_v0_4_imgres_h0_eval_unfiltered.json"
DEFAULT_OUTPUT = ROOT / "data" / "logs" / "phase4_v0_4_r_ceiling.json"

FORBIDDEN_FEATURE_PATHS = (
    "true_values/H0_true",
    "true_values/D_delta_t",
    "true_values/dt_true",
    "true_values/mu_true",
    "true_values/dm_params_true",
    "true_values/dm_dim",
    "true_values/theta_E",
    "params/H0",
    "params/M200",
    "params/concentration",
    "params/lens_truth_model",
    "correction_targets/mode1_H0_correction",
    "simplification_errors/mode1_H0_error",
)

USED_FEATURE_PATHS = {
    "params": [
        "approx_outputs/H0_approx",
        "params/z_lens",
        "params/z_source",
        "params/sigma_v",
        "params/q",
        "params/theta_E",
        "observed_features/dt_lc",
        "observed_features/dt_lc_sigma",
        "observed_features/n_epochs_quality",
        "observed_features/baseline_days",
        "observed_features/median_cadence_days",
        "observed_features/median_photometric_error",
        "derived/missing_sigma_v",
        "derived/missing_theta_E",
        "derived/missing_q",
        "derived/approx_level_onehot",
        "derived/target_mode_onehot",
    ],
    "light_curve": [
        "light_curves/F_joint",
        "light_curves/sigma_noise",
        "light_curves/n_epochs",
    ],
    "sigma_curve": ["sigma_curve"],
    # image modality deleted in v0.5 (DECISIONS.md [2026-05-25])
}


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return yaml.safe_load(fp)


def load_scaler(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fp:
        return pickle.load(fp)


def split_system_ids(path: Path, split: str, seed: int) -> np.ndarray:
    """Match scripts/phase4_v0_4_round.py split policy exactly."""

    with h5py.File(path, "r") as f:
        n = int(f["metadata"].attrs["n_systems"])
    rng = np.random.default_rng(seed)
    rng_local = np.random.default_rng(int(rng.integers(0, 2**31)))
    ids = np.arange(n)
    rng_local.shuffle(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    if split == "train":
        return ids[:n_train]
    if split == "val":
        return ids[n_train:n_train + n_val]
    if split == "test":
        return ids[n_train + n_val:]
    raise ValueError(f"unknown split: {split!r}")


def _make_dataset(cfg: dict[str, Any], path: Path) -> LensCorrectionDataset:
    return LensCorrectionDataset(
        [path],
        split="train",
        modes=(1,),
        approx_levels=(1,),
        max_len=int(cfg["data"]["max_lc_len"]),
        sigma_curve_size=int(cfg["data"]["sigma_curve_size"]),
        mode2_max_dm_dim=int(cfg["model"]["mode2_max_dm_dim"]),
        param_norm=cfg["data"]["param_normalization"],
        target_scaler=None,
        observed_feature_config=cfg["data"].get("observed_features", {}),
        seed=int(cfg["seed"]),
    )


def _flatten_sample(sample: dict[str, Any]) -> np.ndarray:
    parts = [
        sample["lc"].numpy().reshape(-1),
        sample["lc_mask"].numpy().astype(np.float32).reshape(-1),
        sample["params"].numpy().reshape(-1),
        sample["sigma_curve"].numpy().reshape(-1),
    ]
    return np.concatenate(parts).astype(np.float32, copy=False)


def extract_features(cfg: dict[str, Any], path: Path, ids: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    ds = _make_dataset(cfg, path)
    ds._index = [(str(path), int(i), 1, 1) for i in np.asarray(ids, dtype=int)]
    rows = []
    dims: dict[str, int] | None = None
    for i in range(len(ds)):
        sample = ds[i]
        if dims is None:
            dims = {
                "lc": int(sample["lc"].numel()),
                "lc_mask": int(sample["lc_mask"].numel()),
                "params": int(sample["params"].numel()),
                "sigma_curve": int(sample["sigma_curve"].numel()),
            }
        rows.append(_flatten_sample(sample))
    x = np.vstack(rows).astype(np.float32, copy=False)
    dims = dims or {}
    dims["total"] = int(x.shape[1])
    expected_params = len(cfg["data"]["param_normalization"]) + 5
    if dims.get("params") != expected_params:
        raise AssertionError(f"ParamEncoder dimension {dims.get('params')} != expected {expected_params}")
    return x, dims


def read_targets(path: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "correction_targets/mode1_H0_correction" in f:
            correction = np.asarray(f["correction_targets/mode1_H0_correction"][:], dtype=np.float64)
        else:
            correction = np.asarray(f["simplification_errors/mode1_H0_error"][:], dtype=np.float64)
        return {
            "correction": correction[ids],
            "h0_true": np.asarray(f["true_values/H0_true"][:], dtype=np.float64)[ids],
            "h0_approx": np.asarray(f["approx_outputs/H0_approx"][:], dtype=np.float64)[ids],
        }


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def metric_point(truth: dict[str, np.ndarray], pred_corr: np.ndarray) -> dict[str, float]:
    oracle_h0 = truth["h0_approx"] + pred_corr
    no_corr = truth["h0_approx"]
    return {
        "oracle_h0_r": pearson_r(truth["h0_true"], oracle_h0),
        "oracle_correction_r": pearson_r(truth["correction"], pred_corr),
        "oracle_h0_rmse": rmse(oracle_h0 - truth["h0_true"]),
        "oracle_correction_rmse": rmse(pred_corr - truth["correction"]),
        "no_correction_h0_r": pearson_r(truth["h0_true"], no_corr),
        "no_correction_h0_rmse": rmse(no_corr - truth["h0_true"]),
    }


def load_model_r(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open() as fp:
        data = json.load(fp)
    try:
        return float(data["best"]["mode1"]["h0"]["model"]["r"])
    except (KeyError, TypeError, ValueError):
        return None


def _ci(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "ci95": None}
    return {
        "mean": float(np.mean(arr)),
        "ci95": [float(x) for x in np.percentile(arr, [2.5, 97.5])],
    }


def bootstrap_metrics(
    truth: dict[str, np.ndarray],
    pred_corr: np.ndarray,
    *,
    model_r: float | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(pred_corr)
    boot = {
        "oracle_h0_r": [],
        "oracle_correction_r": [],
        "oracle_h0_rmse": [],
        "oracle_correction_rmse": [],
        "model_to_oracle_h0_r_ratio": [],
    }
    for _ in range(n_bootstrap):
        b = rng.integers(0, n, n)
        point = metric_point({k: v[b] for k, v in truth.items()}, pred_corr[b])
        for key in ("oracle_h0_r", "oracle_correction_r", "oracle_h0_rmse", "oracle_correction_rmse"):
            boot[key].append(point[key])
        if model_r is not None and np.isfinite(point["oracle_h0_r"]) and point["oracle_h0_r"] != 0.0:
            boot["model_to_oracle_h0_r_ratio"].append(float(model_r / point["oracle_h0_r"]))
    return {key: _ci(values) for key, values in boot.items()}


def evaluate_set(
    *,
    label: str,
    path: Path,
    ids: np.ndarray,
    cfg: dict[str, Any],
    estimator: ExtraTreesRegressor,
    model_eval_json: Path,
    bootstrap_n: int,
    seed: int,
) -> dict[str, Any]:
    x, dims = extract_features(cfg, path, ids)
    truth = read_targets(path, ids)
    pred_corr = estimator.predict(x).astype(np.float64)
    point = metric_point(truth, pred_corr)
    model_r = load_model_r(model_eval_json)
    ratio = None if model_r is None or point["oracle_h0_r"] == 0.0 else float(model_r / point["oracle_h0_r"])
    boot = bootstrap_metrics(
        truth,
        pred_corr,
        model_r=model_r,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )
    return {
        "eval_set": label,
        "data": _display(path),
        "n": int(len(ids)),
        "ids_source": "phase4_v0_4_round.split_system_ids(val)" if label == "filtered_val" else "all systems",
        "feature_dimensions": dims,
        "h0_space": {
            "oracle_r": point["oracle_h0_r"],
            "oracle_r_bootstrap": boot["oracle_h0_r"],
            "oracle_rmse": point["oracle_h0_rmse"],
            "oracle_rmse_bootstrap": boot["oracle_h0_rmse"],
            "no_correction_r": point["no_correction_h0_r"],
            "no_correction_rmse": point["no_correction_h0_rmse"],
            "model_r_existing_eval_json": model_r,
            "model_to_oracle_r_ratio": ratio,
            "model_to_oracle_r_ratio_bootstrap": boot["model_to_oracle_h0_r_ratio"],
        },
        "correction_space_diagnostic": {
            "oracle_r": point["oracle_correction_r"],
            "oracle_r_bootstrap": boot["oracle_correction_r"],
            "oracle_rmse": point["oracle_correction_rmse"],
            "oracle_rmse_bootstrap": boot["oracle_correction_rmse"],
        },
    }


def floor_to_2_decimals(value: float) -> float:
    return math.floor(float(value) * 100.0) / 100.0


def validate_leak_guard() -> None:
    used = [path for paths in USED_FEATURE_PATHS.values() for path in paths]
    forbidden = set(FORBIDDEN_FEATURE_PATHS)
    overlap = sorted(path for path in used if path in forbidden)
    if overlap:
        raise AssertionError(f"forbidden truth-side feature paths are configured as inputs: {overlap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--unfiltered", type=Path, default=DEFAULT_UNFILTERED)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--filtered-eval-json", type=Path, default=DEFAULT_FILTERED_EVAL)
    parser.add_argument("--unfiltered-eval-json", type=Path, default=DEFAULT_UNFILTERED_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.5)
    args = parser.parse_args()

    validate_leak_guard()
    cfg = load_cfg(args.config)
    scaler = load_scaler(args.scaler)
    split_seed = int(cfg["seed"])
    train_ids = split_system_ids(args.train, "train", split_seed)
    filtered_ids = split_system_ids(args.train, "val", split_seed)
    with h5py.File(args.unfiltered, "r") as f:
        unfiltered_ids = np.arange(int(f["metadata"].attrs["n_systems"]))

    x_train, dims = extract_features(cfg, args.train, train_ids)
    y_train = read_targets(args.train, train_ids)["correction"]
    estimator_params = {
        "n_estimators": int(args.n_estimators),
        "min_samples_leaf": int(args.min_samples_leaf),
        "max_features": float(args.max_features),
        "random_state": int(args.seed),
        "n_jobs": -1,
    }
    estimator = ExtraTreesRegressor(**estimator_params)
    estimator.fit(x_train, y_train)

    filtered = evaluate_set(
        label="filtered_val",
        path=args.train,
        ids=filtered_ids,
        cfg=cfg,
        estimator=estimator,
        model_eval_json=args.filtered_eval_json,
        bootstrap_n=int(args.bootstrap_n),
        seed=int(args.seed) + 1,
    )
    unfiltered = evaluate_set(
        label="unfiltered_all",
        path=args.unfiltered,
        ids=unfiltered_ids,
        cfg=cfg,
        estimator=estimator,
        model_eval_json=args.unfiltered_eval_json,
        bootstrap_n=int(args.bootstrap_n),
        seed=int(args.seed) + 2,
    )

    filtered_min = floor_to_2_decimals(0.80 * filtered["h0_space"]["oracle_r"])
    result = {
        "round": ROUND,
        "created_by": Path(__file__).name,
        "method": {
            "definition": "inputs-conditioned oracle fit in correction space; final ceiling is H0-space Pearson r",
            "estimator": "sklearn.ensemble.ExtraTreesRegressor",
            "hyperparameters": estimator_params,
            "seed": int(args.seed),
            "bootstrap_n": int(args.bootstrap_n),
            "split_ids_policy": {
                "train_fit": "phase4_v0_4.h5 train split",
                "filtered_eval": "phase4_v0_4.h5 val split",
                "unfiltered_eval": "phase4_v0_4_eval_unfiltered.h5 all systems",
                "split_seed": split_seed,
                "train_n": int(len(train_ids)),
                "filtered_val_n": int(len(filtered_ids)),
                "unfiltered_n": int(len(unfiltered_ids)),
            },
        },
        "inputs_used": {
            "modalities": USED_FEATURE_PATHS,
            "feature_dimensions": dims,
            "param_encoder_order": list(cfg["data"]["param_normalization"].keys())
            + ["approx_level_is_1", "approx_level_is_2", "target_mode_1", "target_mode_2", "target_mode_3"],
            "dataset_path": "ml/training/dataset.py LensCorrectionDataset Mode 1 path",
            "image_modality": "deleted in v0.5 (DECISIONS.md [2026-05-25]); features are LC+mask+params+sigma_curve only",
        },
        "leak_guard": {
            "forbidden_feature_paths": list(FORBIDDEN_FEATURE_PATHS),
            "used_feature_paths": USED_FEATURE_PATHS,
            "assertion": "no forbidden path appears in used_feature_paths; labels/H0_true are read only after feature extraction for evaluation",
        },
        "data": {
            "train": _display(args.train),
            "unfiltered": _display(args.unfiltered),
            "config": _display(args.config),
            "scaler": None if not args.scaler.exists() else _display(args.scaler),
            "target_scaler_mode1": scaler.get("mode1") if scaler else None,
        },
        "filtered_val": filtered,
        "unfiltered_all": unfiltered,
        "acceptance_recommendation": {
            "filtered_h0_r_min": filtered_min,
            "formula": "floor_to_2_decimals(0.80 * filtered_val.h0_space.oracle_r)",
            "ceiling_attainment_band": {
                "fail": "< 0.80",
                "acceptable": "0.80 <= model_r / oracle_r < 0.85",
                "strong": ">= 0.85",
            },
            "basis": "inputs-conditioned oracle H0-space ceiling; threshold fixed before changing round acceptance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fp:
        json.dump(result, fp, indent=2)
    print(json.dumps({
        "output": _display(args.output),
        "filtered_oracle_h0_r": filtered["h0_space"]["oracle_r"],
        "unfiltered_oracle_h0_r": unfiltered["h0_space"]["oracle_r"],
        "filtered_h0_r_min": filtered_min,
        "unfiltered_model_to_oracle_ratio": unfiltered["h0_space"]["model_to_oracle_r_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
