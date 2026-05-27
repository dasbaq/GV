"""Shared Phase 4/5 Mode 1 evaluation utilities.

The functions here are used by both ``CorrectorTrainer.evaluate`` and the
round scripts so target-scaler inversion, calibration coverage, bootstrap CI,
and leak/acceptance checks stay identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from scipy import stats
from scipy.stats import beta as beta_dist


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": rmse(residual),
        "bias": float(np.mean(residual)),
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def ci_overlap(a: list[float], b: list[float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def mode2_correction_availability(path: Path, *, atol: float = 1.0e-8) -> dict:
    """Report whether Mode 2 has nonzero training targets.

    Units: Mode 2 correction target follows the project public order
    ``[theta_E, q, position_angle, sigma_v]``. SIE 표준 근사 가정: this only
    checks data availability and does not wire new Mode 2 inputs. If all labels
    are zero, Mode 2 real training must wait for a non-placeholder catalog.
    """

    candidates = (
        "correction_targets/mode2_dm_correction",
        "simplification_errors/mode2_dm_error",
    )
    with h5py.File(path, "r") as f:
        for key in candidates:
            if key in f:
                arr = np.asarray(f[key][:], dtype=np.float64)
                finite = np.isfinite(arr)
                nonzero = finite & (np.abs(arr) > float(atol))
                return {
                    "path": key,
                    "shape": list(arr.shape),
                    "finite_fraction": float(np.mean(finite)) if arr.size else 0.0,
                    "nonzero_fraction": float(np.mean(nonzero)) if arr.size else 0.0,
                    "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else 0.0,
                    "available_for_real_mode2_training": bool(np.any(nonzero)),
                    "decision": (
                        "mode2_targets_available"
                        if np.any(nonzero)
                        else "데이터 선행 필요: Mode 2 correction targets are all zero placeholders."
                    ),
                }
    return {
        "path": None,
        "shape": None,
        "finite_fraction": 0.0,
        "nonzero_fraction": 0.0,
        "max_abs": 0.0,
        "available_for_real_mode2_training": False,
        "decision": "데이터 선행 필요: no Mode 2 correction target dataset found.",
    }


def _bootstrap_mode1(
    y_true: np.ndarray,
    model_h0: np.ndarray,
    coverage: np.ndarray,
    bootstrap_n: int,
) -> dict:
    rng = np.random.default_rng(20260504)
    boot = {"model_RMSE": [], "model_r": [], "coverage": []}
    n = len(y_true)
    for _ in range(int(bootstrap_n)):
        b = rng.integers(0, n, n)
        boot["model_RMSE"].append(rmse(model_h0[b] - y_true[b]))
        boot["model_r"].append(float(np.corrcoef(y_true[b], model_h0[b])[0, 1]))
        boot["coverage"].append(float(np.mean(coverage[b])))
    return {
        k: {
            "mean": None if len(v) == 0 else float(np.mean(v)),
            "ci95": None if len(v) == 0 else [float(x) for x in np.percentile(v, [2.5, 97.5])],
        }
        for k, v in boot.items()
    }


def evaluate_mode1_h0_on_loader(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    path: Path,
    ids: np.ndarray,
    scaler: dict,
    bootstrap_n: int,
    output_path: Path | None,
    label: str,
    training_summary: dict,
    infra: dict,
    checkpoint_display: str,
    data_display: str,
    scaler_display: str,
    phase4_floor_analysis: dict | None = None,
    move_batch: Callable[[dict, torch.device], dict] | None = None,
) -> dict:
    """Evaluate Mode 1 H0 correction with shared round semantics.

    Units: H0 and corrections are [km/s/Mpc]; predicted sigma is converted back
    to [km/s/Mpc] with the Mode 1 target scaler. SIE 표준 근사 가정:
    corrected H0 is ``H0_approx + correction`` with Phase 4 sign convention
    ``correction = H0_true - H0_approx``.
    """

    model.eval().to(device)
    pred_corr = []
    pred_sigma = []
    with torch.no_grad():
        for batch in loader:
            batch_d = move_batch(batch, device) if move_batch else {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                lc=batch_d["lc"],
                lc_mask=batch_d["lc_mask"],
                params=batch_d["params"],
                sigma_curve=batch_d["sigma_curve"],
                image=batch_d["image"],
                target_mode=batch_d["target_mode"].to(device),
            )
            pred_scaled = out["mode1"]["h0_correction"].detach().cpu().numpy()
            logsig_scaled = out["mode1"]["log_sigma"].detach().cpu().numpy()
            pred_corr.append(pred_scaled * scaler["mode1"]["scale"] + scaler["mode1"]["mean"])
            pred_sigma.append(np.exp(logsig_scaled) * scaler["mode1"]["scale"])

    pred_corr = np.concatenate(pred_corr).astype(np.float64)
    pred_sigma = np.concatenate(pred_sigma).astype(np.float64)
    ids = np.asarray(ids, dtype=int)
    with h5py.File(path, "r") as f:
        y_true = np.asarray(f["true_values/H0_true"][:], dtype=np.float64)[ids]
        h0_approx = np.asarray(f["approx_outputs/H0_approx"][:], dtype=np.float64)[ids]
        h0_all = np.asarray(f["true_values/H0_true"][:], dtype=np.float64)
        true_corr = np.asarray(f["correction_targets/mode1_H0_correction"][:], dtype=np.float64)[ids]

    model_h0 = h0_approx + pred_corr
    residual = model_h0 - y_true
    z = residual / pred_sigma
    coverage = np.abs(z) <= 1.0
    n = len(y_true)
    boot_ci = _bootstrap_mode1(y_true, model_h0, coverage, bootstrap_n)
    k = int(coverage.sum())
    coverage_ci = [
        0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k)),
    ]
    q_levels = stats.norm.cdf(np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64))
    qq = [
        {
            "normal_quantile": int(q),
            "residual_over_pred_sigma_quantile": float(v),
        }
        for q, v in zip([-3, -2, -1, 0, 1, 2, 3], np.quantile(z, q_levels))
    ]
    ks = stats.kstest(h0_all, "uniform", args=(60.0, 20.0))
    pred_positive = pred_corr > 0.0

    result = {
        "eval_set": label,
        "training": training_summary,
        "infrastructure": infra,
        "best_checkpoint": checkpoint_display,
        "data": data_display,
        "scaler": scaler_display,
        "phase4_floor_analysis": phase4_floor_analysis,
        "best": {
            "mode1": {
                "val_m1_scaled_MSE": float(np.mean(((pred_corr - true_corr) / scaler["mode1"]["scale"]) ** 2)),
                "correction_prediction": {
                    "mean": float(pred_corr.mean()),
                    "std": float(pred_corr.std()),
                    "min": float(pred_corr.min()),
                    "max": float(pred_corr.max()),
                    "positive_fraction": float(np.mean(pred_positive)),
                },
                "h0": {
                    "model": regression_metrics(y_true, model_h0),
                    "no_correction": regression_metrics(y_true, h0_approx),
                    "perfect_joint_oracle": regression_metrics(y_true, y_true),
                    "target_sign_note": "Phase 4 HDF5 correction = H0_true - H0_approx, so corrected H0 is H0_approx + model_output.",
                },
                "log_sigma_calibration": {
                    "coverage_abs_residual_le_1sigma": float(np.mean(coverage)),
                    "coverage_abs_residual_le_1sigma_ci95_clopper_pearson": coverage_ci,
                    "coverage_abs_residual_le_2sigma": float(np.mean(np.abs(z) <= 2.0)),
                    "outlier_rate_abs_residual_gt_2sigma": float(np.mean(np.abs(z) > 2.0)),
                    "outlier_rate_abs_residual_gt_3sigma": float(np.mean(np.abs(z) > 3.0)),
                    "predicted_sigma_physical": {
                        "mean": float(pred_sigma.mean()),
                        "std": float(pred_sigma.std()),
                        "min": float(pred_sigma.min()),
                        "max": float(pred_sigma.max()),
                    },
                    "abs_residual_over_pred_sigma": {
                        "mean": float(np.mean(np.abs(z))),
                        "std": float(np.std(np.abs(z))),
                        "min": float(np.min(np.abs(z))),
                        "max": float(np.max(np.abs(z))),
                    },
                },
            }
        },
        "bootstrap": {
            "n": int(bootstrap_n),
            "model_RMSE": boot_ci["model_RMSE"],
            "model_r": boot_ci["model_r"],
            "coverage_1sigma_resample": boot_ci["coverage"],
            "coverage_1sigma_clopper_pearson": {
                "count": k,
                "n": n,
                "point": float(np.mean(coverage)),
                "ci95": coverage_ci,
            },
        },
        "distribution_checks": {
            "qq_residual_over_pred_sigma": qq,
            "val_H0_true_KS_against_U_60_80": {
                "statistic": float(ks.statistic),
                "pvalue": float(ks.pvalue),
            },
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
    return result


def mode1_metric_summary(result: dict) -> dict:
    m = result["best"]["mode1"]["h0"]["model"]
    cal = result["best"]["mode1"]["log_sigma_calibration"]
    return {
        "rmse": float(m["RMSE"]),
        "rmse_ci": result["bootstrap"]["model_RMSE"]["ci95"],
        "r": float(m["r"]),
        "coverage": float(cal["coverage_abs_residual_le_1sigma"]),
        "coverage_ci": cal["coverage_abs_residual_le_1sigma_ci95_clopper_pearson"],
        "positive_fraction": float(result["best"]["mode1"]["correction_prediction"]["positive_fraction"]),
    }


def acceptance_report(
    *,
    filtered: dict,
    unfiltered: dict,
    infra: dict,
    training_summary: dict,
    acceptance: dict,
    leak_triggers_config: dict,
    param_encoder_input_dim: int,
) -> dict:
    """Build shared Mode 1 acceptance and leak report."""

    fm = mode1_metric_summary(filtered)
    um = mode1_metric_summary(unfiltered)
    ratio = float(um["rmse"] / fm["rmse"]) if fm["rmse"] > 0 else float("inf")
    fwd = infra.get("forward_only") or infra.get("equivalence_handoff", {}).get("forward_only", {})
    max_diff = max(fwd.get("diffs", {"missing": float("inf")}).values())
    coverage_target = acceptance["coverage_ci_overlap"]
    pass_rows = [
        {
            "metric": "Kaggle CUDA forward diff (CPU vs CUDA)",
            "value": max_diff,
            "pass": bool(max_diff <= acceptance["cuda_forward_diff_max"]),
            "criterion": f"<= {acceptance['cuda_forward_diff_max']}",
        },
        {
            "metric": "filtered eval model RMSE 95% CI lower",
            "value": None if fm["rmse_ci"] is None else fm["rmse_ci"][0],
            "pass": bool(fm["rmse_ci"] is not None and fm["rmse_ci"][0] > acceptance["filtered_rmse_ci_lower_min"]),
            "criterion": f"> {acceptance['filtered_rmse_ci_lower_min']}",
        },
        {
            "metric": "filtered eval model RMSE 95% CI upper",
            "value": None if fm["rmse_ci"] is None else fm["rmse_ci"][1],
            "pass": bool(fm["rmse_ci"] is not None and fm["rmse_ci"][1] < acceptance["filtered_rmse_ci_upper_max"]),
            "criterion": f"< {acceptance['filtered_rmse_ci_upper_max']}",
        },
        {
            "metric": "filtered eval model RMSE point estimate",
            "value": fm["rmse"],
            "pass": bool(acceptance["filtered_rmse_point_band"][0] <= fm["rmse"] <= acceptance["filtered_rmse_point_band"][1]),
            "criterion": f"inside {acceptance['filtered_rmse_point_band']}",
        },
        {
            "metric": "unfiltered/filtered RMSE ratio",
            "value": ratio,
            "pass": bool(ratio <= acceptance["unfiltered_filtered_rmse_ratio_max"]),
            "criterion": f"<= {acceptance['unfiltered_filtered_rmse_ratio_max']}",
        },
        {
            "metric": "1sigma coverage CI",
            "value": fm["coverage_ci"],
            "pass": bool(ci_overlap(fm["coverage_ci"], coverage_target)),
            "criterion": f"overlap {coverage_target}",
        },
        {
            "metric": "sign of correction predictions (filtered)",
            "value": fm["positive_fraction"],
            "pass": bool(fm["positive_fraction"] >= acceptance["positive_fraction_min"]),
            "criterion": f">= {acceptance['positive_fraction_min']}",
        },
        {
            "metric": "sign of correction predictions (unfiltered)",
            "value": um["positive_fraction"],
            "pass": bool(um["positive_fraction"] >= acceptance["positive_fraction_min"]),
            "criterion": f">= {acceptance['positive_fraction_min']}",
        },
        {
            "metric": "H0 r (model H0 vs true H0, filtered)",
            "value": fm["r"],
            "pass": bool(fm["r"] >= acceptance["filtered_h0_r_min"]),
            "criterion": f">= {acceptance['filtered_h0_r_min']}",
        },
        {
            "metric": "best val_m1",
            "value": training_summary["best_val_m1"],
            "pass": None,
            "criterion": "record only",
        },
    ]
    leak_triggers = {
        "filtered_rmse_ci_upper_below_nfw_oracle_lower": bool(
            fm["rmse_ci"] is not None
            and fm["rmse_ci"][1] < leak_triggers_config["filtered_rmse_ci_upper_below_nfw_oracle_lower"]
        ),
        "unfiltered_rmse_ratio_gt_3p18": bool(ratio > leak_triggers_config["unfiltered_filtered_rmse_ratio_max"]),
        "param_encoder_input_dim_changed": bool(
            leak_triggers_config["param_encoder_input_dim"] != int(param_encoder_input_dim)
        ),
    }
    return {
        "acceptance": acceptance,
        "pass_rows": pass_rows,
        "all_pass_excluding_record_only": all(r["pass"] is not False for r in pass_rows),
        "filtered_metrics": fm,
        "unfiltered_metrics": um,
        "unfiltered_filtered_rmse_ratio": ratio,
        "leak_triggers": leak_triggers,
        "leak_triggered": any(leak_triggers.values()),
    }
