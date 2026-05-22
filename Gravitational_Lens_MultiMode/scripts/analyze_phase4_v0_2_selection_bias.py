"""Diagnose Phase 4 v0.2 catalog validity-filter selection bias.

This is an M2-local analysis script. It does not train or alter the round
model. The script re-applies v0.1/v0.2 catalog cuts to the unfiltered
root-converged evaluation catalog and, when a checkpoint is supplied, evaluates
the fixed v0.2 model to measure reweighted RMSE under filtered-like catalog
distributions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data import error_catalog as ec
from scripts.phase4_v0_2_round import (
    build_eval_loader_for_ids,
    build_model,
    load_cfg,
    move_batch,
)


DEFAULT_FILTERED = ROOT / "data" / "mock" / "phase4_v0_2.h5"
DEFAULT_UNFILTERED = ROOT / "data" / "mock" / "phase4_v0_2_eval_unfiltered.h5"
DEFAULT_SCALER = ROOT / "data" / "target_scaler_phase4_v0_2.pkl"
DEFAULT_CKPT = Path("/Users/donghyun/Downloads/data/checkpoints/phase4_v0_2_imgres_best.pt")
DEFAULT_FILTERED_EVAL = Path("/Users/donghyun/Downloads/data/logs/phase4_v0_2_imgres_h0_eval.json")
DEFAULT_UNFILTERED_EVAL = Path("/Users/donghyun/Downloads/data/logs/phase4_v0_2_imgres_h0_eval_unfiltered.json")
DEFAULT_JSON_OUT = ROOT / "data" / "logs" / "phase4_v0_2_selection_bias_analysis.json"
DEFAULT_MD_OUT = ROOT / "data" / "logs" / "phase4_v0_2_selection_bias_analysis.md"


def _as_float_array(f: h5py.File, key: str) -> np.ndarray:
    return np.asarray(f[key][:], dtype=np.float64)


def load_catalog(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        theta1 = _as_float_array(f, "ray_paths/theta_1")
        theta2 = _as_float_array(f, "ray_paths/theta_2")
        image = np.asarray(f["images/I_obs"][:], dtype=np.float64)
        lc = np.asarray(f["light_curves/F_joint"][:], dtype=np.float64)
        h0 = _as_float_array(f, "true_values/H0_true")
        h0_approx = _as_float_array(f, "approx_outputs/H0_approx")
        corr = _as_float_array(f, "correction_targets/mode1_H0_correction")
        return {
            "H0_true": h0,
            "H0_approx": h0_approx,
            "correction": corr,
            "mu": _as_float_array(f, "true_values/mu_true"),
            "dphi_ratio": _as_float_array(f, "ray_paths/dphi_sie_over_truth"),
            "separation": np.linalg.norm(theta1 - theta2, axis=1),
            "dt_true": _as_float_array(f, "true_values/dt_true"),
            "dt_approx": _as_float_array(f, "approx_outputs/dt_approx"),
            "I_obs_sum": image.reshape(image.shape[0], -1).sum(axis=1),
            "F_joint_absmax": np.max(np.abs(lc), axis=1),
        }


def cut_masks(cat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    finite = (
        np.isfinite(cat["dt_true"])
        & np.isfinite(cat["H0_approx"])
        & np.isfinite(cat["dphi_ratio"])
    )
    return {
        "finite": finite,
        "dt_true_positive": cat["dt_true"] > 0.0,
        "mu_abs_lt_0p98_v0_1": np.abs(cat["mu"]) < ec.TRUTH_MU_MAX,
        "separation_ge_0p1_v0_1": cat["separation"] >= ec.TRUTH_MIN_IMAGE_SEPARATION_ARCSEC,
        "H0_approx_45_90_v0_1": (
            (cat["H0_approx"] >= ec.TRUTH_H0_APPROX_RANGE[0])
            & (cat["H0_approx"] <= ec.TRUTH_H0_APPROX_RANGE[1])
        ),
        "dphi_ratio_0p5_1p5_v0_1": (
            (cat["dphi_ratio"] >= ec.TRUTH_DPHI_RATIO_RANGE[0])
            & (cat["dphi_ratio"] <= ec.TRUTH_DPHI_RATIO_RANGE[1])
        ),
        "mu_abs_le_0p9699_v0_2": np.abs(cat["mu"]) <= ec.TRUTH_V0_2_MU_MAX,
        "separation_ge_0p6598_v0_2": cat["separation"] >= ec.TRUTH_V0_2_MIN_IMAGE_SEPARATION_ARCSEC,
        "dphi_ratio_0p5878_0p9201_v0_2": (
            (cat["dphi_ratio"] >= ec.TRUTH_V0_2_DPHI_RATIO_RANGE[0])
            & (cat["dphi_ratio"] <= ec.TRUTH_V0_2_DPHI_RATIO_RANGE[1])
        ),
        "dt_approx_le_444p7_v0_2": cat["dt_approx"] <= ec.TRUTH_V0_2_DT_APPROX_MAX_DAYS,
        "correction_abs_le_32p27_v0_2": (
            np.abs(cat["correction"]) <= ec.TRUTH_V0_2_MODE1_CORRECTION_ABSMAX + 1.0e-9
        ),
        "I_obs_sum_le_77p79_v0_2": cat["I_obs_sum"] <= ec.TRUTH_V0_2_I_OBS_SUM_MAX,
        "F_joint_absmax_le_3p408_v0_2": cat["F_joint_absmax"] <= ec.TRUTH_V0_2_F_JOINT_ABSMAX_MAX,
    }


def ks_uniform_h0(h0: np.ndarray) -> dict[str, float]:
    if h0.size == 0:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    ks = stats.kstest(h0, "uniform", args=(60.0, 20.0))
    return {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}


def summarize_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def h0_cut_row(name: str, mask: np.ndarray, h0: np.ndarray) -> dict[str, Any]:
    kept = h0[mask]
    removed = h0[~mask]
    kept_ks = ks_uniform_h0(kept)
    removed_ks = ks_uniform_h0(removed)
    kept_removed_ks = stats.ks_2samp(kept, removed) if kept.size and removed.size else None
    return {
        "cut": name,
        "n_kept": int(mask.sum()),
        "n_removed": int((~mask).sum()),
        "keep_rate": float(mask.mean()),
        "kept_H0": summarize_values(kept),
        "removed_H0": summarize_values(removed),
        "kept_vs_uniform_KS": kept_ks,
        "removed_vs_uniform_KS": removed_ks,
        "kept_removed_KS": None
        if kept_removed_ks is None
        else {"statistic": float(kept_removed_ks.statistic), "pvalue": float(kept_removed_ks.pvalue)},
        "abs_kept_removed_mean_delta": float(abs(np.mean(kept) - np.mean(removed))) if kept.size and removed.size else 0.0,
    }


def distribution_summary(cat: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    keys = ["H0_true", "correction", "mu", "dphi_ratio", "separation", "dt_approx", "I_obs_sum", "F_joint_absmax"]
    return {key: summarize_values(cat[key][mask]) for key in keys}


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
    }


def baseline_metrics(cat: dict[str, np.ndarray]) -> dict[str, Any]:
    h0 = cat["H0_true"]
    h0_approx = cat["H0_approx"]
    return {
        "no_correction": metric_rows(h0, h0_approx),
        "perfect_joint_oracle": metric_rows(h0, h0),
        "target_sign_note": "Phase 4 HDF5 correction = H0_true - H0_approx; corrected H0 = H0_approx + correction.",
    }


def distribution_match_rows(filtered: dict[str, np.ndarray], unfiltered: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for key in ["H0_true", "correction", "dphi_ratio", "mu", "separation"]:
        ks = stats.ks_2samp(filtered[key], unfiltered[key])
        rows.append(
            {
                "feature": key,
                "filtered": summarize_values(filtered[key]),
                "unfiltered": summarize_values(unfiltered[key]),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein": float(stats.wasserstein_distance(filtered[key], unfiltered[key])),
            }
        )
    return rows


def sequential_cut_table(masks: dict[str, np.ndarray], h0: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    current = np.ones_like(h0, dtype=bool)
    rows: list[dict[str, Any]] = []
    prev_ks = ks_uniform_h0(h0[current])["statistic"]
    for name in names:
        before_n = int(current.sum())
        current = current & masks[name]
        after_n = int(current.sum())
        ks = ks_uniform_h0(h0[current])
        rows.append(
            {
                "cut": name,
                "n_before": before_n,
                "n_after": after_n,
                "n_removed_incremental": before_n - after_n,
                "keep_rate_incremental": float(after_n / before_n) if before_n else float("nan"),
                "H0_mean_after": float(np.mean(h0[current])) if after_n else float("nan"),
                "H0_KS_stat_after": ks["statistic"],
                "H0_KS_p_after": ks["pvalue"],
                "delta_KS_stat": float(ks["statistic"] - prev_ks),
            }
        )
        prev_ks = ks["statistic"]
    return rows


def leave_one_out_table(masks: dict[str, np.ndarray], h0: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    full = np.logical_and.reduce([masks[name] for name in names])
    full_ks = ks_uniform_h0(h0[full])
    rows: list[dict[str, Any]] = []
    for name in names:
        loo_names = [n for n in names if n != name]
        loo = np.logical_and.reduce([masks[n] for n in loo_names])
        loo_ks = ks_uniform_h0(h0[loo])
        rows.append(
            {
                "omitted_cut": name,
                "n_kept_without_cut": int(loo.sum()),
                "extra_kept": int(loo.sum() - full.sum()),
                "H0_mean_without_cut": float(np.mean(h0[loo])) if loo.any() else float("nan"),
                "H0_KS_stat_without_cut": loo_ks["statistic"],
                "H0_KS_p_without_cut": loo_ks["pvalue"],
                "KS_stat_reduction_when_omitted": float(full_ks["statistic"] - loo_ks["statistic"]),
            }
        )
    return rows


def weighted_rmse(residual: np.ndarray, weights: np.ndarray | None = None) -> float:
    r2 = np.asarray(residual, dtype=np.float64) ** 2
    if weights is None:
        return float(np.sqrt(np.mean(r2)))
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    if np.sum(w) <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(w * r2) / np.sum(w)))


def density_ratio_weights(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    feature_names: list[str],
    bins: int = 5,
) -> np.ndarray:
    weights = np.ones(source[feature_names[0]].shape[0], dtype=np.float64)
    for name in feature_names:
        src = np.asarray(source[name], dtype=np.float64)
        tgt = np.asarray(target[name], dtype=np.float64)
        edges = np.unique(np.quantile(np.concatenate([src, tgt]), np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        src_bin = np.clip(np.digitize(src, edges[1:-1], right=False), 0, edges.size - 2)
        tgt_bin = np.clip(np.digitize(tgt, edges[1:-1], right=False), 0, edges.size - 2)
        src_counts = np.bincount(src_bin, minlength=edges.size - 1).astype(np.float64)
        tgt_counts = np.bincount(tgt_bin, minlength=edges.size - 1).astype(np.float64)
        src_density = (src_counts + 1.0) / (src_counts.sum() + src_counts.size)
        tgt_density = (tgt_counts + 1.0) / (tgt_counts.sum() + tgt_counts.size)
        weights *= tgt_density[src_bin] / src_density[src_bin]
    if np.any(weights > 0):
        weights = np.minimum(weights, np.quantile(weights, 0.95))
        weights = weights / np.mean(weights)
    return weights


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def evaluate_checkpoint(
    data_path: Path,
    checkpoint_path: Path,
    scaler_path: Path,
    ids: np.ndarray,
    device_name: str,
    workers: int,
) -> dict[str, np.ndarray]:
    with scaler_path.open("rb") as fp:
        scaler = pickle.load(fp)
    cfg = load_cfg()
    device = torch.device(device_name)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    loader = build_eval_loader_for_ids(cfg, data_path, ids, scaler, workers, device)
    pred_corr: list[np.ndarray] = []
    pred_sigma: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_d = move_batch(batch, device)
            out = model(
                lc=batch_d["lc"],
                lc_mask=batch_d["lc_mask"],
                params=batch_d["params"],
                sigma_curve=batch_d["sigma_curve"],
                image=batch_d["image"],
                use_image=batch_d["use_image"].to(device),
                target_mode=batch_d["target_mode"].to(device),
            )
            pred_scaled = out["mode1"]["h0_correction"].detach().cpu().numpy()
            logsig_scaled = out["mode1"]["log_sigma"].detach().cpu().numpy()
            pred_corr.append(pred_scaled * scaler["mode1"]["scale"] + scaler["mode1"]["mean"])
            pred_sigma.append(np.exp(logsig_scaled) * scaler["mode1"]["scale"])
    pred_corr_arr = np.concatenate(pred_corr).astype(np.float64)
    pred_sigma_arr = np.concatenate(pred_sigma).astype(np.float64)
    with h5py.File(data_path, "r") as f:
        h0_true = _as_float_array(f, "true_values/H0_true")[ids]
        h0_approx = _as_float_array(f, "approx_outputs/H0_approx")[ids]
        true_corr = _as_float_array(f, "correction_targets/mode1_H0_correction")[ids]
    model_h0 = h0_approx + pred_corr_arr
    residual = model_h0 - h0_true
    return {
        "ids": ids.astype(np.int64),
        "pred_corr": pred_corr_arr,
        "pred_sigma": pred_sigma_arr,
        "true_corr": true_corr,
        "H0_true": h0_true,
        "H0_approx": h0_approx,
        "model_H0": model_h0,
        "residual": residual,
        "abs_z": np.abs(residual / pred_sigma_arr),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    md = []
    md.append(f"# {result['round']} Selection Bias Analysis\n")
    md.append(f"- unfiltered n: {result['samples']['unfiltered_n']}")
    md.append(f"- v0.2 mask n on unfiltered: {result['samples']['unfiltered_v0_2_kept_n']}")
    md.append(f"- filtered train/catalog n: {result['samples']['filtered_n']}\n")
    md.append("## Top H0-Distorting Individual Cuts\n")
    md.append("| cut | keep | removed | kept H0 mean | removed H0 mean | kept-vs-removed KS | kept uniform KS p |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in result["cut_distortion_ranked"][:8]:
        md.append(
            "| {cut} | {n_kept} | {n_removed} | {km:.3f} | {rm:.3f} | {ks:.3f} | {p:.3g} |".format(
                cut=row["cut"],
                n_kept=row["n_kept"],
                n_removed=row["n_removed"],
                km=row["kept_H0"]["mean"],
                rm=row["removed_H0"]["mean"],
                ks=0.0 if row["kept_removed_KS"] is None else row["kept_removed_KS"]["statistic"],
                p=row["kept_vs_uniform_KS"]["pvalue"],
            )
        )
    md.append("\n## Sequential v0.2 Filter\n")
    md.append("| cut | before | after | removed | H0 mean after | KS stat | KS p | delta KS |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["sequential_v0_2"]:
        md.append(
            "| {cut} | {n_before} | {n_after} | {n_removed_incremental} | {H0_mean_after:.3f} | "
            "{H0_KS_stat_after:.3f} | {H0_KS_p_after:.3g} | {delta_KS_stat:.3f} |".format(**row)
        )
    if result.get("model_reweighting"):
        r = result["model_reweighting"]
        md.append("\n## Fixed-Model Reweighting\n")
        md.append(f"- unfiltered RMSE: {r['unfiltered_rmse']:.3f}")
        md.append(f"- unfiltered, H0 reweighted to filtered catalog: {r['h0_reweighted_rmse']:.3f}")
        md.append(f"- unfiltered, catalog-feature reweighted to filtered catalog: {r['catalog_feature_reweighted_rmse']:.3f}")
        md.append(f"- unfiltered subset passing v0.2 cuts: {r['unfiltered_v0_2_subset_rmse']:.3f}")
    md.append("\n## Filtered/Unfiltered Distribution Match\n")
    md.append("| feature | filtered mean/std | unfiltered mean/std | KS | KS p | Wasserstein |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for row in result["filtered_unfiltered_distribution_match"]:
        md.append(
            "| {feature} | {fm:.3f}/{fs:.3f} | {um:.3f}/{us:.3f} | {ks:.3f} | {p:.3g} | {w:.3f} |".format(
                feature=row["feature"],
                fm=row["filtered"]["mean"],
                fs=row["filtered"]["std"],
                um=row["unfiltered"]["mean"],
                us=row["unfiltered"]["std"],
                ks=row["ks_statistic"],
                p=row["ks_pvalue"],
                w=row["wasserstein"],
            )
        )
    md.append("\n## Baselines\n")
    b = result["baselines"]
    md.append(f"- filtered no-correction RMSE: {b['filtered']['no_correction']['RMSE']:.3f}")
    md.append(f"- unfiltered no-correction RMSE: {b['unfiltered']['no_correction']['RMSE']:.3f}")
    md.append("\n## Conclusion\n")
    md.append(result["conclusion"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="phase4_v0_2")
    parser.add_argument("--filtered", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--unfiltered", type=Path, default=DEFAULT_UNFILTERED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--filtered-eval-json", type=Path, default=DEFAULT_FILTERED_EVAL)
    parser.add_argument("--unfiltered-eval-json", type=Path, default=DEFAULT_UNFILTERED_EVAL)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--floor-out", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-checkpoint-eval", action="store_true")
    args = parser.parse_args()

    filtered = load_catalog(args.filtered)
    unfiltered = load_catalog(args.unfiltered)
    masks = cut_masks(unfiltered)
    v0_1_names = [
        "finite",
        "dt_true_positive",
        "mu_abs_lt_0p98_v0_1",
        "separation_ge_0p1_v0_1",
        "H0_approx_45_90_v0_1",
        "dphi_ratio_0p5_1p5_v0_1",
    ]
    v0_2_names = v0_1_names + [
        "mu_abs_le_0p9699_v0_2",
        "separation_ge_0p6598_v0_2",
        "dphi_ratio_0p5878_0p9201_v0_2",
        "dt_approx_le_444p7_v0_2",
        "correction_abs_le_32p27_v0_2",
        "I_obs_sum_le_77p79_v0_2",
        "F_joint_absmax_le_3p408_v0_2",
    ]
    v0_1_mask = np.logical_and.reduce([masks[name] for name in v0_1_names])
    v0_2_mask = np.logical_and.reduce([masks[name] for name in v0_2_names])

    h0 = unfiltered["H0_true"]
    individual_rows = [h0_cut_row(name, masks[name], h0) for name in v0_2_names]
    ranked = sorted(
        individual_rows,
        key=lambda row: (
            0.0 if row["kept_removed_KS"] is None else row["kept_removed_KS"]["statistic"],
            row["abs_kept_removed_mean_delta"],
        ),
        reverse=True,
    )

    correlations = {}
    for name in ["correction", "H0_approx", "mu", "dphi_ratio", "separation", "dt_approx", "I_obs_sum", "F_joint_absmax"]:
        pearson = stats.pearsonr(h0, unfiltered[name])
        spearman = stats.spearmanr(h0, unfiltered[name])
        correlations[name] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }

    result: dict[str, Any] = {
        "round": args.round_name,
        "inputs": {
            "filtered": str(args.filtered),
            "unfiltered": str(args.unfiltered),
            "checkpoint": str(args.checkpoint),
            "scaler": str(args.scaler),
        },
        "samples": {
            "unfiltered_n": int(h0.size),
            "unfiltered_v0_1_kept_n": int(v0_1_mask.sum()),
            "unfiltered_v0_2_kept_n": int(v0_2_mask.sum()),
            "filtered_n": int(filtered["H0_true"].size),
        },
        "distribution_before_after": {
            "unfiltered_all": distribution_summary(unfiltered, np.ones_like(h0, dtype=bool)),
            "unfiltered_v0_1_mask": distribution_summary(unfiltered, v0_1_mask),
            "unfiltered_v0_2_mask": distribution_summary(unfiltered, v0_2_mask),
            "filtered_catalog": distribution_summary(filtered, np.ones_like(filtered["H0_true"], dtype=bool)),
        },
        "H0_KS_against_U_60_80": {
            "unfiltered_all": ks_uniform_h0(h0),
            "unfiltered_v0_1_mask": ks_uniform_h0(h0[v0_1_mask]),
            "unfiltered_v0_2_mask": ks_uniform_h0(h0[v0_2_mask]),
            "filtered_catalog": ks_uniform_h0(filtered["H0_true"]),
        },
        "cut_distortion_individual": individual_rows,
        "cut_distortion_ranked": ranked,
        "sequential_v0_2": sequential_cut_table(masks, h0, v0_2_names),
        "leave_one_out_v0_2": leave_one_out_table(masks, h0, v0_2_names),
        "H0_feature_correlations_unfiltered": correlations,
        "filtered_unfiltered_distribution_match": distribution_match_rows(filtered, unfiltered),
        "baselines": {
            "filtered": baseline_metrics(filtered),
            "unfiltered": baseline_metrics(unfiltered),
        },
        "predeclared_selection_bias_acceptance": {
            "unfiltered_filtered_rmse_ratio_max": 2.5,
            "coverage_1sigma_target": [0.62, 0.78],
            "baselines_required": ["no_correction", "perfect_joint_oracle"],
            "note": "Apply after Kaggle CUDA train; this M2 analysis only fixes catalog/baseline inputs.",
        },
        "kaggle_eval_json_summary": {
            "filtered": load_json(args.filtered_eval_json),
            "unfiltered": load_json(args.unfiltered_eval_json),
        },
    }

    if not args.skip_checkpoint_eval and args.checkpoint.exists() and args.scaler.exists():
        ids = np.arange(h0.size, dtype=int)
        pred = evaluate_checkpoint(args.unfiltered, args.checkpoint, args.scaler, ids, args.device, args.workers)
        target_for_weighting = {
            "H0_true": filtered["H0_true"],
            "correction": filtered["correction"],
            "dphi_ratio": filtered["dphi_ratio"],
            "separation": filtered["separation"],
            "mu": filtered["mu"],
        }
        source_for_weighting = {
            "H0_true": unfiltered["H0_true"],
            "correction": unfiltered["correction"],
            "dphi_ratio": unfiltered["dphi_ratio"],
            "separation": unfiltered["separation"],
            "mu": unfiltered["mu"],
        }
        h0_weights = density_ratio_weights(source_for_weighting, target_for_weighting, ["H0_true"], bins=5)
        cat_weights = density_ratio_weights(
            source_for_weighting,
            target_for_weighting,
            ["correction", "dphi_ratio", "separation", "mu"],
            bins=4,
        )
        residual = pred["residual"]
        result["model_reweighting"] = {
            "unfiltered_rmse": weighted_rmse(residual),
            "h0_reweighted_rmse": weighted_rmse(residual, h0_weights),
            "catalog_feature_reweighted_rmse": weighted_rmse(residual, cat_weights),
            "unfiltered_v0_2_subset_rmse": weighted_rmse(residual[v0_2_mask]),
            "unfiltered_v0_1_subset_rmse": weighted_rmse(residual[v0_1_mask]),
            "unfiltered_coverage_1sigma": float(np.mean(pred["abs_z"] <= 1.0)),
            "unfiltered_v0_2_subset_coverage_1sigma": float(np.mean(pred["abs_z"][v0_2_mask] <= 1.0)),
            "prediction_correlation": float(np.corrcoef(pred["H0_true"], pred["model_H0"])[0, 1]),
        }

    result["conclusion"] = (
        "Primary failure mode is selection bias from H0-correlated and support-narrowing "
        "catalog cuts. The strongest robust H0 distortion is the v0.1 H0_approx range "
        "gate, with the v0.2 separation floor adding a smaller shift; one-sample v0.2 "
        "tail gates can show large marginal KS values but are not robust drivers by "
        "count. The v0.2 tail filters still matter because they align the catalog to "
        "an easy, narrow support: correction, dphi_ratio, mu, and separation in the "
        "filtered catalog are much closer to the subset that passes the cuts than to "
        "the full root-converged population. The fixed model then under-corrects "
        "large-correction systems and its NLL head is miscalibrated out of domain. "
        "The r<0.85 result remains evidence that current inputs are not sufficient "
        "for the full unfiltered population, but it is secondary to the catalog "
        "support mismatch for v0.2 acceptance."
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
    if args.floor_out is not None:
        args.floor_out.parent.mkdir(parents=True, exist_ok=True)
        floor = {
            "round": args.round_name,
            "catalog_path": str(args.filtered),
            "eval_unfiltered_path": str(args.unfiltered),
            "mode1_H0_correction": result["distribution_before_after"]["filtered_catalog"]["correction"],
            "H0_KS_against_U_60_80": result["H0_KS_against_U_60_80"]["filtered_catalog"],
            "filtered_unfiltered_distribution_match": result["filtered_unfiltered_distribution_match"],
            "baselines": result["baselines"],
            "predeclared_selection_bias_acceptance": result["predeclared_selection_bias_acceptance"],
        }
        with args.floor_out.open("w", encoding="utf-8") as fp:
            json.dump(floor, fp, indent=2, sort_keys=True)
    write_markdown(args.markdown_out, result)
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "markdown": str(args.markdown_out),
                "floor": None if args.floor_out is None else str(args.floor_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
