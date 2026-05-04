"""Phase 3 v2.6 infrastructure equivalence, retrain, and bootstrap eval.

This script intentionally keeps the model, labels, inputs, dataset, batch size,
and optimizer hyperparameters unchanged from the v2.5 round.  It only controls
device placement, DataLoader worker count, run bookkeeping, and evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy import stats
from scipy.stats import beta as beta_dist
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.dataset import LensCorrectionDataset
from ml.training.losses import composite_loss
from ml.utils.seed import set_seed
from scripts.lib.round_common import (
    add_phase_args,
    build_round_paths,
    dataloader_kwargs,
    default_worker_candidate,
    display_path as _display_path,
    load_equivalence_payload,
    select_device,
    should_run_acceptance,
    skipped_acceptance,
)

DATA_NAME = "real_phase3_v2_6.h5"
PATHS = build_round_paths(
    root=ROOT,
    data_name=DATA_NAME,
    round_name="phase3_v2_6_round",
    checkpoint_name="phase3_v2_6_imgres_best.pt",
    history_name="phase3_v2_6_imgres_long_history.json",
    eval_name="phase3_v2_6_imgres_h0_eval.json",
    infra_name="phase3_v2_6_infra_equivalence.json",
    scaler_name="target_scaler_phase3_v2_6.pkl",
)
DATA = PATHS.data
SCALER = PATHS.scaler
CKPT = PATHS.checkpoint
HISTORY = PATHS.history
EVAL = PATHS.eval
INFRA = PATHS.infra
RUNS = PATHS.runs
EQUIVALENCE = PATHS.work_root / "logs" / "phase3_v2_6_equivalence.json"


def display_path(path: Path) -> str:
    return _display_path(path, ROOT)


def load_cfg(epochs: int | None = None) -> dict:
    with open(ROOT / "config" / "ml.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["epochs"] = 50 if epochs is None else int(epochs)
    cfg["training"]["early_stop_patience"] = 8
    return cfg


def build_model(cfg: dict) -> MultiModalErrorCorrector:
    param_norm = cfg["data"]["param_normalization"]
    model_cfg = dict(cfg["model"])
    model_cfg["param_in_dim"] = len(param_norm) + 5
    model_cfg["image_size"] = cfg["data"]["image_size"]
    return MultiModalErrorCorrector(model_cfg)


def build_dataset(cfg: dict, split: str, modes: list[int], approx_levels: list[int],
                  scaler: dict | None, seed: int) -> LensCorrectionDataset:
    return LensCorrectionDataset(
        h5_paths=[DATA],
        split=split,
        modes=tuple(modes),
        approx_levels=tuple(approx_levels),
        max_len=cfg["data"]["max_lc_len"],
        sigma_curve_size=cfg["data"]["sigma_curve_size"],
        image_size=cfg["data"]["image_size"],
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=cfg["data"]["param_normalization"],
        target_scaler=scaler,
        seed=seed,
    )


def build_loader(cfg: dict, split: str, modes: list[int], approx_levels: list[int],
                 scaler: dict | None, seed: int, workers: int, device: torch.device,
                 train: bool, batch_size: int | None = None) -> DataLoader:
    ds = build_dataset(cfg, split, modes, approx_levels, scaler, seed)
    kwargs = dataloader_kwargs(
        batch_size=batch_size or cfg["training"]["batch_size"],
        workers=workers,
        device=device,
        train=train,
    )
    if train:
        w_map = {1: 1.0, 2: 1.0, 3: 1.0}
        weights = torch.tensor([w_map[e[3]] for e in ds._index], dtype=torch.float)
        gen = torch.Generator()
        gen.manual_seed(seed)
        kwargs["sampler"] = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=gen
        )
    return DataLoader(ds, **kwargs)


def split_system_ids(path: Path, split: str, seed: int) -> np.ndarray:
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
    return ids[n_train + n_val:]


def create_target_scaler(path: Path, out_path: Path, seed: int) -> dict:
    ids = split_system_ids(path, "train", seed)
    with h5py.File(path, "r") as f:
        mode1 = np.asarray(f["simplification_errors/mode1_H0_error"][:], dtype=np.float32)[ids]
        mode3 = np.asarray(f["simplification_errors/mode3_source_residual"][:], dtype=np.float32)[ids]
    scaler = {
        "mode1": {"mean": float(mode1.mean()), "scale": float(mode1.std() + 1.0e-8)},
        "mode2": {
            "mean": np.zeros(4, dtype=np.float32),
            "scale": np.ones(4, dtype=np.float32),
        },
        "mode3": {"mean": float(mode3.mean()), "scale": float(mode3.std() + 1.0e-8)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def load_scaler() -> dict:
    with open(SCALER, "rb") as f:
        return pickle.load(f)


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def run_forward(model: torch.nn.Module, batch: dict, device: torch.device) -> dict[str, np.ndarray]:
    model.eval().to(device)
    batch = move_batch(batch, device)
    with torch.no_grad():
        out = model(
            lc=batch["lc"],
            lc_mask=batch["lc_mask"],
            params=batch["params"],
            sigma_curve=batch["sigma_curve"],
            image=batch["image"],
            use_image=batch["use_image"].to(device),
            target_mode=batch["target_mode"].to(device),
        )
    arrays: dict[str, np.ndarray] = {}
    if out["mode1"] is not None:
        arrays["mode1_pred"] = out["mode1"]["h0_correction"].detach().cpu().numpy()
        arrays["log_sigma_mode1"] = out["mode1"]["log_sigma"].detach().cpu().numpy()
    if out["mode2"] is not None:
        arrays["mode2_pred"] = out["mode2"]["dm_correction"].detach().cpu().numpy()
        arrays["log_sigma_mode2"] = out["mode2"]["log_sigma"].detach().cpu().numpy()
    if out["mode3"] is not None:
        arrays["mode3_pred"] = out["mode3"]["source_residual"].detach().cpu().numpy()
    return arrays


def forward_equivalence(cfg: dict, scaler: dict, target_device_name: str) -> dict:
    set_seed(42)
    cpu = torch.device("cpu")
    target_device = select_device(target_device_name)
    if target_device.type == "cpu":
        raise RuntimeError("Forward equivalence needs a non-CPU target device.")
    loader = build_loader(cfg, "val", [1, 2, 3], [1, 2], scaler, 42, 0, cpu, False, batch_size=6)
    batch = next(iter(loader))
    model_cpu = build_model(cfg)
    state = copy.deepcopy(model_cpu.state_dict())
    cpu_out = run_forward(model_cpu, batch, cpu)
    model_target = build_model(cfg)
    model_target.load_state_dict(state)
    target_out = run_forward(model_target, batch, target_device)
    diffs = {}
    for key in cpu_out:
        diffs[key] = float(np.max(np.abs(cpu_out[key] - target_out[key])))
    passed = bool(max(diffs.values()) <= 1.0e-4)
    return {
        "target_device": target_device.type,
        "threshold": 1.0e-4,
        "diffs": diffs,
        "passed": passed,
    }


def train_one(cfg: dict, seed: int, device_name: str, workers: int,
              scaler: dict, epochs: int = 1, output_ckpt: Path | None = None,
              criterion: str = "mode1_task", patience: int | None = None) -> dict:
    set_seed(seed)
    device = select_device(device_name)
    model = build_model(cfg).to(device)
    train_loader = build_loader(cfg, "train", [1, 2, 3], [1, 2], scaler, seed, workers, device, True)
    val_loader = build_loader(cfg, "val", [1, 2, 3], [1, 2], scaler, seed, workers, device, False)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    warmup = cfg["training"]["warmup_epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(warmup, 1))
        progress = float(epoch - warmup) / float(max(epochs - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = cfg["training"]["early_stop_patience"] if patience is None else patience
    best_val = math.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    total_t0 = time.time()

    for epoch in range(epochs):
        epoch_t0 = time.time()
        tr = run_epoch(model, train_loader, opt, scaler_amp, cfg, device, True, use_amp)
        vl = run_epoch(model, val_loader, opt, scaler_amp, cfg, device, False, use_amp)
        if tr.get("optimizer_step_happened", False):
            scheduler.step()
        wall_time = time.time() - epoch_t0
        entry = {
            "epoch": epoch + 1,
            "train": tr,
            "val": vl,
            "train_loss": tr["total"],
            "val_loss": vl["total"],
            "wall_time": wall_time,
            "lr": opt.param_groups[0]["lr"],
            "nan_detected": bool(tr.get("nan_detected", False) or vl.get("nan_detected", False)),
            "nan_batches": {
                "train": tr.get("nan_batches", []),
                "val": vl.get("nan_batches", []),
            },
            "last_grad_norm": tr.get("last_grad_norm"),
            "last_param_norm": tr.get("last_param_norm"),
        }
        history.append(entry)
        val_key = vl[criterion]
        print(
            f"{device_name} workers={workers} seed={seed} "
            f"epoch {epoch + 1:02d}/{epochs} train={tr['total']:.4f} "
            f"val={vl['total']:.4f} val_m1={vl['mode1_task']:.4f} ({wall_time:.1f}s)",
            flush=True,
        )
        if val_key < best_val:
            best_val = val_key
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            if output_ckpt is not None:
                output_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, output_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        if entry["nan_detected"] and len(history) >= 2 and history[-2].get("nan_detected"):
            print("NaN detected in two consecutive epochs; stopping without tuning.", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "device": device_name,
        "workers": workers,
        "seed": seed,
        "epochs_requested": epochs,
        "ended_epoch": len(history),
        "best_epoch": best_epoch,
        "best_val_m1": float(min(h["val"]["mode1_task"] for h in history)),
        "best_val_criterion": float(best_val),
        "wall_time_total": time.time() - total_t0,
        "history": history,
    }


def run_epoch(model: torch.nn.Module, loader: DataLoader, opt, scaler_amp, cfg: dict,
              device: torch.device, train: bool, use_amp: bool) -> dict:
    model.train(train)
    agg: dict[str, float] = {}
    n = 0
    nan_batches: list[int] = []
    last_grad_norm: float | None = None
    last_param_norm: float | None = None
    optimizer_step_happened = False
    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(
                    lc=batch["lc"],
                    lc_mask=batch["lc_mask"],
                    params=batch["params"],
                    sigma_curve=batch["sigma_curve"],
                    image=batch["image"],
                    use_image=batch["use_image"].to(device),
                    target_mode=batch["target_mode"].to(device),
                )
                losses = composite_loss(pred, batch, cfg["training"]["loss_weights"])
            if not torch.isfinite(losses["total"]):
                nan_batches.append(batch_idx)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
            n += 1
            if nan_batches and nan_batches[-1] == batch_idx:
                continue
            if train:
                opt.zero_grad(set_to_none=True)
                scaler_amp.scale(losses["total"]).backward()
                scaler_amp.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                last_grad_norm = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                scale_before = scaler_amp.get_scale() if use_amp else None
                scaler_amp.step(opt)
                scaler_amp.update()
                scale_after = scaler_amp.get_scale() if use_amp else None
                optimizer_step_happened = optimizer_step_happened or not (
                    use_amp and scale_after is not None and scale_before is not None and scale_after < scale_before
                )
                with torch.no_grad():
                    total_sq = torch.tensor(0.0, device=device)
                    for p in model.parameters():
                        total_sq = total_sq + p.detach().float().pow(2).sum()
                    last_param_norm = float(torch.sqrt(total_sq).detach().cpu())
    result = {k: v / max(n, 1) for k, v in agg.items()}
    result.update({
        "nan_detected": bool(nan_batches),
        "nan_batches": nan_batches,
        "last_grad_norm": last_grad_norm,
        "last_param_norm": last_param_norm,
        "optimizer_step_happened": optimizer_step_happened,
    })
    return result


def distribution_equivalence(cfg: dict, scaler: dict, target_device_name: str,
                             target_workers: int) -> dict:
    rows = []
    target_device = select_device(target_device_name)
    vals = {"cpu": [], target_device.type: []}
    for seed in [42, 1337, 7]:
        for dev, workers in [("cpu", 0), (target_device.type, target_workers)]:
            res = train_one(cfg, seed, dev, workers=workers, scaler=scaler, epochs=1, patience=99)
            val_m1 = float(res["history"][-1]["val"]["mode1_task"])
            rows.append({"seed": seed, "device": dev, "num_workers": workers, "val_m1": val_m1})
            vals[dev].append(val_m1)
    cpu = np.asarray(vals["cpu"], dtype=np.float64)
    target = np.asarray(vals[target_device.type], dtype=np.float64)
    t = stats.ttest_ind(cpu, target, equal_var=False)
    cpu_mean = float(cpu.mean())
    cpu_std = float(cpu.std(ddof=1))
    target_mean = float(target.mean())
    band = [cpu_mean - 2.0 * cpu_std, cpu_mean + 2.0 * cpu_std]
    return {
        "target_device": target_device.type,
        "rows": rows,
        "cpu_mean": cpu_mean,
        "cpu_std": cpu_std,
        f"{target_device.type}_mean": target_mean,
        f"{target_device.type}_std": float(target.std(ddof=1)),
        "cpu_mean_pm_2std": band,
        "welch_p": float(t.pvalue),
        "passed": bool(band[0] <= target_mean <= band[1] and t.pvalue > 0.05),
    }


def speed_table(cfg: dict, scaler: dict, target_device_name: str,
                worker_candidate: int, forced_workers: int | None) -> dict:
    target_device = select_device(target_device_name)
    cases = [("cpu", 0), (target_device.type, 0)]
    if worker_candidate > 0:
        cases.append((target_device.type, worker_candidate))
    rows = []
    for dev, workers in cases:
        res = train_one(cfg, 42, dev, workers=workers, scaler=scaler, epochs=1, patience=99)
        rows.append({
            "case": "CPU" if dev == "cpu" else (
                f"{target_device.type.upper()}+workers" if workers else target_device.type.upper()
            ),
            "device": dev,
            "num_workers": workers,
            "wall_time_sec": float(res["history"][-1]["wall_time"]),
            "val_m1": float(res["history"][-1]["val"]["mode1_task"]),
        })
    cpu_time = rows[0]["wall_time_sec"]
    target0 = rows[1]["wall_time_sec"]
    rows[1]["speedup_vs_cpu"] = cpu_time / target0
    if len(rows) > 2:
        targetw = rows[2]["wall_time_sec"]
        rows[2]["speedup_vs_cpu"] = cpu_time / targetw
        worker_effect = target0 / targetw
        auto_workers = 0 if worker_effect < 1.0 else worker_candidate
    else:
        worker_effect = None
        auto_workers = 0
    selected = auto_workers if forced_workers is None else forced_workers
    return {
        "target_device": target_device.type,
        "rows": rows,
        f"{target_device.type}_speedup_vs_cpu": cpu_time / target0,
        f"{target_device.type}_workers{worker_candidate}_effect_vs_workers0": worker_effect,
        "selected_workers_for_full_retrain": int(selected),
        "selected_workers_source": "cli" if forced_workers is not None else "auto_speed_table",
        "passed": bool(cpu_time / target0 >= 2.0),
    }


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": rmse(residual),
        "bias": float(np.mean(residual)),
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def analytic_floor(ids: np.ndarray) -> dict:
    with h5py.File(DATA, "r") as f:
        h0 = np.asarray(f["true_values/H0_true"][:], dtype=np.float64)
        h0a = np.asarray(f["approx_outputs/H0_approx"][:], dtype=np.float64)
        kappa = np.asarray(f["perturbations/kappa_ext"][:], dtype=np.float64)
        sigrel = np.asarray(f["perturbations/dt_sigma_rel"][:], dtype=np.float64)
        eps = np.asarray(f["perturbations/dt_approx_noise_factor"][:], dtype=np.float64)

    def block(sel: np.ndarray) -> dict:
        h = h0[sel]
        hk = h0a[sel] / (1.0 - kappa[sel])
        return {
            "H0_mean_times_sigma_rel_mean": float(h.mean() * sigrel[sel].mean()),
            "H0_mean_times_sigma_rel_rms": float(h.mean() * np.sqrt(np.mean(sigrel[sel] ** 2))),
            "linear_H0_epsilon_RMSE": rmse(h * eps[sel]),
            "perfect_kappa_oracle": metrics(h, hk),
            "perfect_joint_oracle": metrics(h, h),
        }

    return {
        "all_systems": block(np.arange(len(h0))),
        "val_unique": block(ids),
        "acceptance_rmse_band_from_floor_3p15": [2.7, 3.6],
    }


def evaluate_model(cfg: dict, scaler: dict, training_summary: dict,
                   infra: dict, bootstrap_n: int, device_name: str,
                   workers: int) -> dict:
    device = select_device(device_name)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    loader = build_loader(cfg, "val", [1], [1], scaler, cfg["seed"], workers, device, False)
    ids = split_system_ids(DATA, "val", cfg["seed"])
    pred_corr = []
    pred_sigma = []
    y_true = []
    h0_approx = []
    kappa = []
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
    pred_corr = np.concatenate(pred_corr).astype(np.float64)
    pred_sigma = np.concatenate(pred_sigma).astype(np.float64)
    with h5py.File(DATA, "r") as f:
        y_true = np.asarray(f["true_values/H0_true"][:], dtype=np.float64)[ids]
        h0_approx = np.asarray(f["approx_outputs/H0_approx"][:], dtype=np.float64)[ids]
        kappa = np.asarray(f["perturbations/kappa_ext"][:], dtype=np.float64)[ids]
        h0_all = np.asarray(f["true_values/H0_true"][:], dtype=np.float64)
    model_h0 = h0_approx - pred_corr
    no_corr = h0_approx
    kappa_oracle = h0_approx / (1.0 - kappa)
    residual = model_h0 - y_true
    z = residual / pred_sigma
    coverage = np.abs(z) <= 1.0
    rng = np.random.default_rng(20260504)
    boot = {"gap": [], "model_RMSE": [], "oracle_RMSE": [], "model_r": [], "coverage": []}
    n = len(y_true)
    for _ in range(bootstrap_n):
        b = rng.integers(0, n, n)
        boot["model_RMSE"].append(rmse(model_h0[b] - y_true[b]))
        boot["oracle_RMSE"].append(rmse(kappa_oracle[b] - y_true[b]))
        boot["gap"].append(boot["oracle_RMSE"][-1] - boot["model_RMSE"][-1])
        boot["model_r"].append(float(np.corrcoef(y_true[b], model_h0[b])[0, 1]))
        boot["coverage"].append(float(np.mean(coverage[b])))
    boot_ci = {
        k: {
            "mean": None if len(v) == 0 else float(np.mean(v)),
            "ci95": None if len(v) == 0 else [float(x) for x in np.percentile(v, [2.5, 97.5])],
        }
        for k, v in boot.items()
    }
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
    result = {
        "training": training_summary,
        "infrastructure": infra,
        "best_checkpoint": display_path(CKPT),
        "data": display_path(DATA),
        "scaler": display_path(SCALER),
        "analytic_noise_floor": analytic_floor(ids),
        "best": {
            "mode1": {
                "val_m1_scaled_MSE": float(np.mean(((pred_corr - (h0_approx - y_true)) / scaler["mode1"]["scale"]) ** 2)),
                "h0": {
                    "model": metrics(y_true, model_h0),
                    "no_correction": metrics(y_true, no_corr),
                    "perfect_kappa_oracle": metrics(y_true, kappa_oracle),
                    "perfect_joint_oracle": metrics(y_true, y_true),
                    "target_sign_note": "HDF5 mode1_H0_error = H0_approx - H0_true, so sign-corrected H0 is H0_approx - model_output.",
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
            "n": bootstrap_n,
            "gap_oracle_minus_model_RMSE": boot_ci["gap"],
            "model_RMSE": boot_ci["model_RMSE"],
            "oracle_RMSE": boot_ci["oracle_RMSE"],
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
    EVAL.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL, "w") as f:
        json.dump(result, f, indent=2)
    return result


def delete_partial_outputs() -> list[dict]:
    rows = []
    paths = [CKPT] if PATHS.scaler_from_env else [CKPT, SCALER]
    for path in paths:
        existed = path.exists()
        if existed:
            path.unlink()
        rows.append({"path": display_path(path), "existed": existed, "deleted": existed})
    if PATHS.scaler_from_env:
        rows.append({"path": display_path(SCALER), "existed": SCALER.exists(), "deleted": False})
    return rows


def run_equivalence_phase(cfg: dict, target_device: torch.device, worker_candidate: int) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    temp_scaler = create_target_scaler(DATA, RUNS / "target_scaler_phase3_v2_6_temp.pkl", cfg["seed"])
    fwd = forward_equivalence(cfg, temp_scaler, target_device.type)
    dist_workers = 0 if target_device.type == "mps" else worker_candidate
    dist = distribution_equivalence(cfg, temp_scaler, target_device.type, dist_workers)
    result = {
        "round": "phase3_v2_6",
        "phase": "equivalence",
        "device": {
            "target": target_device.type,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "forward_only": fwd,
        "distribution_equivalence": dist,
        "passed": bool(fwd["passed"] and dist["passed"]),
        "next": "next: --phase train on CUDA",
    }
    EQUIVALENCE.parent.mkdir(parents=True, exist_ok=True)
    with open(EQUIVALENCE, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("next: --phase train on CUDA")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    add_phase_args(parser)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--workers", type=int, default=None,
                        help="Full retrain/eval workers override. Omit to select from speed table.")
    parser.add_argument("--worker-candidate", type=int, default=None,
                        help="Accelerator worker count to benchmark against workers=0.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Full retrain epoch override; default keeps v2.5/v2.6 value 50.")
    args = parser.parse_args()
    cfg = load_cfg(args.epochs)
    target_device = select_device(args.device)
    if target_device.type == "cpu":
        raise SystemExit("This round requires cuda or mps for accelerator equivalence/retrain.")
    worker_candidate = (
        default_worker_candidate(target_device)
        if args.worker_candidate is None
        else int(args.worker_candidate)
    )
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.phase == "equivalence":
        run_equivalence_phase(cfg, target_device, worker_candidate)
        return

    infra = {"equivalence_handoff": load_equivalence_payload(args.equivalence_from)}
    if args.phase == "all":
        temp_scaler = create_target_scaler(DATA, RUNS / "target_scaler_phase3_v2_6_temp.pkl", cfg["seed"])
        fwd = forward_equivalence(cfg, temp_scaler, target_device.type)
        infra["forward_only"] = fwd
        if not fwd["passed"]:
            with open(INFRA, "w") as f:
                json.dump(infra, f, indent=2)
            raise SystemExit("Forward-only equivalence failed.")

        dist_workers = 0 if target_device.type == "mps" else worker_candidate
        dist = distribution_equivalence(cfg, temp_scaler, target_device.type, dist_workers)
        infra["distribution_equivalence"] = dist
        if not dist["passed"]:
            with open(INFRA, "w") as f:
                json.dump(infra, f, indent=2)
            raise SystemExit("Distribution equivalence failed.")

        speed = speed_table(cfg, temp_scaler, target_device.type, worker_candidate, args.workers)
        infra["speed"] = speed
        if not speed["passed"]:
            with open(INFRA, "w") as f:
                json.dump(infra, f, indent=2)
            raise SystemExit(f"{target_device.type.upper()} speedup < 2x.")
        selected_workers = int(speed["selected_workers_for_full_retrain"])
    else:
        selected_workers = default_worker_candidate(target_device) if args.workers is None else int(args.workers)

    infra["partial_output_deletion"] = delete_partial_outputs()
    full_scaler = load_scaler() if PATHS.scaler_from_env else create_target_scaler(DATA, SCALER, cfg["seed"])
    full = train_one(
        cfg,
        cfg["seed"],
        target_device.type,
        workers=selected_workers,
        scaler=full_scaler,
        epochs=cfg["training"]["epochs"],
        output_ckpt=CKPT,
        criterion="mode1_task",
        patience=cfg["training"]["early_stop_patience"],
    )
    training_summary = {
        "criterion": "val.mode1_task",
        "ended_epoch": full["ended_epoch"],
        "early_stop_epoch": full["ended_epoch"] if full["ended_epoch"] < cfg["training"]["epochs"] else None,
        "best_epoch": full["best_epoch"],
        "best_val_m1": full["best_val_m1"],
        "max_epochs": cfg["training"]["epochs"],
        "patience": cfg["training"]["early_stop_patience"],
        "num_workers": selected_workers,
        "device": target_device.type,
        "checkpoint": display_path(CKPT),
        "data": display_path(DATA),
        "scaler": display_path(SCALER),
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "w") as f:
        json.dump({"history": full["history"], "training": training_summary}, f, indent=2)
    infra["full_retrain"] = training_summary
    with open(INFRA, "w") as f:
        json.dump(infra, f, indent=2)
    evaluate_model(cfg, full_scaler, training_summary, infra, args.bootstrap_n,
                   target_device.type, selected_workers)
    print(json.dumps({"infra": infra, "training": training_summary, "eval": str(EVAL)}, indent=2))


if __name__ == "__main__":
    main()
