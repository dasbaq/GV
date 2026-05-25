"""Phase 4 v0.6 재학습 라운드.

v0.4와 동일한 데이터(phase4_v0_4.h5, 20-dim observed_features)를 사용하되,
v0.6 모델 구조(Image 모달리티 복구, Mode 3 삭제 유지)로 재학습한다.

v0.5 → v0.6 변경 요약:
  - Mode1Head in_dim: d_model×2=256 → d_model×3=384 (복구)
    (cat([fused_sub, h_lc[mask]]) → cat([fused_sub, h_lc[mask], h_img[mask]]))
  - ImageEncoder 복구 (I_obs 1ch), Mode3Head 삭제 유지
  - fusion: 3-way 고정 유지 (LC + Param + Σ-curve), image는 head1 전용
  - composite_loss: mode3_task, ssim 항목 삭제 유지
  - dataset: image 키 복구 (I_obs [1,H,W])

데이터·스케일러·승인 기준은 v0.4와 동일하게 유지한다.
Kaggle Dataset: donghyun51/lens-phase4-v0-4 (재업로드 불필요)

SIE 표준 근사 가정: 모든 역산은 SIE 단일 고정 근사 위에서 동작한다.
Units: H0 and corrections [km/s/Mpc].
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models.error_corrector import MultiModalErrorCorrector
from ml.training.dataset import LensCorrectionDataset
from ml.training.losses import composite_loss
from ml.training.physics_pairing import add_paired_physics_predictions
from ml.training.round_eval import (
    acceptance_report as shared_acceptance_report,
    evaluate_mode1_h0_on_loader,
    mode2_correction_availability,
)
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

# v0.5는 v0.4와 동일한 데이터를 사용한다 (재업로드 불필요)
DATA_NAME = "phase4_v0_4.h5"
UNFILTERED_DATA_NAME = "phase4_v0_4_eval_unfiltered.h5"
PATHS = build_round_paths(
    root=ROOT,
    data_name=DATA_NAME,
    round_name="phase4_v0_6_round",
    checkpoint_name="phase4_v0_6_imgres_best.pt",
    history_name="phase4_v0_6_imgres_long_history.json",
    eval_name="phase4_v0_6_imgres_h0_eval.json",
    infra_name="phase4_v0_6_infra_equivalence.json",
    scaler_name="target_scaler_phase4_v0_4.pkl",   # 스케일러는 v0.4와 공유
    unfiltered_name=UNFILTERED_DATA_NAME,
    eval_unfiltered_name="phase4_v0_6_imgres_h0_eval_unfiltered.json",
)
DATA = PATHS.data
UNFILTERED_DATA = PATHS.unfiltered
SCALER = PATHS.scaler
CKPT = PATHS.checkpoint
HISTORY = PATHS.history
EVAL = PATHS.eval
EVAL_UNFILTERED = PATHS.eval_unfiltered
INFRA = PATHS.infra
RUNS = PATHS.runs
EQUIVALENCE = PATHS.work_root / "logs" / "phase4_v0_6_equivalence.json"

# 승인 기준: v0.4와 동일 (데이터 분포가 같으므로 기준 유지)
# 근거: data/logs/phase4_v0_4_floor_analysis.json,
#       data/logs/phase4_v0_4_r_ceiling.json, DECISIONS.md [2026-05-25].
ACCEPTANCE = {
    "cuda_forward_diff_max": 1.0e-4,
    "filtered_rmse_ci_lower_min": 0.5,
    "filtered_rmse_ci_upper_max": 11.08,
    "filtered_rmse_point_band": [0.5, 16.62],
    "unfiltered_filtered_rmse_ratio_max": 2.5,
    "coverage_ci_overlap": [0.62, 0.78],
    "positive_fraction_min": 0.95,
    "filtered_h0_r_min": 0.19,
    "best_val_m1": "record_only",
}

LEAK_TRIGGERS = {
    "filtered_rmse_ci_upper_below_nfw_oracle_lower": 0.5,
    "unfiltered_filtered_rmse_ratio_max": 3.18,
    "param_encoder_input_dim": 20,
}


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
    if model_cfg["param_in_dim"] != LEAK_TRIGGERS["param_encoder_input_dim"]:
        raise RuntimeError(
            f"ParamEncoder input dim changed: {model_cfg['param_in_dim']} "
            f"!= {LEAK_TRIGGERS['param_encoder_input_dim']}"
        )
    return MultiModalErrorCorrector(model_cfg)


def build_dataset(cfg: dict, split: str, modes: list[int], approx_levels: list[int],
                  scaler: dict | None, seed: int, path: Path | None = None) -> LensCorrectionDataset:
    return LensCorrectionDataset(
        h5_paths=[path or DATA],
        split=split,
        modes=tuple(modes),
        approx_levels=tuple(approx_levels),
        max_len=cfg["data"]["max_lc_len"],
        sigma_curve_size=cfg["data"]["sigma_curve_size"],
        image_size=cfg["data"].get("image_size", 64),   # v0.6: I_obs 크기
        mode2_max_dm_dim=cfg["model"]["mode2_max_dm_dim"],
        param_norm=cfg["data"]["param_normalization"],
        target_scaler=scaler,
        seed=seed,
    )


def build_loader(cfg: dict, split: str, modes: list[int], approx_levels: list[int],
                 scaler: dict | None, seed: int, workers: int, device: torch.device,
                 train: bool, batch_size: int | None = None,
                 path: Path | None = None) -> DataLoader:
    ds = build_dataset(cfg, split, modes, approx_levels, scaler, seed, path=path)
    kwargs = dataloader_kwargs(
        batch_size=batch_size or cfg["training"]["batch_size"],
        workers=workers,
        device=device,
        train=train,
    )
    if train:
        w_map = {1: 1.0, 2: 1.0}
        weights = torch.tensor([w_map[e[3]] for e in ds._index], dtype=torch.float)
        gen = torch.Generator()
        gen.manual_seed(seed)
        kwargs["sampler"] = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=gen
        )
    return DataLoader(ds, **kwargs)


def build_eval_loader_for_ids(cfg: dict, path: Path, ids: np.ndarray, scaler: dict,
                              workers: int, device: torch.device,
                              batch_size: int | None = None) -> DataLoader:
    ds = build_dataset(cfg, "train", [1], [1], scaler, cfg["seed"], path=path)
    ds._index = [(str(path), int(i), 1, 1) for i in np.asarray(ids, dtype=int)]
    kwargs = dataloader_kwargs(
        batch_size=batch_size or cfg["training"]["batch_size"],
        workers=workers,
        device=device,
        train=False,
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
    scaler = {
        "mode1": {"mean": float(mode1.mean()), "scale": float(mode1.std() + 1.0e-8)},
        "mode2": {
            "mean": np.zeros(4, dtype=np.float32),
            "scale": np.ones(4, dtype=np.float32),
        },
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
    """v0.6: image(I_obs) 인자 추가."""
    model.eval().to(device)
    batch = move_batch(batch, device)
    with torch.no_grad():
        out = model(
            lc=batch["lc"],
            lc_mask=batch["lc_mask"],
            params=batch["params"],
            sigma_curve=batch["sigma_curve"],
            image=batch["image"],
            target_mode=batch["target_mode"].to(device),
        )
    arrays: dict[str, np.ndarray] = {}
    if out["mode1"] is not None:
        arrays["mode1_pred"] = out["mode1"]["h0_correction"].detach().cpu().numpy()
        arrays["log_sigma_mode1"] = out["mode1"]["log_sigma"].detach().cpu().numpy()
    if out["mode2"] is not None:
        arrays["mode2_pred"] = out["mode2"]["dm_correction"].detach().cpu().numpy()
        arrays["log_sigma_mode2"] = out["mode2"]["log_sigma"].detach().cpu().numpy()
    return arrays


def forward_equivalence(cfg: dict, scaler: dict, target_device_name: str) -> dict:
    set_seed(42)
    cpu = torch.device("cpu")
    target_device = select_device(target_device_name)
    if target_device.type == "cpu":
        raise RuntimeError("Forward equivalence needs a non-CPU target device.")
    loader = build_loader(cfg, "val", [1, 2], [1, 2], scaler, 42, 0, cpu, False, batch_size=6)
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
    # v0.5: Mode 3 삭제됨 → modes=[1, 2]
    train_loader = build_loader(cfg, "train", [1, 2], [1, 2], scaler, seed, workers, device, True)
    val_loader = build_loader(cfg, "val", [1, 2], [1, 2], scaler, seed, workers, device, False)
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
                    target_mode=batch["target_mode"].to(device),
                )
                if cfg["training"]["loss_weights"].get("physics", 0.0) > 0.0:
                    pred = add_paired_physics_predictions(
                        model, pred, batch, use_amp=use_amp
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


def load_floor_analysis() -> dict | None:
    path = ROOT / "data" / "logs" / "phase4_v0_4_floor_analysis.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def evaluate_model_on_path(cfg: dict, scaler: dict, bootstrap_n: int,
                           device_name: str, workers: int, path: Path,
                           ids: np.ndarray, output_path: Path,
                           label: str, training_summary: dict,
                           infra: dict) -> dict:
    device = select_device(device_name)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    loader = build_eval_loader_for_ids(cfg, path, ids, scaler, workers, device)
    return evaluate_mode1_h0_on_loader(
        model=model,
        loader=loader,
        device=device,
        path=path,
        ids=ids,
        scaler=scaler,
        bootstrap_n=bootstrap_n,
        output_path=output_path,
        label=label,
        training_summary=training_summary,
        infra=infra,
        checkpoint_display=display_path(CKPT),
        data_display=display_path(path),
        scaler_display=display_path(SCALER),
        phase4_floor_analysis=load_floor_analysis(),
        move_batch=move_batch,
    )


def evaluate_model(cfg: dict, scaler: dict, training_summary: dict,
                   infra: dict, bootstrap_n: int, device_name: str,
                   workers: int, unfiltered_path: Path) -> tuple[dict, dict]:
    filtered_ids = split_system_ids(DATA, "val", cfg["seed"])
    with h5py.File(unfiltered_path, "r") as f:
        unfiltered_ids = np.arange(int(f["metadata"].attrs["n_systems"]))
    filtered = evaluate_model_on_path(
        cfg, scaler, bootstrap_n, device_name, workers, DATA, filtered_ids,
        EVAL, "filtered_val", training_summary, infra
    )
    unfiltered = evaluate_model_on_path(
        cfg, scaler, bootstrap_n, device_name, workers, unfiltered_path, unfiltered_ids,
        EVAL_UNFILTERED, "unfiltered_all", training_summary, infra
    )
    return filtered, unfiltered


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


def environment_sanity(device: torch.device) -> dict:
    row = {
        "requested_device": device.type,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": None,
        "model_version": "v0.6",
        "mode1_head_in_dim": 384,   # d_model×3 (v0.6, image 복구); v0.5는 256(d_model×2)
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        row["nvidia_smi"] = {
            "returncode": proc.returncode,
            "stdout_head": proc.stdout[:2000],
            "stderr_head": proc.stderr[:1000],
        }
    except Exception as exc:
        row["nvidia_smi"] = {"error": str(exc)}
    return row


def run_equivalence_phase(cfg: dict, target_device: torch.device, worker_candidate: int) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    temp_scaler = create_target_scaler(DATA, RUNS / "target_scaler_phase4_v0_6_temp.pkl", cfg["seed"])
    fwd = forward_equivalence(cfg, temp_scaler, target_device.type)
    dist_workers = worker_candidate if target_device.type == "cuda" else 0
    dist = distribution_equivalence(cfg, temp_scaler, target_device.type, dist_workers)
    result = {
        "round": "phase4_v0_6",
        "phase": "equivalence",
        "device": environment_sanity(target_device),
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


def acceptance_report(filtered: dict, unfiltered: dict, infra: dict,
                      training_summary: dict) -> dict:
    return shared_acceptance_report(
        filtered=filtered,
        unfiltered=unfiltered,
        infra=infra,
        training_summary=training_summary,
        acceptance=ACCEPTANCE,
        leak_triggers_config=LEAK_TRIGGERS,
        param_encoder_input_dim=len(load_cfg()["data"]["param_normalization"]) + 5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 v0.6 round (image restored, Mode3 removed)")
    add_phase_args(parser)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--eval-unfiltered", type=Path, default=None,
                        help="Unfiltered evaluation HDF5 path. Defaults to LENS_DATA_PATH_UNFILTERED.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Full retrain/eval workers override. CUDA default is 4.")
    parser.add_argument("--worker-candidate", type=int, default=None,
                        help="Accelerator worker count for distribution equivalence.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Full retrain epoch override; default 50.")
    args = parser.parse_args()
    cfg = load_cfg(args.epochs)
    target_device = select_device(args.device)
    if target_device.type == "cpu":
        raise SystemExit("This round requires cuda or mps for accelerator equivalence/retrain.")
    unfiltered_path = args.eval_unfiltered or UNFILTERED_DATA
    if unfiltered_path is None:
        raise SystemExit("Unfiltered evaluation path could not be resolved.")
    worker_candidate = (
        default_worker_candidate(target_device)
        if args.worker_candidate is None
        else int(args.worker_candidate)
    )
    selected_workers = default_worker_candidate(target_device) if args.workers is None else int(args.workers)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.phase == "equivalence":
        run_equivalence_phase(cfg, target_device, worker_candidate)
        return

    infra = {
        "environment_sanity": environment_sanity(target_device),
        "model_version": "v0.6",
        "data": display_path(DATA),
        "unfiltered_eval_data": display_path(unfiltered_path),
        "equivalence_handoff": load_equivalence_payload(args.equivalence_from),
        "param_encoder_input_dim": len(cfg["data"]["param_normalization"]) + 5,
        "acceptance_predeclared": ACCEPTANCE,
        "leak_triggers_predeclared": LEAK_TRIGGERS,
        "mode2_correction_availability": mode2_correction_availability(DATA),
        "v0_6_changes": {
            "mode1_head_in_dim": "384 (d_model×3, image 복구 — fused+h_lc+h_img)",
            "image_modality": "restored_I_obs_1ch",
            "mode3_head": "deleted",
            "fusion": "3-way fixed (LC+Param+Sigma)",
            "loss": "mode3_task and ssim deleted",
        },
    }
    if infra["param_encoder_input_dim"] != LEAK_TRIGGERS["param_encoder_input_dim"]:
        with open(INFRA, "w") as f:
            json.dump(infra, f, indent=2)
        raise SystemExit("ParamEncoder input dim changed; leak trigger fired.")

    if args.phase == "all":
        temp_scaler = create_target_scaler(DATA, RUNS / "target_scaler_phase4_v0_6_temp.pkl", cfg["seed"])
        fwd = forward_equivalence(cfg, temp_scaler, target_device.type)
        infra["forward_only"] = fwd
        if not fwd["passed"]:
            with open(INFRA, "w") as f:
                json.dump(infra, f, indent=2)
            raise SystemExit("Forward-only equivalence failed.")

        dist_workers = worker_candidate if target_device.type == "cuda" else 0
        dist = distribution_equivalence(cfg, temp_scaler, target_device.type, dist_workers)
        infra["distribution_equivalence"] = dist
        if not dist["passed"]:
            with open(INFRA, "w") as f:
                json.dump(infra, f, indent=2)
            raise SystemExit("Distribution equivalence failed.")

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
        "round": "phase4_v0_6",
        "model_version": "v0.6",
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
    filtered, unfiltered = evaluate_model(
        cfg, full_scaler, training_summary, infra, args.bootstrap_n,
        target_device.type, selected_workers, unfiltered_path
    )
    early_stopped = training_summary["early_stop_epoch"] is not None
    if should_run_acceptance(
        phase=args.phase,
        epochs_requested=cfg["training"]["epochs"],
        min_epochs=args.min_epochs_for_acceptance,
        ended_epoch=full["ended_epoch"],
        early_stopped=early_stopped,
    ):
        report = acceptance_report(filtered, unfiltered, infra, training_summary)
    else:
        report = skipped_acceptance()
        print("smoke run, acceptance skipped", flush=True)
    infra["stage_b_acceptance_report"] = report
    with open(INFRA, "w") as f:
        json.dump(infra, f, indent=2)
    if report["leak_triggered"]:
        print(json.dumps({"infra": infra, "training": training_summary, "acceptance": report}, indent=2))
        raise SystemExit("Leak trigger fired; stopping before any tuning.")
    print(json.dumps({
        "round": "phase4_v0_6",
        "infra": display_path(INFRA),
        "training": training_summary,
        "filtered_eval": display_path(EVAL),
        "unfiltered_eval": display_path(EVAL_UNFILTERED),
        "acceptance": report,
    }, indent=2))


if __name__ == "__main__":
    main()
