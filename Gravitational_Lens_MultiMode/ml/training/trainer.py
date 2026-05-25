"""
CorrectorTrainer — multi-task 훈련 루프.

AdamW + cosine LR + warmup, AMP 옵션.
Early stopping: val total loss 기준.
Best checkpoint: runs/<YYYYMMDD_HHMMSS>/best.pt + config.yaml.
TensorBoard: mode별 loss / metric 분리 로깅.
"""

from __future__ import annotations

import copy
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.training.losses import composite_loss
from ml.training.physics_pairing import add_paired_physics_predictions
from ml.training.round_eval import evaluate_mode1_h0_on_loader


class CorrectorTrainer:
    """
    Parameters
    ----------
    model  : MultiModalErrorCorrector
    cfg    : ml.yaml 전체 dict
    device : torch.device
    """

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device) -> None:
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device

        t_cfg = cfg["training"]
        self.lr             = t_cfg.get("lr", 3e-4)
        self.weight_decay   = t_cfg.get("weight_decay", 1e-4)
        self.epochs         = t_cfg.get("epochs", 100)
        self.warmup_epochs  = t_cfg.get("warmup_epochs", 5)
        self.patience       = t_cfg.get("early_stop_patience", 10)
        self.loss_weights   = t_cfg.get("loss_weights", {})
        self.use_amp        = t_cfg.get("amp", True) and device.type == "cuda"
        self.best_metric    = t_cfg.get("best_metric", "total")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------ #
    # 스케줄러 (cosine + warmup)                                           #
    # ------------------------------------------------------------------ #
    def _build_scheduler(self, n_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        wu = self.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < wu:
                return float(epoch + 1) / float(max(wu, 1))
            progress = float(epoch - wu) / float(max(n_epochs - wu, 1))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------ #
    # 배치 → device                                                        #
    # ------------------------------------------------------------------ #
    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ #
    # 단일 스텝                                                             #
    # ------------------------------------------------------------------ #
    def _step(self, batch: dict, train: bool) -> dict:
        batch = self._to_device(batch)

        with autocast("cuda", enabled=self.use_amp):
            pred = self.model(
                lc          = batch["lc"],
                lc_mask     = batch["lc_mask"],
                params      = batch["params"],
                sigma_curve = batch["sigma_curve"],
                image       = batch["image"],
                target_mode = batch["target_mode"].to(self.device),
            )
            if self.loss_weights.get("physics", 0.0) > 0.0:
                pred = add_paired_physics_predictions(
                    self.model, pred, batch, use_amp=self.use_amp
                )
            losses = composite_loss(pred, batch, self.loss_weights)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    # ------------------------------------------------------------------ #
    # 에폭 루프                                                             #
    # ------------------------------------------------------------------ #
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        agg: dict = {}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                ls = self._step(batch, train)
                for k, v in ls.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    # ------------------------------------------------------------------ #
    # fit                                                                   #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """합성 데이터 multi-task 사전 학습."""
        run_dir = self._make_run_dir(output_dir)
        self._save_config(run_dir)
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        scheduler = self._build_scheduler(self.epochs)

        best_val   = math.inf
        best_state = None
        no_improve = 0
        history: dict = {"train": [], "val": []}

        for epoch in range(self.epochs):
            t0 = time.time()
            tr = self._run_epoch(train_loader, train=True)
            vl = self._run_epoch(val_loader,   train=False)
            scheduler.step()

            history["train"].append(tr)
            history["val"].append(vl)

            self._log_tb(writer, tr, vl, epoch)

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{self.epochs}  "
                  f"train={tr['total']:.4f}  val={vl['total']:.4f}  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                  f"({elapsed:.1f}s)")

            criterion = vl.get(self.best_metric, vl["total"])
            if criterion < best_val:
                best_val   = criterion
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, run_dir / "best.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stop at epoch {epoch+1}")
                    break

        writer.close()
        if best_state:
            self.model.load_state_dict(best_state)
        return {"run_dir": str(run_dir), "best_val_loss": best_val, "history": history}

    # ------------------------------------------------------------------ #
    # fine_tune                                                            #
    # ------------------------------------------------------------------ #
    def fine_tune(
        self,
        real_loader: DataLoader,
        val_loader: DataLoader,
        freeze_encoders: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """실측 데이터 fine-tune.  freeze_encoders=True → 인코더 파라미터 고정."""
        if freeze_encoders:
            for name, param in self.model.named_parameters():
                if any(enc in name for enc in ("lc_enc", "par_enc", "sig_enc")):
                    param.requires_grad_(False)
            # 옵티마이저 재생성 (학습 파라미터만)
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr * 0.1,
                weight_decay=self.weight_decay,
            )

        return self.fit(real_loader, val_loader, output_dir=output_dir)

    # ------------------------------------------------------------------ #
    # evaluate                                                             #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        test_loader: DataLoader,
        *,
        h5_path: Path | None = None,
        ids=None,
        target_scaler: dict | None = None,
        output_path: Path | None = None,
        label: str = "eval",
        training_summary: dict | None = None,
        infra: dict | None = None,
        checkpoint_display: str = "",
        data_display: str | None = None,
        scaler_display: str = "",
        phase4_floor_analysis: dict | None = None,
        bootstrap_n: int = 1000,
    ) -> dict:
        """mode별 metric 분리 보고 또는 공용 Mode 1 round 평가.

        If ``h5_path``, ``ids`` and ``target_scaler`` are provided this delegates
        to ``ml.training.round_eval`` so trainer path A and round path B use the
        same target-scaler inversion, H0 metric, coverage, and bootstrap code.
        """
        if h5_path is not None and ids is not None and target_scaler is not None:
            return evaluate_mode1_h0_on_loader(
                model=self.model,
                loader=test_loader,
                device=self.device,
                path=Path(h5_path),
                ids=ids,
                scaler=target_scaler,
                bootstrap_n=bootstrap_n,
                output_path=output_path,
                label=label,
                training_summary=training_summary or {},
                infra=infra or {},
                checkpoint_display=checkpoint_display,
                data_display=data_display or str(h5_path),
                scaler_display=scaler_display,
                phase4_floor_analysis=phase4_floor_analysis,
                move_batch=lambda batch, device: self._to_device(batch),
            )

        self.model.eval()
        agg: dict = {}
        n_mode: dict = {1: 0, 2: 0}
        per_mode: dict = {1: {}, 2: {}}

        with torch.no_grad():
            for batch in test_loader:
                batch = self._to_device(batch)

                pred = self.model(
                    lc          = batch["lc"],
                    lc_mask     = batch["lc_mask"],
                    params      = batch["params"],
                    sigma_curve = batch["sigma_curve"],
                    image       = batch["image"],
                    target_mode = batch["target_mode"].to(self.device),
                )
                losses = composite_loss(pred, batch, self.loss_weights)
                for k, v in losses.items():
                    agg[k] = agg.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

                # mode별 MSE 계산
                tm = batch["target_mode"]
                for mode_id in [1, 2]:
                    m = (tm == mode_id)
                    if not m.any():
                        continue
                    n_mode[mode_id] += m.sum().item()
                    if mode_id == 1 and pred["mode1"]:
                        t1 = batch["target"][m, 0].to(self.device)
                        err = ((pred["mode1"]["h0_correction"] - t1) ** 2).mean().item()
                        per_mode[1]["mse"] = per_mode[1].get("mse", 0.0) + err
                    if mode_id == 2 and pred["mode2"]:
                        per_mode[2]["mse"] = per_mode[2].get("mse", 0.0) + (
                            losses["mode2_task"].item()
                            if torch.is_tensor(losses["mode2_task"])
                            else losses["mode2_task"]
                        )

        n_batches = max(len(test_loader), 1)
        result = {k: v / n_batches for k, v in agg.items()}
        result["per_mode"] = {}
        for mode_id in [1, 2]:
            if n_mode[mode_id] > 0:
                result["per_mode"][f"mode{mode_id}_mse"] = (
                    per_mode[mode_id].get("mse", 0.0) / n_batches
                )
        return result

    # ------------------------------------------------------------------ #
    # 헬퍼                                                                  #
    # ------------------------------------------------------------------ #
    def _make_run_dir(self, base: Optional[Path]) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (Path(base) if base else Path("runs")) / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_config(self, run_dir: Path) -> None:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg, f)

    def _log_tb(self, writer: SummaryWriter, tr: dict,
                vl: dict, epoch: int) -> None:
        for k in tr:
            writer.add_scalars(k, {"train": tr[k], "val": vl.get(k, 0.0)}, epoch)
