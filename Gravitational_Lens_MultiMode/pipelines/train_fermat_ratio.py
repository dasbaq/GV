"""Train the separate H0-blind Fermat-ratio posterior on Kaggle CUDA."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from ml.models.fermat_ratio import FermatRatioPosterior, mixture_nll
from ml.training.fermat_dataset import FermatRatioDataset
from ml.training.fermat_eval import evaluate_fermat_posterior


def train(catalog: str | Path, output: str | Path, config_path: str | Path, *, device: str = "cuda") -> dict:
    """Fit only the H0-blind posterior; Phase4 H0 artifacts are never loaded."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    # Split by counterfactual family, never by individual H0 audit replicas.
    all_ds = FermatRatioDataset(catalog)
    with h5py.File(catalog, "r") as f:
        family = np.asarray(f["counterfactual_family_id"][:], dtype=np.int64)
    unique_family = np.unique(family)
    order = torch.randperm(len(unique_family), generator=torch.Generator().manual_seed(int(cfg["seed"]))).numpy()
    cut = max(1, int(.8 * len(unique_family)))
    train_family, val_family = unique_family[order[:cut]], unique_family[order[cut:]]
    train_ids = np.flatnonzero(np.isin(family, train_family)); val_ids = np.flatnonzero(np.isin(family, val_family))
    # Counterfactual replicas are validation/invariance aids, not extra train weight.
    train_ds = FermatRatioDataset(catalog, train_ids, canonical_only=True)
    val_ds = FermatRatioDataset(catalog, val_ids)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = FermatRatioPosterior(**cfg["model"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"].get("weight_decay", 0.)))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]))
    for _ in range(int(cfg["training"]["epochs"])):
        model.train()
        for batch in train_loader:
            b = {k:v.to(dev) for k,v in batch.items()}; opt.zero_grad(); loss = mixture_nll(model(params=b["params"], geometry=b["geometry"], image=b["image"]), b["target"]); loss.backward(); opt.step()
    report = evaluate_fermat_posterior(model, val_loader, dev)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "track": "fermat_ratio_h0_blind_v1"}, output)
    output.with_suffix(".eval.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--catalog", required=True); p.add_argument("--output", required=True); p.add_argument("--config", default="config/ml_fermat_ratio.yaml"); p.add_argument("--device", default="cuda")
    a = p.parse_args(); print(json.dumps(train(a.catalog, a.output, a.config, device=a.device), indent=2))
if __name__ == "__main__": main()
