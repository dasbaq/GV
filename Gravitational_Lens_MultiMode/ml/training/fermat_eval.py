"""Evaluation for the H0-blind Fermat-ratio posterior."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import kstest
from ml.models.fermat_ratio import mixture_nll, posterior_mean_std


def mixture_cdf(pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(pred["logits"], -1)
    z = (target.unsqueeze(-1) - pred["mean"]) / pred["log_scale"].exp()
    return (w * .5 * (1. + torch.erf(z / math.sqrt(2.)))).sum(-1)


def evaluate_fermat_posterior(model, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []; means: list[np.ndarray] = []; stds: list[np.ndarray] = []; pits: list[np.ndarray] = []; nlls: list[float] = []
    families: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = {k: v.to(device) for k, v in batch.items()}
            out = model(params=moved["params"], geometry=moved["geometry"], image=moved["image"])
            mean, std = posterior_mean_std(out)
            nlls.append(float(mixture_nll(out, moved["target"]).cpu()))
            targets.append(moved["target"].cpu().numpy()); means.append(mean.cpu().numpy()); stds.append(std.cpu().numpy())
            pits.append(mixture_cdf(out, moved["target"]).cpu().numpy()); families.append(batch["family_id"].numpy())
    y = np.concatenate(targets); m = np.concatenate(means); s = np.concatenate(stds); pit = np.concatenate(pits); fam = np.concatenate(families)
    abs_err = np.abs(m - y)
    coverage = {str(level): float(np.mean(abs_err <= z * s)) for level, z in ((.68, .9944579), (.95, 1.959964))}
    # Within a counterfactual family input and posterior must be H0-invariant.
    diffs = []
    for f in np.unique(fam):
        vals = m[fam == f]
        if vals.size > 1: diffs.append(float(vals.max() - vals.min()))
    return {"target": "log_dphi_truth_over_sie", "n": int(y.size), "nll": float(np.mean(nlls)),
            "MAE": float(abs_err.mean()), "RMSE": float(np.sqrt(np.mean((m-y)**2))),
            "CRPS_gaussian_moment_proxy": float(np.mean(abs_err - s / math.sqrt(math.pi))),
            "coverage": coverage, "pit_ks_pvalue": float(kstest(pit, "uniform").pvalue),
            "pit_mean": float(pit.mean()), "counterfactual_posterior_mean_max_range": float(max(diffs, default=0.0)),
            "counterfactual_invariant": bool(max(diffs, default=0.0) <= 1e-6)}
