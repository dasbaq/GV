"""Inference adapter for the H0-blind Fermat-ratio posterior."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
import yaml
from ml.models.fermat_ratio import FermatRatioPosterior, posterior_mean_std


def _image(path: str | Path, index: int, image_size: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "images/I_obs" not in f: raise ValueError("phi correction requires observation images/I_obs")
        x = np.asarray(f["images/I_obs"][index], dtype=np.float32)
    if x.shape != (image_size, image_size):
        from skimage.transform import resize
        x = resize(x, (image_size, image_size), preserve_range=True, anti_aliasing=True).astype(np.float32)
    return x[None]


def run_fermat_ratio_posterior(*, input_path: str | Path, system_index: int, observation: Any, sie_fit: dict[str, Any], checkpoint_path: str | Path, device_name: str = "auto", samples: int = 256) -> dict[str, Any]:
    """Predict a dimensionless Fermat-ratio posterior without H0 or delay input."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("track") != "fermat_ratio_h0_blind_v1": raise RuntimeError("checkpoint is not an H0-blind Fermat-ratio artifact")
    cfg = payload["config"]; dev = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    model = FermatRatioPosterior(**cfg["model"]).to(dev); model.load_state_dict(payload["state_dict"]); model.eval()
    theta_e = float(sie_fit["theta_E"])
    if theta_e <= 0.: raise ValueError("SIE theta_E must be positive")
    geometry = np.asarray(observation.image_positions[:2], dtype=np.float32).reshape(4) / theta_e
    raw = np.asarray([observation.z_lens, observation.z_source, sie_fit["sigma_v"], sie_fit["q"], theta_e], dtype=np.float32)
    params = (raw - np.asarray([.1,.5,100.,.5,.5], dtype=np.float32)) / np.asarray([.9,3.,400.,.5,1.5], dtype=np.float32)
    image_size = int(getattr(cfg.get("data", {}), "image_size", 64)) if False else 64
    image = _image(input_path, system_index, image_size)
    with torch.no_grad():
        out = model(params=torch.from_numpy(params)[None].to(dev), geometry=torch.from_numpy(geometry)[None].to(dev), image=torch.from_numpy(image)[None].to(dev))
        mean, std = posterior_mean_std(out); w = torch.softmax(out["logits"], -1)
        comp = torch.multinomial(w[0], int(samples), replacement=True)
        draw = out["mean"][0, comp] + out["log_scale"][0, comp].exp() * torch.randn(int(samples), device=dev)
    y = draw.cpu().numpy()
    return {"applied": True, "track": "fermat_ratio_h0_blind_v1", "forbidden_inputs": ["H0", "H0_approx", "dt_lc", "dt_lc_sigma", "light_curves", "sigma_curve"],
            "log_dphi_truth_over_sie": {"mean": float(mean.item()), "std": float(std.item()), "samples": y.tolist()},
            "dphi_ratio_truth_over_sie": {"median": float(np.exp(np.median(y))), "ci68": np.exp(np.percentile(y,[16,84])).tolist(), "ci95": np.exp(np.percentile(y,[2.5,97.5])).tolist()},
            "checkpoint": str(checkpoint_path), "device": dev.type}
