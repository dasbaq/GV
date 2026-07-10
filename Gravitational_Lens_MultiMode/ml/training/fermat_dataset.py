"""Dataset for the H0-blind Fermat-ratio track."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FermatRatioDataset(Dataset):
    """Expose only image, normalized geometry, and SIE structural parameters.

    No H0, time-delay, light-curve, or truth-nuisance key is read.
    """
    forbidden_inputs = frozenset({"H0", "H0_approx", "dt_lc", "dt_lc_sigma", "light_curves", "sigma_curve"})

    def __init__(self, path: str | Path, indices: np.ndarray | None = None, *, canonical_only: bool = False) -> None:
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            if f["metadata"].attrs.get("track", "") != "fermat_ratio_h0_blind_v1":
                raise ValueError("not a fermat_ratio_h0_blind_v1 catalog")
            self.n = int(f["metadata"].attrs["n_systems"])
        self.indices = np.arange(self.n, dtype=int) if indices is None else np.asarray(indices, dtype=int)
        if canonical_only:
            with h5py.File(self.path, "r") as f:
                family = np.asarray(f["counterfactual_family_id"][self.indices], dtype=np.int64)
            _, first = np.unique(family, return_index=True)
            self.indices = self.indices[np.sort(first)]

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[index])
        with h5py.File(self.path, "r") as f:
            family = int(f["counterfactual_family_id"][i])
            theta_e = float(f["sie_parameters/theta_E"][family])
            if not np.isfinite(theta_e) or theta_e <= 0.0:
                raise ValueError("theta_E must be finite and positive")
            geometry = np.asarray(f["observables_by_family/image_positions_arcsec"][family], dtype=np.float32).reshape(4) / theta_e
            params = np.asarray([f[f"sie_parameters/{k}"][family] for k in ("z_lens", "z_source", "sigma_v", "q", "theta_E")], dtype=np.float32)
            # Fixed, documented scales avoid learning from H0-population statistics.
            params = (params - np.asarray([0.1, 0.5, 100., .5, .5], dtype=np.float32)) / np.asarray([.9, 3., 400., .5, 1.5], dtype=np.float32)
            image = np.asarray(f["observables_by_family/I_obs"][family], dtype=np.float32)[None]
            target = np.float32(f["target/log_dphi_truth_over_sie"][i])
        return {"params": torch.from_numpy(params), "geometry": torch.from_numpy(geometry),
                "image": torch.from_numpy(image), "target": torch.tensor(target),
                "family_id": torch.tensor(family, dtype=torch.long)}
