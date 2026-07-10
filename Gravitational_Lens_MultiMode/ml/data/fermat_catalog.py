"""H0-blind Fermat-ratio catalog generation.

The learning target is ``log(|Delta phi_truth| / |Delta phi_SIE|)``.  It is
dimensionless and depends on lens structure, not on the Hubble constant.  The
H0 values written under ``audit`` are deliberately not consumed by this track.
SIE is the project's fixed standard approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ml.data.error_catalog import CatalogConfig, _collect_valid_pairs


@dataclass(frozen=True)
class FermatCatalogConfig:
    n_families: int = 200
    seed: int = 42
    image_size: int = 64
    pixel_scale: float = 0.1
    h0_counterfactuals: tuple[float, ...] = (60.0, 70.0, 80.0)


def build_fermat_ratio_catalog(output_path: Path, config: FermatCatalogConfig = FermatCatalogConfig()) -> dict[str, Any]:
    """Create a validity-only, H0-blind training catalog.

    Images and image positions are observation-side quantities.  Every family
    is repeated for audit-only H0 values; all tensors exposed to the dataset
    and the Fermat-ratio target are exactly invariant within a family.
    """
    if not config.h0_counterfactuals or any(v <= 0.0 for v in config.h0_counterfactuals):
        raise ValueError("h0_counterfactuals must contain positive values")
    phase4 = CatalogConfig(
        n_systems=config.n_families, seed=config.seed, image_size=config.image_size,
        pixel_scale=config.pixel_scale, validity_filter="v0_4",
    )
    bases, pairs, reject_log = _collect_valid_pairs(phase4)
    n_cf = len(config.h0_counterfactuals)
    n = len(pairs) * n_cf
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def repeat(value: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(value), n_cf, axis=0)

    phi_sie = np.asarray([p["approx"].fermat_potential for p in pairs], dtype=np.float64)
    phi_truth = np.asarray([p["truth_fermat_potential"] for p in pairs], dtype=np.float64)
    if not (np.isfinite(phi_sie).all() and np.isfinite(phi_truth).all() and (phi_sie > 0).all() and (phi_truth > 0).all()):
        raise RuntimeError("Fermat-ratio catalog requires finite positive Fermat differences")
    target = np.log(phi_truth / phi_sie).astype(np.float32)
    theta_e = np.asarray([p["theta_E"] for p in pairs], dtype=np.float32)
    positions = np.stack([
        np.stack([p["theta_truth_1"], p["theta_truth_2"]]).astype(np.float32) for p in pairs
    ])
    if not np.all(np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1) >= 0.1):
        raise RuntimeError("physical image-separation validity was violated")

    h0_values = np.tile(np.asarray(config.h0_counterfactuals, dtype=np.float32), len(pairs))
    family = np.repeat(np.arange(len(pairs), dtype=np.int64), n_cf)
    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["track"] = "fermat_ratio_h0_blind_v1"
        meta.attrs["n_families"] = len(pairs)
        meta.attrs["n_systems"] = n
        meta.attrs["validity_filter"] = "physical_only_v0_4"
        meta.attrs["forbidden_model_inputs"] = "H0,H0_approx,dt_lc,dt_lc_sigma,light_curves,sigma_curve,truth_nuisance"
        meta.attrs["target"] = "log(dphi_truth/dphi_sie)"
        meta.attrs["standard_approximation"] = "SIE"
        f.create_dataset("counterfactual_family_id", data=family)
        f.create_dataset("target/log_dphi_truth_over_sie", data=repeat(target))
        # Counterfactuals share all observation-side inputs.  Store these once
        # per physical family rather than triplicating a large image tensor.
        obs = f.create_group("observables_by_family")
        obs.create_dataset("image_positions_arcsec", data=positions)
        obs.create_dataset("I_obs", data=np.stack([p["I_obs"] for p in pairs]).astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
        sie = f.create_group("sie_parameters")
        sie.create_dataset("z_lens", data=np.asarray([b["z_lens"] for b in bases], dtype=np.float32))
        sie.create_dataset("z_source", data=np.asarray([b["z_source"] for b in bases], dtype=np.float32))
        sie.create_dataset("sigma_v", data=np.asarray([b["sigma_v"] for b in bases], dtype=np.float32))
        sie.create_dataset("q", data=np.asarray([b["q"] for b in bases], dtype=np.float32))
        sie.create_dataset("theta_E", data=theta_e)
        audit = f.create_group("audit")
        audit.create_dataset("H0_counterfactual", data=h0_values)
        audit.create_dataset("dphi_truth", data=repeat(phi_truth.astype(np.float32)))
        audit.create_dataset("dphi_sie", data=repeat(phi_sie.astype(np.float32)))
        audit.create_dataset("mu_abs", data=repeat(np.asarray([p["mu_true"] for p in pairs], dtype=np.float32)))

    return {"catalog_path": str(output_path), "n_families": len(pairs), "n_systems": n,
            "target_mean": float(target.mean()), "target_std": float(target.std()),
            "reject_counts": reject_log["reject_counts"], "validity_filter": "physical_only_v0_4"}
