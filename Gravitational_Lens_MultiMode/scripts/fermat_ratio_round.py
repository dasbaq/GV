"""H0-blind Fermat-ratio round dispatcher.

Run catalog/equivalence on M2, then use ``--phase train --device cuda`` on
Kaggle.  This script deliberately does not accept Phase4 H0 scalers.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.data.fermat_catalog import FermatCatalogConfig, build_fermat_ratio_catalog
from ml.models.fermat_ratio import FermatRatioPosterior
from ml.training.fermat_dataset import FermatRatioDataset
from pipelines.train_fermat_ratio import train

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("equivalence", "train"), required=True); p.add_argument("--catalog", default=str(ROOT / "data/mock/fermat_ratio_v1.h5")); p.add_argument("--checkpoint", default=str(ROOT / "data/checkpoints/fermat_ratio_v1.pt")); p.add_argument("--device", default="cuda"); p.add_argument("--families", type=int, default=200)
    a = p.parse_args(); catalog = Path(a.catalog)
    if a.phase == "equivalence":
        summary = build_fermat_ratio_catalog(catalog, FermatCatalogConfig(n_families=a.families))
        b = next(iter(DataLoader(FermatRatioDataset(catalog), batch_size=2)))
        m = FermatRatioPosterior(); out = m(params=b["params"], geometry=b["geometry"], image=b["image"])
        summary["cpu_forward_finite"] = bool(all(torch.isfinite(v).all() for v in out.values()))
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(train(catalog, a.checkpoint, ROOT / "config/ml_fermat_ratio.yaml", device=a.device), indent=2))
if __name__ == "__main__": main()
