from __future__ import annotations
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from ml.data.fermat_catalog import FermatCatalogConfig, build_fermat_ratio_catalog
from ml.models.fermat_ratio import FermatRatioPosterior, mixture_nll
from ml.training.fermat_dataset import FermatRatioDataset
from ml.training.fermat_eval import evaluate_fermat_posterior


def test_h0_blind_catalog_and_counterfactuals(tmp_path):
    path = tmp_path / "fermat.h5"
    build_fermat_ratio_catalog(path, FermatCatalogConfig(n_families=3, seed=19, h0_counterfactuals=(60., 80.)))
    with h5py.File(path, "r") as f:
        assert "target/log_dphi_truth_over_sie" in f
        ratio = f["audit/dphi_truth"][:] / f["audit/dphi_sie"][:]
        np.testing.assert_allclose(f["target/log_dphi_truth_over_sie"][:], np.log(ratio), atol=1e-6)
        assert np.all(f["audit/mu_abs"][:] < 1.)
        assert "H0_approx" not in f["sie_parameters"]
        assert "light_curves" not in f
    ds = FermatRatioDataset(path)
    a, b = ds[0], ds[1]
    assert a["family_id"] == b["family_id"]
    for key in ("params", "geometry", "image", "target"):
        assert torch.equal(a[key], b[key])


def test_posterior_smoke_and_evaluation(tmp_path):
    path = tmp_path / "fermat.h5"
    build_fermat_ratio_catalog(path, FermatCatalogConfig(n_families=3, seed=23, h0_counterfactuals=(60., 70.)))
    loader = DataLoader(FermatRatioDataset(path), batch_size=2)
    batch = next(iter(loader)); model = FermatRatioPosterior(d_model=16, components=5)
    out = model(params=batch["params"], geometry=batch["geometry"], image=batch["image"])
    loss = mixture_nll(out, batch["target"]); loss.backward()
    assert torch.isfinite(loss)
    report = evaluate_fermat_posterior(model, loader, torch.device("cpu"))
    assert report["n"] == 6
    assert "pit_ks_pvalue" in report and "coverage" in report


def test_inference_uses_no_h0_or_delay_inputs(tmp_path):
    from inversion.observation_io import ObservedLensSystem, ObservedLightCurves
    from ml.inference.fermat_ratio import run_fermat_ratio_posterior
    catalog = tmp_path / "input.h5"
    with h5py.File(catalog, "w") as f:
        f.create_group("images").create_dataset("I_obs", data=np.zeros((1, 64, 64), dtype=np.float32))
    checkpoint = tmp_path / "phi.pt"
    model = FermatRatioPosterior(d_model=16, components=5)
    torch.save({"track": "fermat_ratio_h0_blind_v1", "state_dict": model.state_dict(),
                "config": {"model": {"d_model": 16, "components": 5, "dropout": .1}}}, checkpoint)
    obs = ObservedLensSystem(image_positions=np.array([[1., 0.], [-1., 0.]], dtype=np.float32),
        light_curves=ObservedLightCurves(F=np.ones((1, 2), dtype=np.float32), t_obs=np.array([[0., 1.]], dtype=np.float32), sigma_noise=np.ones((1, 2), dtype=np.float32)), z_lens=.4, z_source=1.5)
    out = run_fermat_ratio_posterior(input_path=catalog, system_index=0, observation=obs,
        sie_fit={"theta_E": 1., "sigma_v": 250., "q": .8}, checkpoint_path=checkpoint, device_name="cpu", samples=8)
    assert out["applied"] and len(out["log_dphi_truth_over_sie"]["samples"]) == 8
    assert "H0" in out["forbidden_inputs"] and "dt_lc" in out["forbidden_inputs"]
