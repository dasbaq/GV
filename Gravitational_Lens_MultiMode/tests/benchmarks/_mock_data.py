"""Mock benchmark data for Phase 1 when real benchmark files are unavailable."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _ou_process(t: np.ndarray, sf_inf: float, tau: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t, dtype=float)
    x[0] = rng.normal(0.0, sf_inf)
    for i in range(1, t.size):
        dt = max(t[i] - t[i - 1], 0.0)
        phi = np.exp(-dt / tau)
        x[i] = phi * x[i - 1] + rng.normal(0.0, sf_inf * np.sqrt(max(1 - phi * phi, 0.0)))
    return x


def make_system6_synthetic(
    dt_true: float = 24.14,
    mu_true: float = 0.7,
    n_epochs: int = 200,
    cadence: float = 3.0,
    snr: float = 50.0,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Create a Bag et al.-style synthetic system 6 light curve."""
    rng = np.random.default_rng(seed)
    t_obs = np.arange(n_epochs, dtype=float) * cadence
    t_full = np.arange(-dt_true - 10 * cadence, t_obs[-1] + cadence, cadence)
    f1_full = 1.0 + _ou_process(t_full, sf_inf=0.3, tau=300.0, seed=seed)
    f1 = np.interp(t_obs, t_full, f1_full)
    f1_delayed = np.interp(t_obs - dt_true, t_full, f1_full)
    clean = f1 + mu_true * f1_delayed
    sigma = np.full_like(clean, np.std(clean) / snr)
    F = clean + rng.normal(0.0, sigma)
    return {
        "F": F,
        "sigma_noise": sigma,
        "t_obs": t_obs,
        "dt_true": float(dt_true),
        "mu_true": float(mu_true),
        "f1_true": f1,
    }


def write_system6_h5(path: Path, **kwargs) -> Path:
    data = make_system6_synthetic(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["full_truth_available"] = True
        lc = h5.create_group("light_curves")
        lc.create_dataset("F_joint", data=data["F"][None, :])
        lc.create_dataset("sigma_noise", data=data["sigma_noise"][None, :])
        lc.create_dataset("t_obs", data=data["t_obs"][None, :])
        lc.create_dataset("n_epochs", data=np.array([len(data["t_obs"])]))
        truth = h5.create_group("true_values")
        truth.create_dataset("dt_true", data=np.array([data["dt_true"]]))
        truth.create_dataset("mu_true", data=np.array([data["mu_true"]]))
    return path
