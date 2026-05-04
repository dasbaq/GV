from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src_py.simulation.noise_model import photometric_noise
else:
    from .noise_model import photometric_noise


class QuasarLightCurve:
    def __init__(
        self,
        tau_days: float = 200.0,
        sigma_drw_mag: float = 0.1,
        seed: int | None = None,
    ):
        self.tau_days = float(tau_days)
        self.sigma_drw_mag = float(sigma_drw_mag)
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_epochs: int = 200,
        total_days: float = 1000.0,
        survey: str = "ztf",
        cadence_jitter: float = 0.2,
    ) -> dict:
        """
        Generate a DRW/CARMA AR(1) quasar light curve.

        Units: `t_obs` in days; `flux` and `flux_err` in normalized magnitude
        residual units. This is Phase 3 truth-input simulation and does not
        introduce any selectable lens approximation; downstream lensing remains
        under the project-wide fixed SIE standard approximation.
        """
        if n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if total_days <= 0:
            raise ValueError("total_days must be positive")
        if self.tau_days <= 0:
            raise ValueError("tau_days must be positive")
        if self.sigma_drw_mag < 0:
            raise ValueError("sigma_drw_mag must be non-negative")
        if cadence_jitter < 0:
            raise ValueError("cadence_jitter must be non-negative")

        base_grid = np.linspace(0.0, float(total_days), n_epochs, dtype=np.float64)
        nominal_dt = total_days / max(n_epochs - 1, 1)
        jitter = self.rng.uniform(
            -cadence_jitter * nominal_dt,
            cadence_jitter * nominal_dt,
            n_epochs,
        )
        t_obs = np.sort(base_grid + jitter)
        t_obs = np.maximum.accumulate(t_obs + np.arange(n_epochs) * 1e-9)
        t_obs -= t_obs[0]

        clean_flux = np.zeros(n_epochs, dtype=np.float64)
        for i in range(1, n_epochs):
            dt_i = t_obs[i] - t_obs[i - 1]
            phi_i = np.exp(-dt_i / self.tau_days)
            innov_var_i = self.sigma_drw_mag**2 * (1.0 - phi_i**2)
            clean_flux[i] = phi_i * clean_flux[i - 1] + self.rng.normal(
                0.0,
                np.sqrt(max(0.0, innov_var_i)),
            )

        clean_flux = (clean_flux - clean_flux.mean()) / (clean_flux.std() + 1e-12)
        noisy_flux, flux_err = photometric_noise(clean_flux, survey=survey, rng=self.rng)

        return {
            "t_obs": t_obs.astype(np.float32),
            "flux": noisy_flux.astype(np.float32),
            "flux_err": flux_err.astype(np.float32),
            "tau_true": self.tau_days,
            "sigma_true": self.sigma_drw_mag,
            "survey": survey,
        }


def _print_stats(name: str, curve: dict) -> None:
    flux = curve["flux"]
    flux_err = curve["flux_err"]
    print(
        f"{name}: t={curve['t_obs'].shape} flux={flux.shape} flux_err={flux_err.shape} "
        f"mean={flux.mean():.4f} std={flux.std():.4f} "
        f"min={flux.min():.4f} max={flux.max():.4f}"
    )


if __name__ == "__main__":
    n_epochs = 200
    for idx, survey_name in enumerate(["ztf", "lsst", "ideal"]):
        generator = QuasarLightCurve(seed=42 + idx)
        result = generator.generate(n_epochs=n_epochs, survey=survey_name)
        _print_stats(survey_name, result)
        assert result["t_obs"].shape == (n_epochs,)
        assert result["flux"].shape == (n_epochs,)
        assert result["flux_err"].shape == (n_epochs,)
        assert np.all(np.diff(result["t_obs"]) > 0.0)
        assert np.isfinite(result["t_obs"]).all()
        assert np.isfinite(result["flux"]).all()
        assert np.isfinite(result["flux_err"]).all()
        assert not np.isnan(result["flux"]).any()
        assert not np.isnan(result["flux_err"]).any()
    print("PASS")
