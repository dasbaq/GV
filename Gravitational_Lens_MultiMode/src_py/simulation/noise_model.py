from __future__ import annotations

import numpy as np


def photometric_noise(
    flux_normalized: np.ndarray,
    survey: str = "ztf",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns noisy flux and per-epoch sigma in normalized magnitude residual units.
    """
    rng = rng or np.random.default_rng()
    flux = np.asarray(flux_normalized, dtype=np.float64)
    survey_key = survey.lower()

    if survey_key == "ztf":
        sigma_floor = 0.02
        sigma_bright = 0.005
        clip = (0.005, 0.10)
        sigma_mag = np.sqrt(sigma_floor**2 + sigma_bright**2 * 10.0 ** (0.8 * flux))
        sigma = np.clip(sigma_mag, clip[0], clip[1])
    elif survey_key == "lsst":
        sigma_floor = 0.005
        sigma_bright = 0.001
        clip = (0.001, 0.05)
        sigma_mag = np.sqrt(sigma_floor**2 + sigma_bright**2 * 10.0 ** (0.8 * flux))
        sigma = np.clip(sigma_mag, clip[0], clip[1])
    elif survey_key == "ideal":
        sigma = np.full_like(flux, 0.01, dtype=np.float64)
    else:
        raise ValueError("survey must be one of: ztf, lsst, ideal")

    noisy_flux = flux + rng.normal(0.0, sigma, size=flux.shape)
    return noisy_flux.astype(np.float32), sigma.astype(np.float32)


if __name__ == "__main__":
    base = np.linspace(-2.0, 2.0, 200, dtype=np.float32)
    for i, survey_name in enumerate(["ztf", "lsst", "ideal"]):
        noisy, sigma = photometric_noise(base, survey=survey_name, rng=np.random.default_rng(42 + i))
        print(
            f"{survey_name}: noisy={noisy.shape} sigma={sigma.shape} "
            f"sigma_min={sigma.min():.4f} sigma_max={sigma.max():.4f}"
        )
        assert noisy.shape == base.shape
        assert sigma.shape == base.shape
        assert np.isfinite(noisy).all()
        assert np.isfinite(sigma).all()
        assert not np.isnan(noisy).any()
        assert not np.isnan(sigma).any()
    print("PASS")
