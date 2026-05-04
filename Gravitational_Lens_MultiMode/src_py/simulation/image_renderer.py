from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def render_lensed_image(
    lens,
    beta: np.ndarray,
    image_size: int = 64,
    pixel_scale_arcsec: float = 0.05,
    psf_fwhm_arcsec: float = 0.1,
    source_size_arcsec: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render lensed image of a Gaussian source via inverse ray tracing and Gaussian PSF.

    Units: image-plane and source-plane coordinates are arcsec; output is normalized
    dimensionless surface brightness in [0, 1]. Uses the project-wide SIE lens
    convention where beta(theta) = theta - alpha(theta).
    """
    del rng
    beta_arr = np.asarray(beta, dtype=np.float64)
    if beta_arr.shape != (2,):
        raise ValueError("beta must have shape (2,) in arcsec")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if pixel_scale_arcsec <= 0 or psf_fwhm_arcsec <= 0 or source_size_arcsec <= 0:
        raise ValueError("pixel/source/PSF scales must be positive")

    fov = image_size * pixel_scale_arcsec
    axis = np.linspace(-fov / 2.0, fov / 2.0, image_size, dtype=np.float64)
    theta_y, theta_x = np.meshgrid(axis, axis, indexing="ij")
    theta = np.stack([theta_x.ravel(), theta_y.ravel()], axis=-1)

    beta_pixel = theta - lens.deflection(theta)
    delta = beta_pixel - beta_arr
    image = np.exp(-0.5 * np.sum(delta * delta, axis=1) / source_size_arcsec**2)
    image = image.reshape(image_size, image_size)

    sigma_pix = psf_fwhm_arcsec / (2.355 * pixel_scale_arcsec)
    image = gaussian_filter(image, sigma=sigma_pix, mode="nearest")
    max_val = float(np.max(image))
    if max_val > 0.0:
        image = image / max_val
    return np.clip(image, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    from core.physics.lens_models import SIELens

    lens = SIELens(sigma_v=220.0, q=0.7, position_angle=0.3, z_lens=0.5, z_source=2.0)
    rendered = render_lensed_image(lens, np.array([0.1, 0.05], dtype=np.float32))
    assert rendered.shape == (64, 64)
    assert np.isfinite(rendered).all()
    assert not np.isnan(rendered).any()
    assert abs(float(rendered.max()) - 1.0) < 1e-5
    assert float(rendered.min()) >= 0.0
    print("PASS")
