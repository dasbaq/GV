"""Small config loader for Phase 2 physics modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


_DEFAULTS: dict[str, Any] = {
    "constants": {
        "c_km_s": 299792.458,
        "c_m_s": 299792458.0,
        "G_si": 6.67430e-11,
        "M_sun_kg": 1.98847e30,
        "Mpc_m": 3.0856775814913673e22,
        "kpc_m": 3.0856775814913673e19,
        "arcsec_to_rad": 4.84813681109536e-6,
        "rad_to_arcsec": 206264.80624709636,
        "day_s": 86400.0,
    },
    "cosmology": {"H0": 70.0, "Omega_m": 0.3, "Omega_lambda": 0.7},
    "numerics": {"eps": 1.0e-12, "integration_n": 4096, "ray_step_m": 1.0e19, "ray_n_steps": 2048},
}


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_physics_config() -> dict[str, Any]:
    """Load ``config/physics.yaml`` with fallback defaults."""
    path = Path(__file__).resolve().parents[2] / "config" / "physics.yaml"
    data: dict[str, Any] = {}
    try:
        import yaml

        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except Exception:
        data = {}

    cfg = _merge(_DEFAULTS, data)
    constants = cfg.setdefault("constants", {})
    constants.setdefault("c_km_s", data.get("c_km_s", _DEFAULTS["constants"]["c_km_s"]))
    constants.setdefault("c_m_s", data.get("c_m_s", _DEFAULTS["constants"]["c_m_s"]))
    constants.setdefault("G_si", data.get("G", _DEFAULTS["constants"]["G_si"]))
    constants.setdefault("M_sun_kg", data.get("M_sun", _DEFAULTS["constants"]["M_sun_kg"]))
    constants.setdefault("Mpc_m", data.get("Mpc_m", _DEFAULTS["constants"]["Mpc_m"]))
    constants.setdefault("arcsec_to_rad", data.get("arcsec_rad", _DEFAULTS["constants"]["arcsec_to_rad"]))
    constants.setdefault("day_s", data.get("days_s", _DEFAULTS["constants"]["day_s"]))
    constants.setdefault("rad_to_arcsec", 1.0 / constants["arcsec_to_rad"])
    for group in ("constants", "cosmology", "numerics"):
        for key, value in list(cfg[group].items()):
            if isinstance(value, str):
                try:
                    cfg[group][key] = float(value)
                except ValueError:
                    pass
    return cfg


def constants() -> dict[str, float]:
    return load_physics_config()["constants"]


def numerics() -> dict[str, float]:
    return load_physics_config()["numerics"]


def default_cosmology() -> dict[str, float]:
    return load_physics_config()["cosmology"]
