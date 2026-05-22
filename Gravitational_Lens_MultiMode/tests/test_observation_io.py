from pathlib import Path

import numpy as np
import pytest

from inversion.observation_io import ObservedLensSystem, from_dict, from_hdf5


def _observation_mapping() -> dict:
    return {
        "name": "RXJ1131-like",
        "image_positions": [[0.8, 0.2], [-0.6, 0.1]],
        "light_curves": {
            "F": [[1.0, 0.9, 1.1], [0.7, 0.65, 0.75]],
            "t_obs": [0.0, 1.0, 2.0],
            "sigma_noise": [[0.02, 0.02, 0.03], [0.03, 0.03, 0.04]],
        },
        "z_lens": 0.31,
        "z_source": 0.66,
        "image_fluxes": [1.0, 0.7],
        "M200": 1.0e13,
        "concentration": 8.0,
        "kappa_ext": 0.02,
        "nfw_offset": [0.1, 0.1],
    }


def test_from_dict_roundtrip_observation_units_and_shapes():
    system = from_dict(_observation_mapping())
    assert isinstance(system, ObservedLensSystem)
    assert system.name == "RXJ1131-like"
    assert system.image_positions.shape == (2, 2)
    assert system.light_curves.F.shape == (2, 3)
    assert system.light_curves.t_obs.shape == (2, 3)
    assert system.light_curves.sigma_noise.shape == (2, 3)
    np.testing.assert_allclose(system.image_fluxes, [1.0, 0.7])

    roundtripped = from_dict(system.to_dict())
    np.testing.assert_allclose(roundtripped.image_positions, system.image_positions)
    np.testing.assert_allclose(roundtripped.light_curves.F, system.light_curves.F)
    assert roundtripped.z_source > roundtripped.z_lens + 0.05


def test_from_hdf5_loads_existing_simulation_schema():
    mock_path = Path("data/mock/phase4_v0_2.h5")
    if not mock_path.exists():
        pytest.skip("local mock HDF5 artifact is not available")

    system = from_hdf5(mock_path, system_index=0)
    assert system.image_positions.shape == (2, 2)
    assert system.light_curves.F.ndim == 2
    assert system.light_curves.F.shape[0] == 1
    assert system.light_curves.F.shape == system.light_curves.t_obs.shape
    assert system.light_curves.F.shape == system.light_curves.sigma_noise.shape
    assert system.z_source > system.z_lens + 0.05


def test_from_dict_rejects_invalid_redshift_order():
    mapping = _observation_mapping()
    mapping["z_lens"] = 0.6
    mapping["z_source"] = 0.62
    with pytest.raises(ValueError, match="z_source"):
        from_dict(mapping)


def test_from_dict_rejects_single_image():
    mapping = _observation_mapping()
    mapping["image_positions"] = [[0.8, 0.2]]
    mapping["image_fluxes"] = [1.0]
    with pytest.raises(ValueError, match="at least two image"):
        from_dict(mapping)
