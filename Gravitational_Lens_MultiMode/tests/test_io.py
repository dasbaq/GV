from pathlib import Path

import numpy as np

from core.light_curve.io import load_light_curve, save_extraction_results
from tests.benchmarks._mock_data import write_system6_h5


def test_light_curve_hdf5_roundtrip(tmp_path: Path):
    input_path = write_system6_h5(tmp_path / "system6.h5", n_epochs=12)
    loaded = load_light_curve(input_path, system_idx=0)
    assert loaded["n_epochs"] == 12
    assert loaded["F"].shape == (12,)

    result = {
        "dt": 24.1,
        "dt_uncertainty": 0.1,
        "mu": 0.7,
        "mu_uncertainty": 0.02,
        "confidence_grade": "conservative",
        "sigma_min": -3.0,
        "grid": {
            "dt_grid": np.array([24.0, 24.1]),
            "mu_grid": np.array([-0.7, 0.7]),
            "sigma_map": np.zeros((2, 2)),
        },
    }
    out = tmp_path / "result.h5"
    save_extraction_results(out, [result], {"io": {"compression": "gzip"}}, True)
    assert out.exists()
