import numpy as np

from fk_transport.plotting import _grid


def test_grid_uses_only_selected_filling_parameter_axes():
    rows = [
        {
            "branch": "arith", "temperature": "0.05", "target_filling": "0.3",
            "interaction": "1.0", "disorder_full_width": "0.0", "L12": "-1.0",
        },
        {
            "branch": "arith", "temperature": "0.05", "target_filling": "0.3",
            "interaction": "1.0", "disorder_full_width": "1.0", "L12": "1.0",
        },
        {
            "branch": "arith", "temperature": "0.05", "target_filling": "0.4",
            "interaction": "0.9375", "disorder_full_width": "0.5", "L12": "-2.0",
        },
        {
            "branch": "arith", "temperature": "0.05", "target_filling": "0.4",
            "interaction": "0.9375", "disorder_full_width": "0.9", "L12": "2.0",
        },
    ]
    disorder, interaction, values = _grid(
        rows, "arith", 0.05, 0.4, "L12", positive_only=False
    )
    np.testing.assert_allclose(disorder, [0.5, 0.9])
    np.testing.assert_allclose(interaction, [0.9375])
    np.testing.assert_allclose(values, [[-2.0, 2.0]])
