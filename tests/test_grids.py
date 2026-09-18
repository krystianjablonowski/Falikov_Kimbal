import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.grids import assert_frequency_grid, frequency_grid


class GridTests(unittest.TestCase):
    def test_odd_symmetric_grid_contains_exact_zero(self):
        omega = frequency_grid(10.0, 2021)
        self.assertEqual(omega.size, 2021)
        self.assertEqual(omega[1010], 0.0)
        self.assertTrue(np.array_equal(omega, -omega[::-1]))
        assert_frequency_grid(omega)

    def test_even_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            frequency_grid(1.0, 2020)


if __name__ == "__main__":
    unittest.main()
