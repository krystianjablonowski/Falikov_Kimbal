import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.bethe import bethe_dos, bethe_green
from fk_transport.grids import band_quadrature


class BetheTests(unittest.TestCase):
    def test_dos_normalization_for_general_bandwidth(self):
        for d in (0.5, 1.0, 2.0):
            eps, weights = band_quadrature(d, 512)
            self.assertAlmostEqual(float(np.sum(weights * bethe_dos(eps, d))), 1.0, places=8)

    def test_retarded_green_has_negative_imaginary_part(self):
        z = np.linspace(-0.4, 0.4, 11) + 1e-3j
        self.assertTrue(np.all(bethe_green(z, 0.5).imag < 0))


if __name__ == "__main__":
    unittest.main()
