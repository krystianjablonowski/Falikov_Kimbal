import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.grids import frequency_grid
from fk_transport.observables import minus_fermi_derivative, transport_observables


class ObservableTests(unittest.TestCase):
    def test_fermi_derivative_is_stable_and_normalized(self):
        omega = frequency_grid(2.0, 20001)
        derivative = minus_fermi_derivative(omega, 0.02)
        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertAlmostEqual(float(np.trapezoid(derivative, omega)), 1.0, places=8)

    def test_constant_tau_has_wiedemann_franz_limit(self):
        omega = frequency_grid(2.0, 20001)
        result = transport_observables(omega, np.ones_like(omega), 0.02)
        self.assertAlmostEqual(float(result["L12"]), 0.0, places=12)
        self.assertAlmostEqual(float(result["lorenz"]), np.pi**2 / 3.0, places=8)
        self.assertTrue(result["cauchy_schwarz_ok"])


if __name__ == "__main__":
    unittest.main()
