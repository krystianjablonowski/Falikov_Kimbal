import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.config import DEFAULTS
from fk_transport.solver import solve_medium


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["grid"] = {"omega_max": 2.0, "n_omega": 1001}
        self.cfg["numerics"].update(
            {
                "broadening": 0.005,
                "tolerance": 1e-8,
                "max_iterations": 400,
                "hilbert_padding_factor": 16,
            }
        )
        self.cfg["sweep"]["temperatures"] = [0.02]

    def test_clean_arithmetic_limit(self):
        result = solve_medium(self.cfg, 0.0, 0.0, "arith", chemical_potential=0.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "success")
        self.assertLess(result.metrics["final_residual"], self.cfg["numerics"]["tolerance"] * 1.1)
        self.assertGreater(result.metrics["rho_zero"], 0)

    def test_particle_hole_symmetry(self):
        result = solve_medium(self.cfg, 0.3, 0.4, "arith", chemical_potential=0.15)
        relative = np.max(np.abs(result.rho_arith - result.rho_arith[::-1])) / np.max(result.rho_arith)
        self.assertLess(relative, 1e-10)


if __name__ == "__main__":
    unittest.main()
