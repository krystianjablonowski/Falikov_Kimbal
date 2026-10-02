import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.config import DEFAULTS, validate_config
from fk_transport.impurity import local_green
from fk_transport.sweep import build_tasks


class CorrelatedDisorderTests(unittest.TestCase):
    def test_lambda_zero_reproduces_original_local_green(self):
        omega = np.array([-0.2, 0.1])
        hybridization = np.array([0.03 - 0.2j, -0.04 - 0.15j])
        disorder = np.array([-0.3, 0.2])
        mu, interaction, w1, eta = 0.15, 0.7, 0.4, 1.0e-3
        bath = omega[None, :] + 1j * eta + mu - hybridization[None, :]
        expected = ((1.0 - w1) / (bath - disorder[:, None])
                    + w1 / (bath - disorder[:, None] - interaction))
        actual = local_green(
            omega, hybridization, disorder, mu, interaction, w1, eta, 0.0
        )
        np.testing.assert_array_equal(actual, expected)

    def test_correlated_local_potentials_follow_hamiltonian(self):
        omega = np.array([0.2])
        hybridization = np.array([0.01 - 0.1j])
        disorder = np.array([-0.4, 0.3])
        mu, interaction, w1, eta, lam = 0.25, 0.8, 0.5, 0.002, 0.6
        bath = omega[None, :] + 1j * eta + mu - hybridization[None, :]
        expected = (
            (1.0 - w1) / (bath - (1.0 - lam) * disorder[:, None])
            + w1 / (bath - interaction - (1.0 + lam) * disorder[:, None])
        )
        actual = local_green(
            omega, hybridization, disorder, mu, interaction, w1, eta, lam
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_nonzero_lambda_disables_particle_hole_shortcut(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["model"]["disorder_correlation_lambda"] = 0.5
        cfg["sweep"].update(
            {
                "interactions": [0.5],
                "disorder_full_widths": [0.6],
                "branches": ["arith", "typ"],
                "temperatures": [0.02, 0.05],
                "target_fillings": [0.5],
            }
        )
        validate_config(cfg)
        tasks = build_tasks(cfg)
        self.assertEqual(len(tasks), 4)
        self.assertTrue(all(not task["half_filling"] for task in tasks))
        self.assertTrue(all(task["disorder_correlation_lambda"] == 0.5 for task in tasks))

    def test_nonfinite_lambda_is_rejected(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["model"]["disorder_correlation_lambda"] = float("inf")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_config(cfg)

    def test_lambda_sweep_is_a_separate_task_dimension(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["sweep"].update(
            {
                "interactions": [0.5],
                "disorder_full_widths": [0.6],
                "disorder_correlation_lambdas": [0.1, 0.5, 1.5],
                "branches": ["arith", "typ"],
                "temperatures": [0.05],
                "target_fillings": [0.4],
            }
        )
        validate_config(cfg)
        tasks = build_tasks(cfg)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(
            {task["disorder_correlation_lambda"] for task in tasks},
            {0.1, 0.5, 1.5},
        )


if __name__ == "__main__":
    unittest.main()
