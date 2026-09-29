import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.config import DEFAULTS
from fk_transport.sweep import build_tasks


class SweepTests(unittest.TestCase):
    def test_half_filling_temperatures_share_one_spectral_task(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["sweep"].update(
            {
                "interactions": [0.0, 0.3],
                "disorder_full_widths": [0.0, 1.0],
                "branches": ["arith", "typ"],
                "temperatures": [0.01, 0.02, 0.04],
                "target_fillings": [0.5],
            }
        )
        tasks = build_tasks(cfg)
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task["temperatures"] == [0.01, 0.02, 0.04] for task in tasks))
        self.assertEqual([task["index"] for task in tasks], list(range(8)))

    def test_irregular_parameter_points_do_not_form_cartesian_product(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["sweep"].update(
            {
                "parameter_points": [
                    {"interaction": 0.2, "disorder_full_width": 0.7},
                    {"interaction": 0.4, "disorder_full_width": 1.1},
                ],
                "branches": ["arith", "typ"],
                "temperatures": [0.01, 0.02],
                "target_fillings": [0.5],
            }
        )
        tasks = build_tasks(cfg)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            {(task["interaction"], task["disorder_full_width"]) for task in tasks},
            {(0.2, 0.7), (0.4, 1.1)},
        )


if __name__ == "__main__":
    unittest.main()
