import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.boundary_refinement import generate_boundary_config
from fk_transport.config import DEFAULTS, load_config


class BoundaryRefinementTests(unittest.TestCase):
    def test_steep_cell_generates_irregular_new_points(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = root / "summary.csv"
            fields = [
                "branch", "temperature", "target_filling", "interaction",
                "disorder_full_width", "sigma",
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for interaction in (0.0, 1.0, 2.0):
                    for disorder in (0.0, 1.0, 2.0):
                        writer.writerow(
                            {
                                "branch": "typ", "temperature": 0.01,
                                "target_filling": 0.5, "interaction": interaction,
                                "disorder_full_width": disorder,
                                "sigma": 1.0 if disorder < 1.5 else 1.0e-6,
                            }
                        )
            base = copy.deepcopy(DEFAULTS)
            base["_config_path"] = str(root / "base.json")
            output = root / "stage2.json"
            report = generate_boundary_config(
                base, summary, output, fields=["sigma"], u_step_ratio=0.25,
                disorder_step_ratio=0.25, log_jump=2.0, padding_cells=0,
            )
            generated = load_config(output)
            points = generated["sweep"]["parameter_points"]
            self.assertGreater(len(points), 0)
            self.assertEqual(report["parameter_points"], len(points))
            self.assertTrue(all(1.0 <= point["disorder_full_width"] <= 2.0 for point in points))
            self.assertNotIn(
                {"interaction": 0.0, "disorder_full_width": 1.0}, points
            )


if __name__ == "__main__":
    unittest.main()
