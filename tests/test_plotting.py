from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fk_transport.plotting import plot_summary, plot_transport_heatmaps


class PlottingTests(unittest.TestCase):
    def test_regular_grid_creates_zero_and_finite_temperature_maps(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is optional")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            fieldnames = [
                "branch",
                "interaction",
                "disorder_full_width",
                "temperature",
                "target_filling",
                "sigma_T0",
                "sigma",
                "kappa_e",
                "lorenz_over_L0",
                "iterations",
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for branch in ("arith", "typ"):
                    for interaction in (0.0, 0.5):
                        for disorder in (0.0, 1.0):
                            writer.writerow(
                                {
                                    "branch": branch,
                                    "interaction": interaction,
                                    "disorder_full_width": disorder,
                                    "temperature": 0.01,
                                    "target_filling": 0.5,
                                    "sigma_T0": 1.0 + interaction + disorder,
                                    "sigma": 0.5 + interaction + disorder,
                                    "kappa_e": 0.01 + interaction + disorder,
                                    "lorenz_over_L0": 1.0 + interaction,
                                    "iterations": 20 + int(10 * disorder),
                                }
                            )
            outputs = plot_transport_heatmaps(summary)
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertTrue(all(path.with_suffix(".pdf").is_file() for path in outputs))
            overview = plot_summary(summary)
            self.assertTrue(overview.is_file() and overview.stat().st_size > 0)
            self.assertTrue(overview.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
