from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fk_transport.plotting import (
    plot_all_temperature_summaries,
    plot_summary,
    plot_transport_heatmaps,
)


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
                "c_v_electronic",
                "K0_thermo",
                "lorenz_over_L0",
                "charge_diffusivity_proxy",
                "thermal_diffusivity_proxy",
                "transport_energy_variance",
                "iterations",
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for temperature in (0.01, 0.02):
                    for branch in ("arith", "typ"):
                        for interaction in (0.0, 0.5):
                            for disorder in (0.0, 1.0):
                                base = 1.0 + interaction + disorder + temperature
                                writer.writerow(
                                    {
                                        "branch": branch,
                                        "interaction": interaction,
                                        "disorder_full_width": disorder,
                                        "temperature": temperature,
                                        "target_filling": 0.5,
                                        "sigma_T0": base,
                                        "sigma": 0.5 * base,
                                        "kappa_e": 0.01 * base,
                                        "c_v_electronic": 0.02 * base,
                                        "K0_thermo": 0.3 * base,
                                        "lorenz_over_L0": base,
                                        "charge_diffusivity_proxy": 0.4 * base,
                                        "thermal_diffusivity_proxy": 0.5 * base,
                                        "transport_energy_variance": 0.1 * base,
                                        "iterations": 20 + int(10 * disorder),
                                    }
                                )
            outputs = plot_transport_heatmaps(summary)
            self.assertEqual(len(outputs), 9)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertTrue(all(path.suffix == ".pdf" for path in outputs))
            self.assertFalse(any(root.glob("*.png")))
            overview = plot_summary(summary)
            self.assertTrue(overview.is_file() and overview.stat().st_size > 0)
            self.assertEqual(overview.suffix, ".pdf")
            summaries = plot_all_temperature_summaries(summary)
            self.assertEqual(len(summaries), 2)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in summaries))
            self.assertFalse(any(root.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
