from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fk_transport.finite_filling_analysis import (
    _plot_thermopower_zero_crossing_maps,
    derive_finite_filling_rows,
)


class FiniteFillingAnalysisTests(unittest.TestCase):
    def test_derives_ratios_performance_and_coupled_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / "summary.csv"
            fields = [
                "branch", "target_filling", "temperature", "interaction",
                "disorder_full_width", "sigma", "kappa_e", "thermopower",
                "L11", "L12", "L22", "K0_thermo", "K1_thermo", "K2_thermo",
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for branch, factor in (("arith", 1.0), ("typ", 0.1)):
                    writer.writerow(
                        {
                            "branch": branch,
                            "target_filling": 0.4,
                            "temperature": 0.02,
                            "interaction": 1.0,
                            "disorder_full_width": 1.5,
                            "sigma": 2.0 * factor,
                            "kappa_e": 4.0 * factor,
                            "thermopower": -3.0,
                            "L11": 2.0 * factor,
                            "L12": 0.0,
                            "L22": 8.0 * factor,
                            "K0_thermo": 1.0,
                            "K1_thermo": 0.0,
                            "K2_thermo": 4.0,
                        }
                    )
            rows = derive_finite_filling_rows(summary)
            typical = next(row for row in rows if row["branch"] == "typ")
            self.assertAlmostEqual(float(typical["relative_sigma"]), 0.1)
            self.assertAlmostEqual(float(typical["relative_kappa"]), 0.1)
            self.assertAlmostEqual(float(typical["power_factor"]), 1.8)
            self.assertAlmostEqual(float(typical["zt_electronic"]), 0.09)
            self.assertAlmostEqual(float(typical["coupled_diffusivity_minus"]), 0.2)
            self.assertAlmostEqual(float(typical["coupled_diffusivity_plus"]), 0.2)

    def test_plots_signed_thermopower_and_l12_with_zero_crossings(self) -> None:
        rows = []
        for branch, factor in (("arith", 1.0), ("typ", 0.1)):
            for interaction in (0.0, 1.0):
                for disorder in (0.0, 1.0):
                    signed_value = factor * (disorder - interaction + 0.25)
                    rows.append(
                        {
                            "branch": branch,
                            "target_filling": "0.3",
                            "temperature": "0.01",
                            "interaction": str(interaction),
                            "disorder_full_width": str(disorder),
                            "thermopower": str(signed_value),
                            "L12": str(0.01 * signed_value),
                        }
                    )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "zero_crossings.png"
            result = _plot_thermopower_zero_crossing_maps(rows, 0.3, 0.01, output)
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
