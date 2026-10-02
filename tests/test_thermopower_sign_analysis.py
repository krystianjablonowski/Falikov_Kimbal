from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fk_transport.thermopower_sign_analysis import (
    analyze_thermopower_sign,
    thermopower_sign_diagnostics,
)


def _curve(
    filling: float,
    branch: str,
    thermopowers: tuple[float, ...],
    temperature: float = 0.05,
) -> list[dict[str, str]]:
    return [
        {
            "branch": branch,
            "target_filling": str(filling),
            "temperature": str(temperature),
            "interaction": str(interaction),
            "disorder_full_width": "0.0",
            "disorder_correlation_lambda": "0.0",
            "thermopower": str(thermopower),
            "sigma": "1.0",
        }
        for interaction, thermopower in enumerate(thermopowers)
    ]


class ThermopowerSignAnalysisTests(unittest.TestCase):
    def test_forced_crossing_and_below_threshold_control(self) -> None:
        rows = _curve(0.3, "arith", (-1.0, 1.0))
        rows += _curve(0.2, "arith", (-1.0, -0.5))
        diagnostics, crossings = thermopower_sign_diagnostics(rows)
        by_filling = {row["target_filling"]: row for row in diagnostics}
        self.assertTrue(by_filling[0.3]["prediction_met"])
        self.assertEqual(by_filling[0.3]["zero_crossings"], 1)
        self.assertTrue(by_filling[0.2]["prediction_met"])
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0]["interaction"], 0.5)

    def test_localized_tail_is_not_used_as_large_u_evidence(self) -> None:
        rows = _curve(0.3, "typ", (-1.0, -0.2, 1.0))
        rows[-1]["sigma"] = "1e-12"
        diagnostics, crossings = thermopower_sign_diagnostics(
            rows, conductivity_relative_floor=1.0e-8
        )
        self.assertFalse(diagnostics[0]["prediction_met"])
        self.assertEqual(diagnostics[0]["maximum_reliable_interaction"], 1.0)
        self.assertEqual(crossings, [])

    def test_full_analysis_writes_tables_report_and_plots(self) -> None:
        rows = []
        for branch in ("arith", "typ"):
            rows += _curve(0.2, branch, (-1.0, -0.7, -0.4))
            rows += _curve(0.3, branch, (-1.0, -0.2, 0.8))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            products = analyze_thermopower_sign([summary], root / "analysis")
            self.assertEqual(len(products), 5)
            self.assertTrue(all(path.is_file() for path in products))


if __name__ == "__main__":
    unittest.main()
