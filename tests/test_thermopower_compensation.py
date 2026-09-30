from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fk_transport.thermopower_compensation import (
    _transport_distribution,
    analyze_thermopower_compensation,
    extract_zero_crossings,
    match_branch_crossings,
)


def _synthetic_rows() -> list[dict[str, str]]:
    rows = []
    index = 0
    for temperature, typical_shift in ((0.01, 0.10), (0.05, 0.02)):
        for branch, shift in (("arith", 0.0), ("typ", typical_shift)):
            for interaction in (0.0, 1.0):
                crossing = 0.2 + 0.5 * interaction + shift
                for disorder in (0.0, 1.0, 2.0):
                    rows.append(
                        {
                            "index": str(index),
                            "branch": branch,
                            "target_filling": "0.3",
                            "temperature": str(temperature),
                            "interaction": str(interaction),
                            "disorder_full_width": str(disorder),
                            "L12": str(disorder - crossing),
                        }
                    )
                    index += 1
    return rows


class ThermopowerCompensationTests(unittest.TestCase):
    def test_extracts_and_quantifies_branch_convergence(self) -> None:
        crossings = extract_zero_crossings(_synthetic_rows())
        self.assertEqual(len(crossings), 8)
        _, summaries = match_branch_crossings(crossings)
        self.assertEqual(len(summaries), 2)
        self.assertAlmostEqual(float(summaries[0]["rms_separation_over_W"]), 0.10)
        self.assertAlmostEqual(float(summaries[1]["rms_separation_over_W"]), 0.02)

    def test_transport_distribution_splits_l12_by_energy_sign(self) -> None:
        omega = np.linspace(-2.0, 2.0, 4001)
        probability, normalization, negative, positive = _transport_distribution(
            omega, np.ones_like(omega), 0.1
        )
        self.assertGreater(normalization, 0.0)
        self.assertAlmostEqual(float(np.trapz(probability, omega)), 1.0, places=10)
        self.assertAlmostEqual(negative + positive, 0.0, places=10)

    def test_full_summary_analysis_writes_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            rows = _synthetic_rows()
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            outputs = analyze_thermopower_compensation(summary, root / "analysis")
            self.assertTrue(outputs)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertTrue((root / "analysis" / "compensation_separation_n_0p3.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
