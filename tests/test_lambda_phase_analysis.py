from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fk_transport.lambda_phase_analysis import (
    _load_records,
    analyze_lambda_phase,
    linearized_typical_log_multiplier,
    phase_fraction_rows,
    reentrant_rows,
)


class LambdaPhaseAnalysisTests(unittest.TestCase):
    def test_zero_disorder_multiplier_matches_two_local_levels(self) -> None:
        interaction = 0.8
        chemical_potential = 0.3
        expected_factor = (
            0.5 / chemical_potential**2
            + 0.5 / (chemical_potential - interaction) ** 2
        )
        expected = np.log(0.25**2 * expected_factor)
        actual = linearized_typical_log_multiplier(
            interaction=interaction,
            disorder_full_width=0.0,
            correlation_lambda=0.0,
            chemical_potential=chemical_potential,
            hybridization_zero=0.0j,
            bandwidth=1.0,
            w1=0.5,
        )
        self.assertAlmostEqual(actual, expected)

    def test_fraction_uses_full_common_grid_and_reports_missing_points(self) -> None:
        records = []
        for interaction in (0.0, 1.0):
            for disorder in (0.0, 1.0):
                records.append(
                    {
                        "disorder_correlation_lambda": 0.0,
                        "interaction": interaction,
                        "disorder_full_width": disorder,
                        "metal": interaction == disorder,
                    }
                )
        records.extend(
            {
                "disorder_correlation_lambda": 1.0,
                "interaction": interaction,
                "disorder_full_width": disorder,
                "metal": True,
            }
            for interaction, disorder in ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        )
        rows = phase_fraction_rows(records)
        self.assertEqual(rows[0]["metal_components"], 2)
        self.assertEqual(rows[0]["metal_fraction"], 0.5)
        self.assertTrue(rows[0]["regular_grid_complete"])
        self.assertEqual(rows[1]["missing_points"], 1)
        self.assertEqual(rows[1]["metal_fraction"], 0.75)
        self.assertFalse(rows[1]["regular_grid_complete"])

    def test_reentrant_insulator_metal_insulator_is_detected(self) -> None:
        records = [
            {
                "interaction": 1.0,
                "disorder_full_width": 1.5,
                "disorder_correlation_lambda": correlation_lambda,
                "log_lambda_typ": value,
            }
            for correlation_lambda, value in ((0.0, -1.0), (1.0, 2.0), (2.0, -1.0))
        ]
        result = reentrant_rows(records)[0]
        self.assertTrue(result["reentrant_AI_metal_AI"])
        self.assertEqual(result["phase_sequence"], "I-M-I")
        self.assertEqual(result["zero_crossings"], 2)
        crossings = [float(value) for value in result["crossing_lambdas"].split(";")]
        np.testing.assert_allclose(crossings, [1.0 / 3.0, 5.0 / 3.0])

    def test_legacy_summary_without_lambda_is_loaded_as_lambda_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = root / "points"
            point = points / "point_000007"
            point.mkdir(parents=True)
            np.savez_compressed(
                point / "solution.npz",
                omega=np.array([-0.1, 0.0, 0.1]),
                hybridization=np.array([-0.1j, -0.2j, -0.1j]),
            )
            summary = root / "summary.csv"
            rows = [
                {
                    "index": 7,
                    "branch": "typ",
                    "target_filling": 0.4,
                    "temperature": 0.05,
                    "interaction": 1.0,
                    "disorder_full_width": 1.5,
                    "chemical_potential": 0.2,
                    "rho_typ_zero": 0.01,
                    "rho_arith_zero": 0.1,
                    "sigma": 0.02,
                }
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records = _load_records(
                [summary], [points], 0.4, 0.05, 1.0, 0.5, 32
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["disorder_correlation_lambda"], 0.0)
        self.assertTrue(np.isfinite(records[0]["log_lambda_typ"]))

    def test_full_analysis_writes_tables_report_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = root / "points"
            rows = []
            index = 0
            for correlation_lambda in (0.0, 1.0):
                for interaction in (0.5, 1.0):
                    for disorder in (0.5, 1.0):
                        point = points / f"point_{index:06d}"
                        point.mkdir(parents=True)
                        np.savez_compressed(
                            point / "solution.npz",
                            omega=np.array([-0.1, 0.0, 0.1]),
                            hybridization=np.array([-0.1j, -0.2j, -0.1j]),
                        )
                        rows.append(
                            {
                                "index": index,
                                "branch": "typ",
                                "target_filling": 0.4,
                                "temperature": 0.05,
                                "interaction": interaction,
                                "disorder_full_width": disorder,
                                "disorder_correlation_lambda": correlation_lambda,
                                "chemical_potential": 0.2,
                                "rho_typ_zero": 0.01,
                                "rho_arith_zero": 0.1,
                                "sigma": 0.02,
                            }
                        )
                        index += 1
            summary = root / "summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            products = analyze_lambda_phase(
                [summary], [points], root / "analysis", quadrature_order=16
            )
            self.assertEqual(len(products), 7)
            self.assertTrue(all(path.is_file() for path in products))


if __name__ == "__main__":
    unittest.main()
