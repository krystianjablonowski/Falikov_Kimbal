from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from fk_transport.config import DEFAULTS
from fk_transport.localization_convergence import (
    analyze_localization_convergence,
    prepare_localization_convergence,
    select_localization_points,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LocalizationConvergenceTests(unittest.TestCase):
    def test_selects_neighbors_around_ratio_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.csv"
            rows = []
            for interaction in (0.75, 1.5):
                for disorder, ratio in ((0.5, 0.8), (1.0, 0.12), (1.5, 0.02), (2.0, 0.001)):
                    rows.append({
                        "branch": "typ", "target_filling": 0.4, "temperature": 0.02,
                        "interaction": interaction, "disorder_full_width": disorder,
                        "rho_typ_over_arith_zero": ratio,
                    })
            _write(path, rows)
            selected = select_localization_points(path, 0.4, 0.02, [0.75, 1.5], 0.1, 1)
            self.assertEqual(len(selected), 6)
            self.assertIn((1.5, 1.0), selected)

    def test_prepares_four_small_configs_and_analyzes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostics = root / "diagnostics.csv"
            rows = []
            for disorder, ratio in ((0.5, 0.8), (1.0, 0.1), (1.5, 0.01)):
                rows.append({
                    "branch": "typ", "target_filling": 0.4, "temperature": 0.02,
                    "interaction": 1.5, "disorder_full_width": disorder,
                    "rho_typ_over_arith_zero": ratio,
                })
            _write(diagnostics, rows)
            report = prepare_localization_convergence(
                DEFAULTS, diagnostics, root / "scan", 0.4, 0.02, [1.5], 0.1, 1
            )
            self.assertEqual(len(report["configs"]), 4)
            summaries, configs = [], []
            for number, item in enumerate(report["configs"][:2]):
                config = Path(item["config"])
                summary = root / f"summary_{number}.csv"
                summary_rows = []
                for disorder in (0.5, 1.0, 1.5):
                    summary_rows.append({
                        "branch": "typ", "target_filling": 0.4, "temperature": 0.02,
                        "interaction": 1.5, "disorder_full_width": disorder,
                        "rho_arith_zero": 1.0, "rho_typ_zero": 0.1 / (1.0 + disorder + number),
                        "sigma": 0.01 / (1.0 + disorder), "chemical_potential": 0.2,
                        "final_residual": 1.0e-9, "filling_error": 1.0e-7,
                        "iterations": 20, "status": "success",
                    })
                _write(summary, summary_rows)
                summaries.append(summary)
                configs.append(config)
            products = analyze_localization_convergence(summaries, configs, root / "analysis")
            self.assertTrue(all(path.is_file() for path in products))
            with (root / "analysis/localization_convergence_spread.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                spread = list(csv.DictReader(handle))
            self.assertEqual(len(spread), 3)
            self.assertTrue(all(row["all_success"] == "True" for row in spread))


if __name__ == "__main__":
    unittest.main()
