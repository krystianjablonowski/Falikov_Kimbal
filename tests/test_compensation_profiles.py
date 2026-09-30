from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fk_transport.compensation_profiles import (
    plot_compensation_profiles,
    select_profile_points,
)


def _rows() -> list[dict[str, str]]:
    rows = []
    index = 0
    for branch in ("arith", "typ"):
        for disorder in (0.0, 0.5, 1.0, 1.5):
            sigma_arith = np.exp(-disorder)
            ratio = 1.0 if branch == "arith" else 10.0 ** (-3.0 * disorder)
            sigma = sigma_arith * ratio
            l12 = disorder - (0.75 if branch == "arith" else 0.80)
            rows.append(
                {
                    "index": str(index),
                    "branch": branch,
                    "target_filling": "0.3",
                    "temperature": "0.05",
                    "interaction": "1.0",
                    "disorder_full_width": str(disorder),
                    "sigma": str(sigma),
                    "thermopower": str(-l12 / (0.05 * sigma)),
                    "L12": str(l12),
                }
            )
            index += 1
    return rows


class CompensationProfileTests(unittest.TestCase):
    def test_selects_separate_interpolated_compensation_points(self) -> None:
        selected = select_profile_points(_rows(), [1.0])
        self.assertEqual(
            {row["selection"] for row in selected},
            {
                "metallic", "max_abs_S_typ", "compensation_arith",
                "compensation_typ", "localized_edge",
            },
        )
        by_kind = {row["selection"]: row for row in selected}
        arithmetic = by_kind["compensation_arith"]
        typical = by_kind["compensation_typ"]
        self.assertAlmostEqual(float(arithmetic["disorder_full_width"]), 0.75)
        self.assertAlmostEqual(float(arithmetic["L12_arith"]), 0.0)
        self.assertAlmostEqual(float(typical["disorder_full_width"]), 0.80)
        self.assertAlmostEqual(float(typical["L12_typ"]), 0.0)
        self.assertEqual(arithmetic["point_index_arith_low"], 1)
        self.assertEqual(arithmetic["point_index_arith_high"], 2)

    def test_renders_profiles_from_saved_tau(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            rows = _rows()
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            points = root / "points"
            omega = np.linspace(-1.0, 1.0, 1001)
            for row in rows:
                point = points / f"point_{int(row['index']):06d}"
                point.mkdir(parents=True)
                tau = np.exp(-((omega - 0.1 * int(row["index"])) / 0.4) ** 2)
                np.savez_compressed(point / "solution.npz", omega=omega, tau=tau)
            outputs = plot_compensation_profiles(
                summary, points, root / "analysis", [1.0]
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertTrue(outputs[1].with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
