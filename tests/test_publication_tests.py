from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fk_transport.publication_tests import calculate_publication_diagnostics, fit_large_u_boundaries


class PublicationTestTests(unittest.TestCase):
    def test_covariance_identity_and_kelvin_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, points = root / "summary.csv", root / "points"
            omega = np.linspace(-1.0, 1.0, 2001)
            temperature = 0.08
            rows = []
            for index, branch in enumerate(("arith", "typ")):
                slope = 0.15 if branch == "arith" else -0.1
                tau = np.exp(-omega**2 / 0.15) * (1.0 + slope * omega)
                rho = np.exp(-((omega - 0.04) / 0.35) ** 2)
                point = points / f"point_{index:06d}"
                point.mkdir(parents=True)
                np.savez_compressed(point / "solution.npz", omega=omega, tau=tau,
                                    rho_arith=rho, rho_typ=0.4 * rho)
                weight = 1.0 / (4.0 * temperature * np.cosh(omega / (2.0 * temperature)) ** 2)
                l11 = np.trapz(weight * tau, omega)
                l12 = np.trapz(weight * omega * tau, omega)
                rows.append({"index": index, "branch": branch, "target_filling": 0.3,
                             "temperature": temperature, "interaction": 2.0,
                             "disorder_full_width": 1.0, "sigma": l11,
                             "thermopower": -l12 / (temperature * l11), "L12": l12,
                             "lorenz_over_L0": 1.0})
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

            diagnostics, covariance, _ = calculate_publication_diagnostics([summary], [points])
            self.assertEqual(len(diagnostics), 2)
            self.assertEqual(len(covariance), 1)
            self.assertLess(abs(float(covariance[0]["identity_absolute_error"])), 1.0e-10)
            self.assertTrue(all(np.isfinite(float(row["S_kelvin"])) for row in diagnostics))

    def test_large_u_fit_recovers_inverse_square_limit(self) -> None:
        rows, threshold = [], 1.0e-4
        disorders = np.linspace(0.5, 2.5, 81)
        for interaction in (1.5, 1.75, 2.0, 2.5, 3.0):
            critical = 1.2 + 0.45 / interaction**2
            for disorder in disorders:
                rows.append({"branch": "typ", "target_filling": 0.3, "temperature": 0.02,
                             "interaction": interaction, "disorder_full_width": disorder,
                             "rho_typ_over_arith_zero": threshold * 10.0 ** (critical - disorder)})
        boundaries, fits = fit_large_u_boundaries(rows, threshold, 1.5)
        self.assertEqual(len(boundaries), 5)
        self.assertEqual(len(fits), 1)
        self.assertTrue(np.isclose(float(fits[0]["delta_infinity"]), 1.2, atol=2.0e-3))
        self.assertTrue(np.isclose(float(fits[0]["coefficient_over_u2"]), 0.45, atol=5.0e-3))


if __name__ == "__main__":
    unittest.main()
