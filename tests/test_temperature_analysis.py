from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.temperature_analysis import (
    fit_activated_law,
    fit_activation_groups,
    reweight_point,
)


class TemperatureAnalysisTests(unittest.TestCase):
    def test_generalized_activation_fit_recovers_parameters(self):
        temperature = np.asarray([0.01, 0.012, 0.016, 0.022, 0.03, 0.04])
        amplitude, power, energy = 2.5, 1.25, 0.08
        values = amplitude * temperature**power * np.exp(-energy / temperature)
        fit = fit_activated_law(temperature, values, free_power=True)
        self.assertAlmostEqual(float(fit["amplitude"]), amplitude, places=10)
        self.assertAlmostEqual(float(fit["power"]), power, places=10)
        self.assertAlmostEqual(float(fit["activation_energy"]), energy, places=10)
        self.assertAlmostEqual(float(fit["r_squared_log"]), 1.0, places=12)

    def test_reweight_saved_point_writes_transport_and_heat_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            point = Path(temporary)
            omega = np.linspace(-2.0, 2.0, 20001)
            np.savez_compressed(
                point / "solution.npz",
                omega=omega,
                tau=np.ones_like(omega),
                rho_arith=np.full_like(omega, 0.25),
            )
            (point / "metadata.json").write_text(
                json.dumps(
                    {
                        "chemical_potential": 0.0,
                        "task": {
                            "interaction": 0.0,
                            "disorder_full_width": 0.0,
                            "branch": "arith",
                            "target_filling": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = reweight_point(point, [0.01, 0.02])
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("transport_energy_variance", rows[0])
            self.assertIn("c_v_electronic", rows[0])
            self.assertIn("thermal_diffusivity_proxy", rows[0])
            self.assertGreater(float(rows[0]["c_v_electronic"]), 0.0)

    def test_grouped_activation_fits_do_not_mix_parameter_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "branch",
                    "interaction",
                    "disorder_full_width",
                    "target_filling",
                    "temperature",
                    "sigma",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for disorder, energy in ((0.5, 0.04), (1.5, 0.12)):
                    for temperature in (0.01, 0.015, 0.02, 0.03):
                        writer.writerow(
                            {
                                "branch": "typ",
                                "interaction": 0.3,
                                "disorder_full_width": disorder,
                                "target_filling": 0.5,
                                "temperature": temperature,
                                "sigma": 2.0 * np.exp(-energy / temperature),
                            }
                        )
            output = fit_activation_groups(summary, fields=["sigma"])
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            arrhenius = [row for row in rows if row["fit_name"] == "arrhenius"]
            self.assertEqual(len(arrhenius), 2)
            energies = sorted(float(row["activation_energy"]) for row in arrhenius)
            self.assertAlmostEqual(energies[0], 0.04, places=10)
            self.assertAlmostEqual(energies[1], 0.12, places=10)


if __name__ == "__main__":
    unittest.main()
