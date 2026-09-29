import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.grids import frequency_grid
from fk_transport.observables import (
    combined_observables,
    minus_fermi_derivative,
    thermodynamic_observables,
    transport_observables,
)


class ObservableTests(unittest.TestCase):
    def test_fermi_derivative_is_stable_and_normalized(self):
        omega = frequency_grid(2.0, 20001)
        derivative = minus_fermi_derivative(omega, 0.02)
        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertAlmostEqual(float(np.trapz(derivative, omega)), 1.0, places=8)

    def test_constant_tau_has_wiedemann_franz_limit(self):
        omega = frequency_grid(2.0, 20001)
        result = transport_observables(omega, np.ones_like(omega), 0.02)
        self.assertAlmostEqual(float(result["L12"]), 0.0, places=12)
        self.assertAlmostEqual(float(result["lorenz"]), np.pi**2 / 3.0, places=8)
        self.assertAlmostEqual(
            float(result["kappa_e"]), float(result["kappa_from_variance"]), places=14
        )
        self.assertLess(float(result["variance_identity_relative_error"]), 1.0e-13)
        self.assertTrue(result["cauchy_schwarz_ok"])

    def test_fixed_density_heat_capacity_for_flat_symmetric_dos(self):
        omega = frequency_grid(2.0, 40001)
        rho = np.full_like(omega, 0.25)
        temperature = 0.02
        result = thermodynamic_observables(omega, rho, temperature, chemical_potential=0.0)
        sommerfeld = (np.pi**2 / 3.0) * 0.25 * temperature
        self.assertAlmostEqual(float(result["particle_density_thermo"]), 0.5, places=10)
        self.assertAlmostEqual(float(result["K1_thermo"]), 0.0, places=12)
        self.assertAlmostEqual(float(result["dmu_dT_fixed_density"]), 0.0, places=11)
        self.assertAlmostEqual(float(result["c_v_electronic"]), sommerfeld, places=8)
        self.assertEqual(result["thermodynamic_status"], "success")

    def test_half_filled_scalar_diffusivity_proxies(self):
        omega = frequency_grid(2.0, 20001)
        result = combined_observables(
            omega,
            np.ones_like(omega),
            np.full_like(omega, 0.25),
            temperature=0.02,
            chemical_potential=0.0,
        )
        self.assertTrue(result["scalar_diffusivities_decoupled"])
        self.assertAlmostEqual(float(result["thermoelectric_coupling"]), 0.0, places=12)
        self.assertGreater(float(result["charge_diffusivity_proxy"]), 0.0)
        self.assertGreater(float(result["thermal_diffusivity_proxy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
