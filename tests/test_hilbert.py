import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.bethe import bethe_dos, bethe_green
from fk_transport.grids import frequency_grid
from fk_transport.hilbert import hilbert_green_direct, hilbert_green_fft


class HilbertTests(unittest.TestCase):
    def test_fft_transform_of_semicircle(self):
        omega = frequency_grid(4.0, 4001)
        rho = bethe_dos(omega, 0.5)
        transformed = hilbert_green_fft(rho, omega, 16)
        expected = bethe_green(omega + 1e-8j, 0.5)
        central = np.abs(omega) < 0.4
        relative = np.max(np.abs(transformed[central].real - expected[central].real)) / np.max(
            np.abs(expected[central].real)
        )
        self.assertLess(relative, 2e-2)
        self.assertTrue(np.allclose(transformed.imag, -np.pi * rho))

    def test_direct_transform_is_retarded(self):
        omega = frequency_grid(3.0, 2001)
        rho = bethe_dos(omega, 0.5)
        result = hilbert_green_direct(rho, omega, np.array([-0.2, 0.0, 0.2]), 0.003)
        self.assertTrue(np.all(result.imag < 0))


if __name__ == "__main__":
    unittest.main()
