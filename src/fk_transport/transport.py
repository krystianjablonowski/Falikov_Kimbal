from __future__ import annotations

import numpy as np

from .bethe import bethe_dos
from .grids import band_quadrature


def transport_function(
    omega: np.ndarray,
    self_energy: np.ndarray,
    chemical_potential: float,
    half_bandwidth: float,
    broadening: float,
    n_band_quadrature: int,
    block_size: int = 256,
) -> np.ndarray:
    """Jour/Freericks Bethe-lattice transport integral in natural units."""
    eps, weights = band_quadrature(half_bandwidth, n_band_quadrature)
    rho0 = bethe_dos(eps, half_bandwidth)
    velocity_weight = rho0 * (half_bandwidth**2 - eps**2) / 3.0
    tau = np.empty_like(omega, dtype=float)
    for start in range(0, omega.size, block_size):
        stop = min(start + block_size, omega.size)
        denominator = (
            omega[start:stop, None]
            + chemical_potential
            + 1j * broadening
            - self_energy[start:stop, None]
            - eps[None, :]
        )
        spectral = -(1.0 / denominator).imag / np.pi
        tau[start:stop] = np.sum(
            weights[None, :] * velocity_weight[None, :] * spectral**2, axis=1
        )
    return tau
