from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def frequency_grid(omega_max: float, n_omega: int) -> np.ndarray:
    if n_omega % 2 != 1:
        raise ValueError("n_omega must be odd")
    middle = int(n_omega) // 2
    positive = np.arange(middle + 1, dtype=float) * (float(omega_max) / middle)
    positive[-1] = float(omega_max)
    omega = np.concatenate((-positive[:0:-1], positive))
    assert_frequency_grid(omega)
    return omega


def assert_frequency_grid(omega: np.ndarray) -> None:
    if omega.ndim != 1 or omega.size % 2 != 1:
        raise ValueError("frequency grid must be one-dimensional and odd-length")
    if omega[omega.size // 2] != 0.0:
        raise ValueError("frequency grid must contain exact zero at its center")
    if not np.array_equal(omega, -omega[::-1]):
        raise ValueError("frequency grid must be exactly symmetric")
    spacing = np.diff(omega)
    if not np.all(spacing > 0) or not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-15):
        raise ValueError("frequency grid must be increasing and uniform")


def disorder_quadrature(disorder_full_width: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    width = float(disorder_full_width)
    if width < 0:
        raise ValueError("disorder_full_width must be non-negative")
    if width == 0:
        return np.array([0.0]), np.array([1.0])
    nodes, weights = leggauss(int(order))
    return 0.5 * width * nodes, 0.5 * weights


def band_quadrature(half_bandwidth: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(int(order))
    d = float(half_bandwidth)
    return d * nodes, d * weights
