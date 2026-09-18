from __future__ import annotations

import numpy as np


def bethe_dos(energy: np.ndarray, half_bandwidth: float) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    d = float(half_bandwidth)
    inside = np.maximum(d * d - energy * energy, 0.0)
    rho = 2.0 * np.sqrt(inside) / (np.pi * d * d)
    rho[np.abs(energy) > d] = 0.0
    return rho


def bethe_green(z: np.ndarray | complex, half_bandwidth: float) -> np.ndarray:
    """Retarded Hilbert transform of the normalized semicircular DOS."""
    z = np.asarray(z, dtype=complex)
    d = float(half_bandwidth)
    root = np.sqrt(z - d) * np.sqrt(z + d)
    return 2.0 * (z - root) / (d * d)
