from __future__ import annotations

import numpy as np


def local_green(
    omega: np.ndarray,
    hybridization: np.ndarray,
    disorder_nodes: np.ndarray,
    chemical_potential: float,
    interaction: float,
    w1: float,
    broadening: float,
) -> np.ndarray:
    """Return G_epsilon(omega), shaped (n_disorder, n_omega)."""
    bath = omega[None, :] + 1j * broadening + chemical_potential - hybridization[None, :]
    eps = disorder_nodes[:, None]
    return (1.0 - w1) / (bath - eps) + w1 / (bath - eps - interaction)
