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
    disorder_correlation_lambda: float = 0.0,
) -> np.ndarray:
    """Return G_epsilon(omega), shaped (n_disorder, n_omega).

    The correlated-disorder Hamiltonian uses local potentials

        V_0 = (1-lambda) epsilon,
        V_1 = U + (1+lambda) epsilon,

    for an empty or occupied immobile-fermion site, respectively.  Lambda=0
    therefore reproduces the original uncorrelated-disorder implementation.
    """
    bath = omega[None, :] + 1j * broadening + chemical_potential - hybridization[None, :]
    eps = disorder_nodes[:, None]
    lam = float(disorder_correlation_lambda)
    return ((1.0 - w1) / (bath - (1.0 - lam) * eps)
            + w1 / (bath - interaction - (1.0 + lam) * eps))
