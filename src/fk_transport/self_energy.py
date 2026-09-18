from __future__ import annotations

import numpy as np


def reconstruct_self_energy(
    omega: np.ndarray,
    green: np.ndarray,
    hybridization: np.ndarray,
    chemical_potential: float,
    broadening: float,
) -> np.ndarray:
    return omega + 1j * broadening + chemical_potential - hybridization - 1.0 / green


def causality_metrics(
    sigma: np.ndarray,
    omega: np.ndarray,
    temperatures: list[float] | tuple[float, ...],
    spectral_dos: np.ndarray,
) -> dict[str, float]:
    imag = np.asarray(sigma).imag
    global_max = float(np.max(imag))
    positive_temperatures = [float(t) for t in temperatures if float(t) > 0]
    thermal_window = 12.0 * max(positive_temperatures, default=0.0)
    support_cut = max(float(np.max(spectral_dos)) * 1.0e-10, 1.0e-14)
    mask = (np.abs(omega) <= thermal_window) | (spectral_dos > support_cut)
    relevant_max = float(np.max(imag[mask])) if np.any(mask) else global_max
    return {"max_imag_sigma_global": global_max, "max_imag_sigma_relevant": relevant_max}
