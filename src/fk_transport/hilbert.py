from __future__ import annotations

import numpy as np

from .grids import assert_frequency_grid


def _next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def hilbert_green_fft(rho: np.ndarray, omega: np.ndarray, padding_factor: int = 8) -> np.ndarray:
    """Causal boundary-value Green function using a zero-padded FFT Hilbert transform."""
    rho = np.asarray(rho, dtype=float)
    omega = np.asarray(omega, dtype=float)
    assert_frequency_grid(omega)
    if rho.shape != omega.shape:
        raise ValueError("rho and omega must have equal shapes")
    nfft = _next_power_of_two(max(omega.size * int(padding_factor), omega.size + 2))
    padded = np.zeros(nfft, dtype=float)
    start = (nfft - omega.size) // 2
    padded[start : start + omega.size] = rho
    spectrum = np.fft.fft(padded)
    multiplier = np.zeros(nfft)
    multiplier[0] = 1.0
    if nfft % 2 == 0:
        multiplier[1 : nfft // 2] = 2.0
        multiplier[nfft // 2] = 1.0
    else:
        multiplier[1 : (nfft + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier)
    hilbert = analytic.imag[start : start + omega.size]
    return np.pi * hilbert - 1j * np.pi * rho


def hilbert_green_direct(
    rho: np.ndarray,
    omega: np.ndarray,
    evaluation_omega: np.ndarray,
    broadening: float,
) -> np.ndarray:
    """Independent trapezoidal Cauchy transform for selected frequencies."""
    rho = np.asarray(rho, dtype=float)
    omega = np.asarray(omega, dtype=float)
    evaluation_omega = np.atleast_1d(evaluation_omega).astype(float)
    if rho.shape != omega.shape:
        raise ValueError("rho and omega must have equal shapes")
    kernel = 1.0 / (
        evaluation_omega[:, None] + 1j * float(broadening) - omega[None, :]
    )
    return np.trapz(kernel * rho[None, :], omega, axis=1)
