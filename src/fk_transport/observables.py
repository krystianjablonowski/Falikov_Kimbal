from __future__ import annotations

import numpy as np


def minus_fermi_derivative(omega: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.abs(np.asarray(omega, dtype=float) / temperature)
    decay = np.exp(-scaled)
    return decay / (temperature * (1.0 + decay) ** 2)


def fermi_function(omega: np.ndarray, temperature: float) -> np.ndarray:
    x = np.asarray(omega, dtype=float) / float(temperature)
    result = np.empty_like(x)
    positive = x >= 0
    ex = np.exp(-np.abs(x))
    result[positive] = ex[positive] / (1.0 + ex[positive])
    result[~positive] = 1.0 / (1.0 + ex[~positive])
    return result


def particle_density(omega: np.ndarray, rho_arith_bath: np.ndarray, temperature: float) -> float:
    return float(np.trapezoid(fermi_function(omega, temperature) * rho_arith_bath, omega))


def transport_observables(
    omega: np.ndarray,
    tau: np.ndarray,
    temperature: float,
    l11_floor: float = 1.0e-14,
    moment_tolerance: float = 1.0e-12,
) -> dict[str, float | bool | str]:
    weight = minus_fermi_derivative(omega, temperature)
    l11 = float(np.trapezoid(weight * tau, omega))
    l12 = float(np.trapezoid(weight * omega * tau, omega))
    l22 = float(np.trapezoid(weight * omega**2 * tau, omega))
    determinant = l11 * l22 - l12 * l12
    scale = max(abs(l11 * l22), abs(l12 * l12), 1.0)
    cs_ok = determinant >= -float(moment_tolerance) * scale
    ill_conditioned = l11 <= float(l11_floor)
    if ill_conditioned:
        thermopower = float("nan")
        kappa = float("nan")
        lorenz = float("nan")
        status = "ill_conditioned"
    elif not cs_ok:
        thermopower = float("nan")
        kappa = float("nan")
        lorenz = float("nan")
        status = "moment_inequality_failed"
    else:
        thermopower = -l12 / (temperature * l11)
        kappa = (l22 - l12 * l12 / l11) / temperature
        lorenz = kappa / (l11 * temperature)
        status = "success"
    reflected = np.interp(-omega, omega, tau)
    symmetry_error = float(np.max(np.abs(tau - reflected)) / max(float(np.max(np.abs(tau))), 1.0e-300))
    kappa_half = l22 / temperature
    return {
        "temperature": float(temperature),
        "L11": l11,
        "L12": l12,
        "L22": l22,
        "sigma": l11,
        "thermopower": thermopower,
        "kappa_e": kappa,
        "lorenz": lorenz,
        "lorenz_over_L0": lorenz / (np.pi**2 / 3.0),
        "cauchy_schwarz_determinant": determinant,
        "cauchy_schwarz_ok": bool(cs_ok),
        "ill_conditioned": bool(ill_conditioned),
        "status": status,
        "tau_symmetry_relative_error": symmetry_error,
        "kappa_half_filling_formula": kappa_half,
        "kappa_half_relative_difference": abs(kappa - kappa_half) / max(abs(kappa_half), 1.0e-300)
        if np.isfinite(kappa)
        else float("nan"),
    }
