from __future__ import annotations

from typing import Any

import numpy as np

from .bethe import bethe_dos, bethe_green
from .grids import band_quadrature, frequency_grid
from .hilbert import hilbert_green_direct, hilbert_green_fft
from .observables import transport_observables
from .solver import solve_medium
from .transport import transport_function


def run_validation(cfg: dict) -> dict[str, Any]:
    model = cfg["model"]
    d = float(model["half_bandwidth"])
    omega = frequency_grid(float(cfg["grid"]["omega_max"]), int(cfg["grid"]["n_omega"]))
    eps, weights = band_quadrature(d, max(512, int(cfg["numerics"]["n_band_quadrature"])))
    dos_integral = float(np.sum(weights * bethe_dos(eps, d)))

    broadening = float(cfg["numerics"]["broadening"])
    reference_rho = -bethe_green(omega + 1j * broadening, d).imag / np.pi
    fast = hilbert_green_fft(reference_rho, omega, int(cfg["numerics"]["hilbert_padding_factor"]))
    selected_indices = np.linspace(omega.size // 4, 3 * omega.size // 4, 9, dtype=int)
    direct = hilbert_green_direct(reference_rho, omega, omega[selected_indices], max(broadening, 2 * (omega[1] - omega[0])))
    hilbert_relative_error = float(
        np.max(np.abs(fast[selected_indices].real - direct.real))
        / max(float(np.max(np.abs(direct.real))), 1.0e-15)
    )

    clean = solve_medium(cfg, 0.0, 0.0, "arith", chemical_potential=0.0)
    clean_reference = bethe_green(omega + 1j * broadening, d)
    central = np.abs(omega) <= 0.8 * d
    clean_green_error = float(
        np.max(np.abs(clean.green[central] - clean_reference[central]))
        / max(float(np.max(np.abs(clean_reference[central]))), 1.0e-15)
    )
    clean_typ = solve_medium(cfg, 0.0, 0.0, "typ", chemical_potential=0.0)
    branch_error = float(
        np.max(np.abs(clean.rho_arith[central] - clean_typ.rho_typ[central]))
        / max(float(np.max(clean.rho_arith[central])), 1.0e-15)
    )
    tau = transport_function(
        clean.omega,
        clean.self_energy,
        clean.chemical_potential,
        d,
        broadening,
        int(cfg["numerics"]["n_band_quadrature"]),
    )
    observable = transport_observables(
        clean.omega,
        tau,
        float(cfg["sweep"]["temperatures"][0]),
        float(cfg["numerics"]["l11_floor"]),
        float(cfg["numerics"]["moment_tolerance"]),
    )
    checks = {
        "grid_odd": omega.size % 2 == 1,
        "grid_exact_zero": bool(omega[omega.size // 2] == 0.0),
        "grid_symmetric": bool(np.array_equal(omega, -omega[::-1])),
        "bethe_dos_normalized": abs(dos_integral - 1.0) < 1.0e-8,
        "clean_arith_converged": clean.status == "success",
        "clean_green_matches_semicircle": clean_green_error < 3.0e-2,
        "clean_typ_converged": clean_typ.status == "success",
        "zero_disorder_branches_agree": branch_error < 8.0e-2,
        "tau_nonnegative": bool(np.min(tau) >= -1.0e-14),
        "half_filling_L12_small": abs(float(observable["L12"])) < 1.0e-10,
        "cauchy_schwarz": bool(observable["cauchy_schwarz_ok"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "bethe_dos_integral": dos_integral,
            "hilbert_real_relative_error": hilbert_relative_error,
            "clean_green_relative_error": clean_green_error,
            "zero_disorder_branch_relative_error": branch_error,
            "clean_arith_status": clean.status,
            "clean_typ_status": clean_typ.status,
            "L12": observable["L12"],
            "lorenz": observable["lorenz"],
            "lorenz_reference": float(np.pi**2 / 3.0),
        },
        "warnings": [
            "FFT Hilbert error is diagnostic because the direct check uses finite broadening.",
            "Production scans require separate convergence studies in n_omega, omega_max, broadening, and quadratures.",
        ],
    }
