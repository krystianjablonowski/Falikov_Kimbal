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
    return float(np.trapz(fermi_function(omega, temperature) * rho_arith_bath, omega))


def transport_observables(
    omega: np.ndarray,
    tau: np.ndarray,
    temperature: float,
    l11_floor: float = 1.0e-14,
    moment_tolerance: float = 1.0e-12,
) -> dict[str, float | bool | str]:
    weight = minus_fermi_derivative(omega, temperature)
    l11 = float(np.trapz(weight * tau, omega))
    l12 = float(np.trapz(weight * omega * tau, omega))
    l22 = float(np.trapz(weight * omega**2 * tau, omega))
    determinant = l11 * l22 - l12 * l12
    scale = max(abs(l11 * l22), abs(l12 * l12), 1.0)
    cs_ok = determinant >= -float(moment_tolerance) * scale
    ill_conditioned = l11 <= float(l11_floor)
    if ill_conditioned:
        thermopower = float("nan")
        kappa = float("nan")
        lorenz = float("nan")
        transport_mean_energy = float("nan")
        transport_energy_variance = float("nan")
        kappa_from_variance = float("nan")
        variance_identity_relative_error = float("nan")
        status = "ill_conditioned"
    elif not cs_ok:
        thermopower = float("nan")
        kappa = float("nan")
        lorenz = float("nan")
        transport_mean_energy = float("nan")
        transport_energy_variance = float("nan")
        kappa_from_variance = float("nan")
        variance_identity_relative_error = float("nan")
        status = "moment_inequality_failed"
    else:
        thermopower = -l12 / (temperature * l11)
        kappa = (l22 - l12 * l12 / l11) / temperature
        lorenz = kappa / (l11 * temperature)
        transport_mean_energy = l12 / l11
        transport_energy_variance = l22 / l11 - transport_mean_energy**2
        kappa_from_variance = l11 * transport_energy_variance / temperature
        variance_identity_relative_error = abs(kappa - kappa_from_variance) / max(
            abs(kappa), abs(kappa_from_variance), 1.0e-300
        )
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
        "kappa_e_over_T": kappa / temperature if np.isfinite(kappa) else float("nan"),
        "lorenz": lorenz,
        "lorenz_over_L0": lorenz / (np.pi**2 / 3.0),
        "transport_mean_energy": transport_mean_energy,
        "transport_energy_variance": transport_energy_variance,
        "kappa_from_variance": kappa_from_variance,
        "variance_identity_relative_error": variance_identity_relative_error,
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


def thermodynamic_observables(
    omega: np.ndarray,
    rho_arith: np.ndarray,
    temperature: float,
    chemical_potential: float,
    k0_floor: float = 1.0e-14,
    moment_tolerance: float = 1.0e-12,
) -> dict[str, float | bool | str]:
    """Electronic thermodynamics for a fixed static Falicov--Kimball medium.

    ``rho_arith`` is the physical disorder-averaged one-particle DOS, expressed
    versus omega measured from the chemical potential.  With fixed ``w1`` the
    itinerant-fermion Hamiltonian is quadratic for every static configuration,
    so its energy density is the first spectral moment

        E_c = integral (omega + mu) f(omega) rho_arith(omega) d omega.

    The heat capacity below is at fixed particle density.  It assumes that the
    absolute spectrum is temperature independent, as in the homogeneous solver
    with externally fixed ``w1``.  Temperature-driven order or an equilibrium
    temperature dependence of ``w1`` would require additional terms.
    """
    omega = np.asarray(omega, dtype=float)
    rho_arith = np.asarray(rho_arith, dtype=float)
    temperature = float(temperature)
    chemical_potential = float(chemical_potential)
    if omega.shape != rho_arith.shape:
        raise ValueError("omega and rho_arith must have the same shape")
    if np.any(rho_arith < -float(moment_tolerance)):
        raise ValueError("rho_arith must be non-negative within moment_tolerance")

    fermi = fermi_function(omega, temperature)
    weight = minus_fermi_derivative(omega, temperature)
    density = float(np.trapz(fermi * rho_arith, omega))
    energy_density = float(
        np.trapz((omega + chemical_potential) * fermi * rho_arith, omega)
    )
    k0 = float(np.trapz(weight * rho_arith, omega))
    k1 = float(np.trapz(weight * omega * rho_arith, omega))
    k2 = float(np.trapz(weight * omega**2 * rho_arith, omega))
    determinant = k0 * k2 - k1 * k1
    scale = max(abs(k0 * k2), abs(k1 * k1), 1.0)
    moment_ok = determinant >= -float(moment_tolerance) * scale
    ill_conditioned = k0 <= float(k0_floor)

    if ill_conditioned:
        mean_energy = float("nan")
        energy_variance = float("nan")
        dmu_dT_fixed_density = float("nan")
        heat_capacity = float("nan")
        status = "ill_conditioned"
    elif not moment_ok:
        mean_energy = float("nan")
        energy_variance = float("nan")
        dmu_dT_fixed_density = float("nan")
        heat_capacity = float("nan")
        status = "moment_inequality_failed"
    else:
        mean_energy = k1 / k0
        energy_variance = k2 / k0 - mean_energy**2
        dmu_dT_fixed_density = -mean_energy / temperature
        heat_capacity = k0 * energy_variance / temperature
        status = "success"

    return {
        "particle_density_thermo": density,
        "electronic_energy_density": energy_density,
        "K0_thermo": k0,
        "K1_thermo": k1,
        "K2_thermo": k2,
        "thermodynamic_mean_energy": mean_energy,
        "thermodynamic_energy_variance": energy_variance,
        "dmu_dT_fixed_density": dmu_dT_fixed_density,
        "c_v_electronic": heat_capacity,
        "thermodynamic_moment_determinant": determinant,
        "thermodynamic_moment_ok": bool(moment_ok),
        "thermodynamic_ill_conditioned": bool(ill_conditioned),
        "thermodynamic_status": status,
    }


def combined_observables(
    omega: np.ndarray,
    tau: np.ndarray,
    rho_arith: np.ndarray,
    temperature: float,
    chemical_potential: float,
    moment_floor: float = 1.0e-14,
    moment_tolerance: float = 1.0e-12,
) -> dict[str, float | bool | str]:
    """Return transport, thermodynamics, and scalar diffusion diagnostics.

    The two diffusivity ratios are physical scalar diffusivities in the
    particle-hole-symmetric case where ``L12 == 0``.  Away from that limit they
    are explicitly named proxies because charge and heat form coupled modes.
    """
    transport = transport_observables(
        omega, tau, temperature, moment_floor, moment_tolerance
    )
    thermodynamics = thermodynamic_observables(
        omega,
        rho_arith,
        temperature,
        chemical_potential,
        moment_floor,
        moment_tolerance,
    )
    sigma = float(transport["sigma"])
    kappa = float(transport["kappa_e"])
    compressibility = float(thermodynamics["K0_thermo"])
    heat_capacity = float(thermodynamics["c_v_electronic"])
    charge_diffusivity = (
        sigma / compressibility
        if compressibility > moment_floor and np.isfinite(sigma)
        else float("nan")
    )
    thermal_diffusivity = (
        kappa / heat_capacity
        if heat_capacity > moment_floor and np.isfinite(kappa)
        else float("nan")
    )
    l11, l12, l22 = (
        float(transport["L11"]),
        float(transport["L12"]),
        float(transport["L22"]),
    )
    coupling_scale = np.sqrt(max(l11 * l22, 0.0))
    thermoelectric_coupling = (
        l12 / coupling_scale if coupling_scale > moment_floor else float("nan")
    )
    return {
        **transport,
        **thermodynamics,
        "charge_diffusivity_proxy": charge_diffusivity,
        "thermal_diffusivity_proxy": thermal_diffusivity,
        "thermoelectric_coupling": thermoelectric_coupling,
        "scalar_diffusivities_decoupled": bool(
            coupling_scale > moment_floor
            and abs(l12) <= moment_tolerance * coupling_scale
        ),
    }
