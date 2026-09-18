from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observables import particle_density
from .solver import SolverResult, solve_medium


@dataclass
class FillingResult:
    solution: SolverResult
    target_filling: float
    obtained_filling: float
    filling_error: float
    evaluations: int


def solve_for_filling(
    cfg: dict,
    interaction: float,
    disorder_full_width: float,
    branch: str,
    target_filling: float,
    temperature: float,
) -> FillingResult:
    """Bracketed bisection in mu; every evaluation is a converged medium solve."""
    settings = cfg["filling"]
    lower, upper = float(settings["mu_min"]), float(settings["mu_max"])
    tolerance, max_iterations = float(settings["tolerance"]), int(settings["max_iterations"])
    cache: dict[float, tuple[float, SolverResult]] = {}

    def evaluate(mu: float) -> tuple[float, SolverResult]:
        key = float(mu)
        if key in cache:
            return cache[key]
        initial = None
        if cache:
            nearest = min(cache, key=lambda old_mu: abs(old_mu - key))
            if cache[nearest][1].converged:
                initial = cache[nearest][1].hybridization
        solution = solve_medium(
            cfg,
            interaction,
            disorder_full_width,
            branch,
            chemical_potential=key,
            initial_hybridization=initial,
        )
        if solution.status != "success":
            raise RuntimeError(f"medium solve at mu={key:.16g} has status {solution.status}")
        density = particle_density(solution.omega, solution.rho_arith, temperature)
        cache[key] = (density, solution)
        return cache[key]

    n_lower, sol_lower = evaluate(lower)
    n_upper, sol_upper = evaluate(upper)
    f_lower, f_upper = n_lower - target_filling, n_upper - target_filling
    if f_lower == 0:
        return FillingResult(sol_lower, target_filling, n_lower, f_lower, len(cache))
    if f_upper == 0:
        return FillingResult(sol_upper, target_filling, n_upper, f_upper, len(cache))
    if f_lower * f_upper > 0:
        raise ValueError(
            f"filling target {target_filling} is not bracketed: n({lower})={n_lower}, n({upper})={n_upper}"
        )
    best_solution, best_density = sol_lower, n_lower
    for _ in range(max_iterations):
        middle = 0.5 * (lower + upper)
        n_middle, solution = evaluate(middle)
        error = n_middle - target_filling
        if abs(error) < abs(best_density - target_filling):
            best_solution, best_density = solution, n_middle
        if abs(error) <= tolerance:
            return FillingResult(solution, target_filling, n_middle, error, len(cache))
        if f_lower * error <= 0:
            upper, f_upper = middle, error
        else:
            lower, f_lower = middle, error
    return FillingResult(
        best_solution,
        target_filling,
        best_density,
        best_density - target_filling,
        len(cache),
    )
