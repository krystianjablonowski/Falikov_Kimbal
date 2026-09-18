from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .bethe import bethe_green
from .config import config_hash
from .grids import disorder_quadrature, frequency_grid
from .hilbert import hilbert_green_fft
from .impurity import local_green
from .self_energy import causality_metrics, reconstruct_self_energy


@dataclass
class SolverResult:
    omega: np.ndarray
    hybridization: np.ndarray
    green: np.ndarray
    self_energy: np.ndarray
    rho_arith: np.ndarray
    rho_typ: np.ndarray
    branch: str
    interaction: float
    disorder_full_width: float
    chemical_potential: float
    w1: float
    converged: bool
    status: str
    iterations: int
    residual_history: list[float] = field(default_factory=list)
    sum_rule_history: list[float] = field(default_factory=list)
    min_dos_history: list[float] = field(default_factory=list)
    floor_count_history: list[int] = field(default_factory=list)
    causality_history: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _residual(new: np.ndarray, old: np.ndarray) -> float:
    return float(np.max(np.abs(new - old) / np.maximum(1.0, np.abs(old))))


def _atomic_checkpoint(
    path: Path,
    iteration: int,
    hybridization: np.ndarray,
    signature: str,
    residuals: list[float],
    sum_rules: list[float],
    minima: list[float],
    floor_counts: list[int],
    causal_history: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            iteration=iteration,
            hybridization=hybridization,
            signature=signature,
            residuals=residuals,
            sum_rules=sum_rules,
            minima=minima,
            floor_counts=floor_counts,
            causal_history=causal_history,
        )
    temporary.replace(path)


def _load_checkpoint(
    path: Path | None, shape: tuple[int, ...], signature: str
) -> tuple[int, np.ndarray, list[float], list[float], list[float], list[int], list[float]] | None:
    if path is None or not path.exists():
        return None
    try:
        with np.load(path) as data:
            hybridization = np.asarray(data["hybridization"], dtype=complex)
            iteration = int(data["iteration"])
            stored_signature = str(data["signature"])
            residuals = data["residuals"].astype(float).tolist()
            sum_rules = data["sum_rules"].astype(float).tolist()
            minima = data["minima"].astype(float).tolist()
            floor_counts = data["floor_counts"].astype(int).tolist()
            causal_history = data["causal_history"].astype(float).tolist()
        if (
            stored_signature != signature
            or hybridization.shape != shape
            or not np.all(np.isfinite(hybridization))
        ):
            return None
        if not all(
            len(history) == iteration
            for history in (residuals, sum_rules, minima, floor_counts, causal_history)
        ):
            return None
        return (
            iteration,
            hybridization,
            residuals,
            sum_rules,
            minima,
            floor_counts,
            causal_history,
        )
    except (OSError, KeyError, ValueError):
        return None


def solve_medium(
    cfg: dict,
    interaction: float,
    disorder_full_width: float,
    branch: str,
    chemical_potential: float | None = None,
    initial_hybridization: np.ndarray | None = None,
    checkpoint_path: str | Path | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> SolverResult:
    if branch not in {"arith", "typ"}:
        raise ValueError("branch must be 'arith' or 'typ'")
    model, grid, num = cfg["model"], cfg["grid"], cfg["numerics"]
    d, hopping, w1 = float(model["half_bandwidth"]), float(model["hopping"]), float(model["w1"])
    if abs(d - 2.0 * hopping) > 1.0e-12 * max(1.0, d):
        raise ValueError("half_bandwidth must equal 2*hopping")
    interaction = float(interaction)
    disorder_full_width = float(disorder_full_width)
    mu = float(interaction / 2.0 if chemical_potential is None else chemical_potential)
    broadening = float(num["broadening"])
    omega = frequency_grid(float(grid["omega_max"]), int(grid["n_omega"]))
    disorder_nodes, disorder_weights = disorder_quadrature(
        disorder_full_width, int(num["n_disorder_quadrature"])
    )
    if initial_hybridization is None:
        hybridization = hopping * hopping * bethe_green(omega + 1j * broadening, d)
    else:
        hybridization = np.array(initial_hybridization, dtype=complex, copy=True)
        if hybridization.shape != omega.shape:
            raise ValueError("initial_hybridization has the wrong shape")

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_signature = (
        f"{config_hash(cfg)}|{interaction:.17g}|{disorder_full_width:.17g}|"
        f"{branch}|{mu:.17g}"
    )
    start_iteration = 0
    residuals: list[float] = []
    sum_rules: list[float] = []
    minima: list[float] = []
    floor_counts: list[int] = []
    causal_history: list[float] = []
    loaded = _load_checkpoint(checkpoint, omega.shape, checkpoint_signature)
    if loaded is not None and initial_hybridization is None:
        (
            start_iteration,
            hybridization,
            residuals,
            sum_rules,
            minima,
            floor_counts,
            causal_history,
        ) = loaded
    tolerance = float(num["tolerance"])
    mixing = float(num["mixing"])
    rho_floor = float(num["rho_floor"])
    max_iterations = int(num["max_iterations"])
    checkpoint_every = int(num.get("checkpoint_every", 0))
    converged = False

    def evaluate(hyb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        local = local_green(
            omega, hyb, disorder_nodes, mu, interaction, w1, broadening
        )
        local_rho = -local.imag / np.pi
        rho_arith_local = np.sum(disorder_weights[:, None] * local_rho, axis=0)
        positive = local_rho > rho_floor
        floor_count = int(local_rho.size - np.count_nonzero(positive))
        log_rho = np.log(np.where(positive, local_rho, rho_floor))
        rho_typ_local = np.exp(np.sum(disorder_weights[:, None] * log_rho, axis=0))
        if branch == "arith":
            green_local = np.sum(disorder_weights[:, None] * local, axis=0)
            medium_rho = rho_arith_local
        else:
            green_local = hilbert_green_fft(
                rho_typ_local, omega, int(num["hilbert_padding_factor"])
            )
            medium_rho = rho_typ_local
        return green_local, rho_arith_local, rho_typ_local, medium_rho, floor_count

    last_iteration = start_iteration
    for iteration in range(start_iteration + 1, max_iterations + 1):
        green, rho_arith, rho_typ, medium_rho, floor_count = evaluate(hybridization)
        new_hybridization = hopping * hopping * green
        residual = _residual(new_hybridization, hybridization)
        sigma = reconstruct_self_energy(omega, green, hybridization, mu, broadening)
        residuals.append(residual)
        sum_rules.append(float(np.trapezoid(medium_rho, omega)))
        minima.append(float(np.min(medium_rho)))
        floor_counts.append(floor_count)
        causal_history.append(float(np.max(sigma.imag)))
        last_iteration = iteration
        if progress is not None:
            progress(iteration, residual)
        if residual <= tolerance:
            hybridization = new_hybridization
            converged = True
            break
        hybridization = (1.0 - mixing) * hybridization + mixing * new_hybridization
        if checkpoint is not None and checkpoint_every > 0 and iteration % checkpoint_every == 0:
            _atomic_checkpoint(
                checkpoint,
                iteration,
                hybridization,
                checkpoint_signature,
                residuals,
                sum_rules,
                minima,
                floor_counts,
                causal_history,
            )

    green, rho_arith, rho_typ, medium_rho, floor_count = evaluate(hybridization)
    new_hybridization = hopping * hopping * green
    final_residual = _residual(new_hybridization, hybridization)
    sigma = reconstruct_self_energy(omega, green, hybridization, mu, broadening)
    if not residuals or final_residual != residuals[-1]:
        residuals.append(final_residual)
        sum_rules.append(float(np.trapezoid(medium_rho, omega)))
        minima.append(float(np.min(medium_rho)))
        floor_counts.append(floor_count)
        causal_history.append(float(np.max(sigma.imag)))
    spectral_dos = rho_arith if branch == "arith" else rho_typ
    metrics = causality_metrics(
        sigma, omega, list(cfg["sweep"]["temperatures"]), spectral_dos
    )
    metrics.update(
        {
            "final_residual": final_residual,
            "sum_rule_medium": float(np.trapezoid(spectral_dos, omega)),
            "sum_rule_arith_bath": float(np.trapezoid(rho_arith, omega)),
            "rho_zero": float(spectral_dos[omega.size // 2]),
            "rho_arith_zero": float(rho_arith[omega.size // 2]),
            "rho_typ_zero": float(rho_typ[omega.size // 2]),
            "rho_floor_count_final": float(floor_count),
        }
    )
    causal_tolerance = float(num["causality_tolerance"])
    if not converged:
        status = "not_converged"
    elif metrics["max_imag_sigma_relevant"] > causal_tolerance:
        status = "noncausal"
    else:
        status = "success"
    return SolverResult(
        omega=omega,
        hybridization=hybridization,
        green=green,
        self_energy=sigma,
        rho_arith=rho_arith,
        rho_typ=rho_typ,
        branch=branch,
        interaction=interaction,
        disorder_full_width=disorder_full_width,
        chemical_potential=mu,
        w1=w1,
        converged=converged,
        status=status,
        iterations=last_iteration,
        residual_history=residuals,
        sum_rule_history=sum_rules,
        min_dos_history=minima,
        floor_count_history=floor_counts,
        causality_history=causal_history,
        metrics=metrics,
    )
