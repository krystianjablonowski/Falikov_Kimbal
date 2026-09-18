from __future__ import annotations

import csv
import gzip
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import canonical_config, config_hash
from .solver import SolverResult


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(data),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_gzip_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*columns))
    os.replace(temporary, path)


def point_directory(output_root: Path, index: int) -> Path:
    return output_root / "points" / f"point_{index:06d}"


def code_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={project_root.resolve()}", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def save_point(
    directory: Path,
    cfg: dict,
    task: dict,
    solution: SolverResult,
    tau: np.ndarray,
    observables: list[dict],
    filling_info: dict | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        directory / "solution.npz",
        omega=solution.omega,
        hybridization=solution.hybridization,
        green=solution.green,
        self_energy=solution.self_energy,
        rho_arith=solution.rho_arith,
        rho_typ=solution.rho_typ,
        tau=tau,
    )
    atomic_gzip_csv(
        directory / "spectral.csv.gz",
        [
            "omega",
            "rho_arith",
            "rho_typ",
            "green_real",
            "green_imag",
            "hybridization_real",
            "hybridization_imag",
            "sigma_real",
            "sigma_imag",
        ],
        [
            solution.omega,
            solution.rho_arith,
            solution.rho_typ,
            solution.green.real,
            solution.green.imag,
            solution.hybridization.real,
            solution.hybridization.imag,
            solution.self_energy.real,
            solution.self_energy.imag,
        ],
    )
    atomic_gzip_csv(
        directory / "transport.csv.gz", ["omega", "tau"], [solution.omega, tau]
    )
    convergence_rows = [
        {
            "iteration": i + 1,
            "residual": residual,
            "sum_rule": solution.sum_rule_history[min(i, len(solution.sum_rule_history) - 1)],
            "min_dos": solution.min_dos_history[min(i, len(solution.min_dos_history) - 1)],
            "rho_floor_count": solution.floor_count_history[min(i, len(solution.floor_count_history) - 1)],
            "max_imag_sigma": solution.causality_history[min(i, len(solution.causality_history) - 1)],
        }
        for i, residual in enumerate(solution.residual_history)
    ]
    convergence_path = directory / "convergence.csv"
    temporary = convergence_path.with_name(convergence_path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence_rows[0]))
        writer.writeheader()
        writer.writerows(convergence_rows)
    os.replace(temporary, convergence_path)
    atomic_json(directory / "observables.json", observables)
    atomic_json(directory / "config.resolved.json", canonical_config(cfg))
    metadata = {
        "status": solution.status,
        "task": task,
        "config_hash": config_hash(cfg),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "code_commit": code_commit(Path(__file__).resolve().parents[3]),
        "iterations": solution.iterations,
        "converged": solution.converged,
        "chemical_potential": solution.chemical_potential,
        "metrics": solution.metrics,
        "filling": filling_info,
    }
    # Written last: this file is the completion marker.
    atomic_json(directory / "metadata.json", metadata)
