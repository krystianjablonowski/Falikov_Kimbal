from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import config_hash
from .filling import solve_for_filling
from .io import atomic_json, point_directory, save_point
from .observables import particle_density, transport_observables
from .solver import solve_medium
from .transport import transport_function


def build_tasks(cfg: dict) -> list[dict[str, Any]]:
    sweep = cfg["sweep"]
    tasks: list[dict[str, Any]] = []
    index = 0
    for interaction in sweep["interactions"]:
        for disorder in sweep["disorder_full_widths"]:
            for branch in sweep["branches"]:
                fillings = [float(x) for x in sweep["target_fillings"]]
                if fillings == [0.5] and float(cfg["model"]["w1"]) == 0.5:
                    tasks.append(
                        {
                            "index": index,
                            "interaction": float(interaction),
                            "disorder_full_width": float(disorder),
                            "branch": branch,
                            "target_filling": 0.5,
                            "temperatures": [float(x) for x in sweep["temperatures"]],
                            "half_filling": True,
                        }
                    )
                    index += 1
                else:
                    for filling in fillings:
                        for temperature in sweep["temperatures"]:
                            tasks.append(
                                {
                                    "index": index,
                                    "interaction": float(interaction),
                                    "disorder_full_width": float(disorder),
                                    "branch": branch,
                                    "target_filling": float(filling),
                                    "temperatures": [float(temperature)],
                                    "half_filling": False,
                                }
                            )
                            index += 1
    return tasks


def output_root(cfg: dict) -> Path:
    configured = Path(cfg["output"]["directory"])
    config_path = Path(cfg.get("_config_path", ".")).resolve()
    return configured if configured.is_absolute() else config_path.parent.parent / configured


def write_manifest(cfg: dict) -> Path:
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    atomic_json(path, {"config_hash": config_hash(cfg), "tasks": build_tasks(cfg)})
    return path


def _run_task_unlocked(cfg: dict, index: int) -> dict:
    tasks = build_tasks(cfg)
    if index < 0 or index >= len(tasks):
        raise IndexError(f"task index {index} outside [0, {len(tasks) - 1}]")
    task = tasks[index]
    directory = point_directory(output_root(cfg), index)
    checkpoint = directory / "checkpoint.npz"
    interaction = task["interaction"]
    disorder = task["disorder_full_width"]
    filling_info = None
    if task["half_filling"]:
        solution = solve_medium(
            cfg,
            interaction,
            disorder,
            task["branch"],
            chemical_potential=interaction / 2.0,
            checkpoint_path=checkpoint,
        )
        filling_info = {
            "target": 0.5,
            "obtained_by_temperature": {
                str(t): particle_density(solution.omega, solution.rho_arith, t)
                for t in task["temperatures"]
            },
        }
    else:
        temperature = task["temperatures"][0]
        found = solve_for_filling(
            cfg,
            interaction,
            disorder,
            task["branch"],
            task["target_filling"],
            temperature,
        )
        solution = found.solution
        filling_info = {
            "target": found.target_filling,
            "obtained": found.obtained_filling,
            "error": found.filling_error,
            "solver_evaluations": found.evaluations,
        }
    tau = transport_function(
        solution.omega,
        solution.self_energy,
        solution.chemical_potential,
        float(cfg["model"]["half_bandwidth"]),
        float(cfg["numerics"]["broadening"]),
        int(cfg["numerics"]["n_band_quadrature"]),
    )
    observables = [
        transport_observables(
            solution.omega,
            tau,
            temperature,
            float(cfg["numerics"]["l11_floor"]),
            float(cfg["numerics"]["moment_tolerance"]),
        )
        for temperature in task["temperatures"]
    ]
    save_point(directory, cfg, task, solution, tau, observables, filling_info)
    return {"directory": str(directory), "status": solution.status, "task": task}


def run_task(cfg: dict, index: int) -> dict:
    """Run exactly one deterministic task while preventing concurrent writers."""
    directory = point_directory(output_root(cfg), index)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") == config_hash(cfg) and existing.get("status") == "success":
                return {"directory": str(directory), "status": "already_success", "task": existing["task"]}
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    lock_path = directory / "task.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"task {index} is already locked by another process: {lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        return _run_task_unlocked(cfg, index)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def scan_status(cfg: dict) -> dict[str, Any]:
    statuses: dict[str, list[int]] = {
        "success": [],
        "not_converged": [],
        "noncausal": [],
        "corrupt": [],
        "missing": [],
        "config_mismatch": [],
    }
    expected_hash = config_hash(cfg)
    for task in build_tasks(cfg):
        index = task["index"]
        metadata_path = point_directory(output_root(cfg), index) / "metadata.json"
        if not metadata_path.exists():
            statuses["missing"].append(index)
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config_hash") != expected_hash:
                statuses["config_mismatch"].append(index)
            elif metadata.get("status") in statuses:
                statuses[metadata["status"]].append(index)
            else:
                statuses["corrupt"].append(index)
        except (OSError, json.JSONDecodeError, KeyError):
            statuses["corrupt"].append(index)
    rerun = sorted(
        statuses["not_converged"]
        + statuses["noncausal"]
        + statuses["corrupt"]
        + statuses["missing"]
        + statuses["config_mismatch"]
    )
    report = {"counts": {k: len(v) for k, v in statuses.items()}, "indices": statuses, "rerun": rerun}
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "status.json", report)
    temporary = root / "rerun_indices.txt.tmp"
    temporary.write_text(",".join(str(x) for x in rerun) + "\n", encoding="utf-8")
    os.replace(temporary, root / "rerun_indices.txt")
    return report


def merge_results(cfg: dict) -> Path:
    report = scan_status(cfg)
    root = output_root(cfg)
    rows: list[dict[str, Any]] = []
    expected_hash = config_hash(cfg)
    for index in report["indices"]["success"]:
        directory = point_directory(root, index)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["config_hash"] != expected_hash:
            continue
        observables = json.loads((directory / "observables.json").read_text(encoding="utf-8"))
        for observable in observables:
            row = {
                **metadata["task"],
                "chemical_potential": metadata["chemical_potential"],
                "iterations": metadata["iterations"],
                **metadata["metrics"],
                **observable,
            }
            row["temperatures"] = json.dumps(row["temperatures"])
            rows.append(row)
    path = root / "summary.csv"
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["index", "status"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path
