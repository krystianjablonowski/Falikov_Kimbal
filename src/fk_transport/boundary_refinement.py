from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import canonical_config, validate_config
from .sweep import build_tasks


DEFAULT_FIELDS = (
    "sigma",
    "kappa_e",
    "charge_diffusivity_proxy",
    "thermal_diffusivity_proxy",
)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input summary contains no rows")
    return rows


def _axis(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("refinement steps must be positive")
    count = max(1, int(math.ceil((stop - start) / step)))
    values = [start + i * (stop - start) / count for i in range(count + 1)]
    return [float(f"{value:.12g}") for value in values]


def _selected_grid(
    rows: list[dict[str, str]],
    branch: str,
    temperature: float,
    filling: float,
    fields: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fields = tuple(fields)
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise ValueError(f"summary lacks requested fields: {', '.join(missing)}")
    selected = [
        row
        for row in rows
        if row["branch"] == branch
        and np.isclose(float(row["temperature"]), temperature, rtol=0.0, atol=1e-12)
        and np.isclose(float(row["target_filling"]), filling, rtol=0.0, atol=1e-12)
    ]
    if not selected:
        raise ValueError("no rows match the selected branch, temperature, and filling")
    u_values = np.array(sorted({float(row["interaction"]) for row in selected}))
    d_values = np.array(sorted({float(row["disorder_full_width"]) for row in selected}))
    if u_values.size < 2 or d_values.size < 2:
        raise ValueError("boundary refinement requires at least a 2 x 2 coarse grid")
    u_index = {value: index for index, value in enumerate(u_values)}
    d_index = {value: index for index, value in enumerate(d_values)}
    grids = {field: np.full((u_values.size, d_values.size), np.nan) for field in fields}
    for row in selected:
        i = u_index[float(row["interaction"])]
        j = d_index[float(row["disorder_full_width"])]
        for field in fields:
            grids[field][i, j] = float(row[field])
    return u_values, d_values, grids


def _flag_cells(
    grids: dict[str, np.ndarray], log_jump: float, value_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    shape = next(iter(grids.values())).shape
    flagged = np.zeros((shape[0] - 1, shape[1] - 1), dtype=bool)
    score = np.full(flagged.shape, np.nan)
    for values in grids.values():
        safe = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
        logs = np.log10(np.maximum(safe, value_floor))
        for i in range(flagged.shape[0]):
            for j in range(flagged.shape[1]):
                corners = logs[i : i + 2, j : j + 2].ravel()
                corners = corners[np.isfinite(corners)]
                if corners.size < 2:
                    continue
                cell_score = float(corners.max() - corners.min())
                score[i, j] = cell_score if not np.isfinite(score[i, j]) else max(score[i, j], cell_score)
                if cell_score >= log_jump:
                    flagged[i, j] = True
    return flagged, score


def _dilate(mask: np.ndarray, padding_cells: int) -> np.ndarray:
    if padding_cells < 0:
        raise ValueError("padding_cells must be non-negative")
    result = mask.copy()
    locations = np.argwhere(mask)
    for i, j in locations:
        result[
            max(0, i - padding_cells) : min(mask.shape[0], i + padding_cells + 1),
            max(0, j - padding_cells) : min(mask.shape[1], j + padding_cells + 1),
        ] = True
    return result


def generate_boundary_config(
    base_config: dict[str, Any],
    summary_path: str | Path,
    output_config: str | Path,
    *,
    fields: Iterable[str] = DEFAULT_FIELDS,
    branch: str = "typ",
    temperature: float | None = None,
    filling: float = 0.5,
    u_step_ratio: float = 0.025,
    disorder_step_ratio: float = 0.025,
    log_jump: float = 0.75,
    value_floor: float = 1.0e-14,
    padding_cells: int = 1,
    output_directory: str = "results/stage2_boundaries",
    batch_size: int = 8,
    max_jobs: int = 399,
) -> dict[str, Any]:
    """Create an irregular fine grid by subdividing steep coarse-grid cells.

    Step arguments use the paper convention U/W and Delta/W. Stored solver
    coordinates remain dimensional, as in every other configuration.
    """
    rows = _read_rows(summary_path)
    if batch_size < 1 or max_jobs < 1:
        raise ValueError("batch_size and max_jobs must be positive")
    available_temperatures = sorted({float(row["temperature"]) for row in rows})
    selected_temperature = available_temperatures[0] if temperature is None else float(temperature)
    u_values, d_values, grids = _selected_grid(
        rows, branch, selected_temperature, filling, fields
    )
    flagged, scores = _flag_cells(grids, log_jump, value_floor)
    selected_cells = _dilate(flagged, padding_cells)
    if not selected_cells.any():
        raise ValueError(
            "no boundary cells met the requested log-jump; lower --log-jump or choose another temperature"
        )

    bandwidth = 2.0 * float(base_config["model"]["half_bandwidth"])
    u_step = u_step_ratio * bandwidth
    d_step = disorder_step_ratio * bandwidth
    def point_key(interaction: float, disorder: float) -> tuple[float, float]:
        return round(interaction, 10), round(disorder, 10)

    coarse_points = {point_key(float(u), float(d)) for u in u_values for d in d_values}
    fine_points: set[tuple[float, float]] = set()
    for i, j in np.argwhere(selected_cells):
        for interaction in _axis(float(u_values[i]), float(u_values[i + 1]), u_step):
            for disorder in _axis(float(d_values[j]), float(d_values[j + 1]), d_step):
                point = point_key(interaction, disorder)
                if point not in coarse_points:
                    fine_points.add(point)
    if not fine_points:
        raise ValueError("refinement created no new points; reduce one of the requested steps")

    cfg = copy.deepcopy(canonical_config(base_config))
    cfg["sweep"]["parameter_points"] = [
        {"interaction": interaction, "disorder_full_width": disorder}
        for interaction, disorder in sorted(fine_points)
    ]
    cfg["output"]["directory"] = output_directory
    cfg["refinement"] = {
        "source_summary": str(Path(summary_path)),
        "branch": branch,
        "temperature": selected_temperature,
        "target_filling": filling,
        "fields": list(fields),
        "log_jump_decades": log_jump,
        "value_floor": value_floor,
        "padding_cells": padding_cells,
        "u_step_over_W": u_step_ratio,
        "disorder_step_over_W": disorder_step_ratio,
        "coarse_cells_flagged": int(flagged.sum()),
        "coarse_cells_with_padding": int(selected_cells.sum()),
        "maximum_cell_score_decades": float(np.nanmax(scores)),
    }
    validate_config(cfg)
    tasks = len(build_tasks(cfg))
    jobs = math.ceil(tasks / batch_size)
    required_batch = max(1, math.ceil(tasks / max_jobs))
    report = {
        "output_config": str(Path(output_config)),
        "parameter_points": len(fine_points),
        "spectral_tasks": tasks,
        "batch_size": batch_size,
        "pbs_jobs": jobs,
        "max_jobs": max_jobs,
        "minimum_batch_size_for_limit": required_batch,
        "within_job_limit": jobs <= max_jobs,
        "flagged_cells": int(flagged.sum()),
        "selected_cells_with_padding": int(selected_cells.sum()),
        "temperature": selected_temperature,
    }
    cfg["refinement"]["generation_report"] = report
    output = Path(output_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report
