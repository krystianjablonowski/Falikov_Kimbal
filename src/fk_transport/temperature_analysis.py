from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .observables import combined_observables


def _positive_temperatures(values: Iterable[float]) -> list[float]:
    temperatures = sorted({float(value) for value in values})
    if not temperatures or any(value <= 0.0 for value in temperatures):
        raise ValueError("temperatures must contain positive values")
    return temperatures


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty temperature scan")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fieldnames = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def reweight_point(
    point_directory: str | Path,
    temperatures: Iterable[float],
    output_path: str | Path | None = None,
) -> Path:
    """Recompute transport and fixed-density thermodynamics from a saved spectrum.

    This is exact for the homogeneous fixed-``w1`` implementation when the saved
    spectrum is valid at every requested temperature.  In particular this is
    the inexpensive route for half filling.  Away from half filling, a saved
    point is tied to its chemical potential and should not be reweighted over a
    broad temperature range if fixed density is required.
    """
    directory = Path(point_directory)
    solution_path = directory / "solution.npz"
    metadata_path = directory / "metadata.json"
    config_path = directory / "config.resolved.json"
    if not solution_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"point directory must contain solution.npz and metadata.json: {directory}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    numerics = config.get("numerics", {})
    l11_floor = float(numerics.get("l11_floor", 1.0e-14))
    moment_tolerance = float(numerics.get("moment_tolerance", 1.0e-12))
    chemical_potential = float(metadata["chemical_potential"])

    with np.load(solution_path) as data:
        omega = np.asarray(data["omega"], dtype=float)
        tau = np.asarray(data["tau"], dtype=float)
        rho_arith = np.asarray(data["rho_arith"], dtype=float)

    task = metadata.get("task", {})
    common = {
        "point_directory": str(directory.resolve()),
        "interaction": task.get("interaction"),
        "disorder_full_width": task.get("disorder_full_width"),
        "branch": task.get("branch"),
        "target_filling": task.get("target_filling"),
        "chemical_potential": chemical_potential,
    }
    rows: list[dict] = []
    for temperature in _positive_temperatures(temperatures):
        observable = combined_observables(
            omega,
            tau,
            rho_arith,
            temperature,
            chemical_potential,
            l11_floor,
            moment_tolerance,
        )
        rows.append({**common, **observable})

    destination = (
        Path(output_path)
        if output_path is not None
        else directory / "temperature_scan.csv"
    )
    _write_csv_atomic(destination, rows)
    return destination


def _read_numeric_scan(
    path: str | Path,
    field: str,
    temperature_min: float | None,
    temperature_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[tuple[float, float]] = []
    for row in rows:
        if field not in row or row[field] in {"", "None", "nan"}:
            continue
        temperature = float(row["temperature"])
        value = float(row[field])
        if not np.isfinite(value) or value <= 0.0:
            continue
        if temperature_min is not None and temperature < temperature_min:
            continue
        if temperature_max is not None and temperature > temperature_max:
            continue
        selected.append((temperature, value))
    selected.sort()
    if not selected:
        raise ValueError(f"no positive finite values for field {field!r}")
    return (
        np.asarray([item[0] for item in selected], dtype=float),
        np.asarray([item[1] for item in selected], dtype=float),
    )


def fit_activated_law(
    temperatures: np.ndarray,
    values: np.ndarray,
    free_power: bool,
) -> dict[str, float | int | str]:
    """Fit X(T) = A T^p exp(-E/T) by linear least squares in log X."""
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    if temperatures.shape != values.shape:
        raise ValueError("temperatures and values must have the same shape")
    minimum = 3 if free_power else 2
    if temperatures.size < minimum:
        raise ValueError(
            f"the {'generalized' if free_power else 'Arrhenius'} fit requires at least {minimum} points"
        )
    if np.any(temperatures <= 0.0) or np.any(values <= 0.0):
        raise ValueError("activation fits require positive temperatures and values")

    columns = [np.ones_like(temperatures)]
    if free_power:
        columns.append(np.log(temperatures))
    columns.append(-1.0 / temperatures)
    design = np.column_stack(columns)
    target = np.log(values)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    residual = target - prediction
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    if free_power:
        log_amplitude, power, activation_energy = coefficients
        model = "A*T^p*exp(-E/T)"
    else:
        log_amplitude, activation_energy = coefficients
        power = 0.0
        model = "A*exp(-E/T)"
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return {
        "model": model,
        "points": int(temperatures.size),
        "temperature_min": float(np.min(temperatures)),
        "temperature_max": float(np.max(temperatures)),
        "amplitude": float(np.exp(log_amplitude)),
        "power": float(power),
        "activation_energy": float(activation_energy),
        "r_squared_log": r_squared,
        "rms_log_residual": float(np.sqrt(np.mean(residual**2))),
        "design_rank": int(rank),
        "design_condition_number": condition_number,
    }


def fit_temperature_scan(
    scan_path: str | Path,
    fields: Iterable[str] = ("sigma", "kappa_e"),
    temperature_min: float | None = None,
    temperature_max: float | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Fit Arrhenius and generalized activated laws to scan columns."""
    fits: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for field in fields:
        temperatures, values = _read_numeric_scan(
            scan_path, field, temperature_min, temperature_max
        )
        fits[field] = {
            "arrhenius": fit_activated_law(temperatures, values, free_power=False),
        }
        if temperatures.size >= 3:
            fits[field]["power_law_prefactor"] = fit_activated_law(
                temperatures, values, free_power=True
            )

    destination = (
        Path(output_path)
        if output_path is not None
        else Path(scan_path).with_name("activation_fits.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "source": str(Path(scan_path).resolve()),
                "temperature_min": temperature_min,
                "temperature_max": temperature_max,
                "fits": fits,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def fit_activation_groups(
    summary_path: str | Path,
    fields: Iterable[str] = ("sigma", "kappa_e"),
    group_by: Iterable[str] = (
        "branch",
        "interaction",
        "disorder_full_width",
        "target_filling",
    ),
    temperature_min: float | None = None,
    temperature_max: float | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Fit activated laws independently for every parameter group in a summary."""
    with Path(summary_path).open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("summary is empty")
    fields = list(fields)
    group_by = list(group_by)
    missing = [name for name in [*group_by, "temperature"] if name not in source_rows[0]]
    if missing:
        raise ValueError(f"summary is missing grouping columns: {missing}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in source_rows:
        groups.setdefault(tuple(row[name] for name in group_by), []).append(row)

    output_rows: list[dict] = []
    for key, group_rows in groups.items():
        common = dict(zip(group_by, key))
        for field in fields:
            selected: list[tuple[float, float]] = []
            for row in group_rows:
                raw = row.get(field, "")
                if raw in {"", "None", "nan"}:
                    continue
                temperature = float(row["temperature"])
                value = float(raw)
                if not np.isfinite(value) or value <= 0.0:
                    continue
                if temperature_min is not None and temperature < temperature_min:
                    continue
                if temperature_max is not None and temperature > temperature_max:
                    continue
                selected.append((temperature, value))
            selected.sort()
            if len(selected) < 2:
                continue
            temperatures = np.asarray([item[0] for item in selected], dtype=float)
            values = np.asarray([item[1] for item in selected], dtype=float)
            models = [("arrhenius", False)]
            if len(selected) >= 3:
                models.append(("power_law_prefactor", True))
            for fit_name, free_power in models:
                fit = fit_activated_law(temperatures, values, free_power)
                output_rows.append(
                    {
                        **common,
                        "quantity": field,
                        "fit_name": fit_name,
                        **fit,
                    }
                )

    if not output_rows:
        raise ValueError("no groups contained enough positive data for activation fits")
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(summary_path).with_name("activation_summary.csv")
    )
    _write_csv_atomic(destination, output_rows)
    return destination


def plot_temperature_scan(
    scan_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Create transport/thermodynamic temperature plots from a reweighted point."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("temperature plotting requires matplotlib") from exc

    with Path(scan_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("temperature scan is empty")
    rows.sort(key=lambda row: float(row["temperature"]))
    temperature = np.asarray([float(row["temperature"]) for row in rows])

    fields = [
        ("sigma", r"$\sigma$", True),
        ("kappa_e", r"$\kappa_e$", True),
        ("lorenz_over_L0", r"$L/L_0$", False),
        ("c_v_electronic", r"$c_V^{\rm el}$", True),
    ]
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(scan_path).with_name("temperature_scan.pdf")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for axis, (field, label, logarithmic) in zip(axes.flat, fields):
        values = np.asarray([float(row[field]) for row in rows])
        valid = np.isfinite(values) & ((values > 0.0) if logarithmic else True)
        axis.plot(temperature[valid], values[valid], "o-", markersize=3.0, linewidth=1.0)
        axis.set_xlabel(r"$T$")
        axis.set_ylabel(label)
        if logarithmic:
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.tick_params(which="both", direction="in", top=True, right=True)
        if field == "lorenz_over_L0":
            axis.axhline(1.0, color="0.5", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    destination = destination.with_suffix(".pdf")
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination
