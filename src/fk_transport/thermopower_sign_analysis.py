from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import atomic_json
from .plotting import _publication_style, _save_publication_figure


def _number(row: dict[str, str], field: str, default: float = float("nan")) -> float:
    try:
        value = row.get(field, "")
        return float(value) if value != "" else float(default)
    except (TypeError, ValueError):
        return float(default)


def _write_rows(path: Path, rows: list[dict]) -> Path:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _sign(value: float, tolerance: float) -> int:
    if not np.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0.0 else -1


def _crossings(interactions: np.ndarray, values: np.ndarray) -> list[float]:
    found: list[float] = []
    for index in range(interactions.size):
        if not np.isfinite(values[index]):
            continue
        if values[index] == 0.0:
            found.append(float(interactions[index]))
        if index + 1 == interactions.size:
            continue
        left, right = values[index], values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right) or left * right >= 0.0:
            continue
        crossing = interactions[index] - left * (
            interactions[index + 1] - interactions[index]
        ) / (right - left)
        found.append(float(crossing))
    unique: list[float] = []
    for value in sorted(found):
        if not unique or not np.isclose(value, unique[-1], rtol=0.0, atol=1.0e-12):
            unique.append(value)
    return unique


def thermopower_sign_diagnostics(
    rows: list[dict[str, str]],
    conductivity_relative_floor: float = 1.0e-8,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[float, float, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        correlation_lambda = _number(row, "disorder_correlation_lambda", 0.0)
        if not np.isclose(correlation_lambda, 0.0, rtol=0.0, atol=1.0e-12):
            continue
        key = (
            _number(row, "target_filling"),
            _number(row, "temperature"),
            str(row.get("branch", "")),
            _number(row, "disorder_full_width"),
        )
        grouped[key].append(row)

    diagnostics: list[dict] = []
    crossing_rows: list[dict] = []
    for (filling, temperature, branch, disorder), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: _number(row, "interaction"))
        interactions = np.asarray([_number(row, "interaction") for row in ordered])
        thermopower = np.asarray([_number(row, "thermopower") for row in ordered])
        conductivity = np.asarray([_number(row, "sigma") for row in ordered])
        maximum_sigma = float(np.nanmax(conductivity))
        reliable = (
            np.isfinite(thermopower)
            & np.isfinite(conductivity)
            & (conductivity > conductivity_relative_floor * maximum_sigma)
        )
        usable = np.where(reliable)[0]
        if usable.size == 0:
            continue
        first, last = int(usable[0]), int(usable[-1])
        filtered_values = np.where(reliable, thermopower, np.nan)
        zeroes = _crossings(interactions, filtered_values)
        scale = max(float(np.nanmax(np.abs(thermopower[reliable]))), 1.0)
        sign_tolerance = 1.0e-10 * scale
        initial_sign = _sign(float(thermopower[first]), sign_tolerance)
        final_sign = _sign(float(thermopower[last]), sign_tolerance)
        if filling < 0.25 - 1.0e-12:
            prediction = "negative_at_large_U"
            prediction_met = final_sign < 0
        elif np.isclose(filling, 0.25, rtol=0.0, atol=1.0e-12):
            prediction = "approaches_zero_at_large_U"
            prediction_met = abs(float(thermopower[last])) < abs(float(thermopower[first]))
        elif filling < 0.5 - 1.0e-12:
            prediction = "forced_negative_to_positive_crossing"
            prediction_met = initial_sign < 0 and final_sign > 0 and bool(zeroes)
        else:
            prediction = "particle_hole_zero"
            prediction_met = initial_sign == 0 and final_sign == 0
        base = {
            "target_filling": filling,
            "temperature": temperature,
            "branch": branch,
            "disorder_full_width": disorder,
            "prediction": prediction,
            "prediction_met": prediction_met,
            "sampled_points": len(ordered),
            "reliable_points": int(reliable.sum()),
            "minimum_interaction": float(interactions[first]),
            "maximum_reliable_interaction": float(interactions[last]),
            "thermopower_at_minimum_interaction": float(thermopower[first]),
            "thermopower_at_maximum_reliable_interaction": float(thermopower[last]),
            "initial_sign": initial_sign,
            "large_U_sign": final_sign,
            "zero_crossings": len(zeroes),
            "first_crossing_interaction": zeroes[0] if zeroes else float("nan"),
            "conductivity_relative_floor": conductivity_relative_floor,
        }
        diagnostics.append(base)
        for crossing_index, crossing in enumerate(zeroes):
            crossing_rows.append(
                {
                    "target_filling": filling,
                    "temperature": temperature,
                    "branch": branch,
                    "disorder_full_width": disorder,
                    "crossing_index": crossing_index,
                    "interaction": crossing,
                    "interaction_minus_disorder": crossing - disorder,
                    "prediction": prediction,
                }
            )
    if not diagnostics:
        raise ValueError("no lambda=0 thermopower curves were found")
    return diagnostics, crossing_rows


def _plot_clean_curves(rows: list[dict[str, str]], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = [
        row
        for row in rows
        if np.isclose(_number(row, "disorder_full_width"), 0.0)
        and np.isclose(_number(row, "disorder_correlation_lambda", 0.0), 0.0)
    ]
    temperatures = sorted({_number(row, "temperature") for row in clean})
    fillings = sorted({_number(row, "target_filling") for row in clean})
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(fillings)))
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(
            len(temperatures), 2, figsize=(7.4, 2.15 * len(temperatures)),
            sharex=True, squeeze=False,
        )
        for row_index, temperature in enumerate(temperatures):
            for column, branch in enumerate(("arith", "typ")):
                axis = axes[row_index, column]
                for color, filling in zip(colors, fillings):
                    selected = [
                        row for row in clean
                        if row.get("branch") == branch
                        and np.isclose(_number(row, "temperature"), temperature)
                        and np.isclose(_number(row, "target_filling"), filling)
                    ]
                    selected.sort(key=lambda row: _number(row, "interaction"))
                    axis.plot(
                        [_number(row, "interaction") for row in selected],
                        [_number(row, "thermopower") for row in selected],
                        color=color,
                        linewidth=1.0,
                        marker="o",
                        markersize=1.8,
                        markevery=4,
                        label=rf"$n_c={filling:g}$",
                    )
                axis.axhline(0.0, color="black", linewidth=0.7)
                axis.set_title(rf"{branch}, $T/W={temperature:g}$")
                axis.tick_params(direction="in", top=True, right=True)
                if column == 0:
                    axis.set_ylabel(r"$S$")
                if row_index + 1 == len(temperatures):
                    axis.set_xlabel(r"$U/W$")
        axes[0, 1].legend(frameon=False, fontsize=6.5, ncol=2)
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_crossings(crossings: list[dict], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disorders = sorted({float(row["disorder_full_width"]) for row in crossings})
    temperatures = sorted({float(row["temperature"]) for row in crossings})
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.9, len(temperatures)))
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(
            2, len(disorders), figsize=(2.25 * len(disorders), 5.0),
            sharex=True, sharey=True, squeeze=False,
        )
        for row_index, branch in enumerate(("arith", "typ")):
            for column, disorder in enumerate(disorders):
                axis = axes[row_index, column]
                for color, temperature in zip(colors, temperatures):
                    selected = [
                        row for row in crossings
                        if row["branch"] == branch
                        and np.isclose(float(row["disorder_full_width"]), disorder)
                        and np.isclose(float(row["temperature"]), temperature)
                        and int(row["crossing_index"]) == 0
                    ]
                    selected.sort(key=lambda row: float(row["target_filling"]))
                    if selected:
                        axis.plot(
                            [float(row["target_filling"]) for row in selected],
                            [float(row["interaction"]) for row in selected],
                            "o-", color=color, markersize=2.5, linewidth=1.0,
                            label=rf"$T/W={temperature:g}$",
                        )
                axis.set_title(rf"{branch}, $\Delta/W={disorder:g}$")
                axis.tick_params(direction="in", top=True, right=True)
                if column == 0:
                    axis.set_ylabel(r"first $S=0$: $U_0/W$")
                if row_index == 1:
                    axis.set_xlabel(r"$n_c$")
        if disorders:
            axes[0, -1].legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def analyze_thermopower_sign(
    summaries: Iterable[str | Path],
    output_directory: str | Path,
    conductivity_relative_floor: float = 1.0e-8,
) -> list[Path]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        with Path(summary).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("summaries contain no rows")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics, crossings = thermopower_sign_diagnostics(
        rows, conductivity_relative_floor
    )
    diagnostics_path = _write_rows(output / "thermopower_sign_diagnostics.csv", diagnostics)
    crossings_path = _write_rows(output / "thermopower_zero_crossings.csv", crossings)
    forced = [
        row for row in diagnostics
        if row["prediction"] == "forced_negative_to_positive_crossing"
    ]
    report = {
        "lambda": 0.0,
        "diagnostic_curves": len(diagnostics),
        "forced_crossing_curves": len(forced),
        "forced_crossing_confirmed": sum(bool(row["prediction_met"]) for row in forced),
        "conductivity_relative_floor": conductivity_relative_floor,
        "interpretation": "The theorem requires a continuous conducting path with M0 > 0.",
    }
    report_path = output / "thermopower_sign_report.json"
    atomic_json(report_path, report)
    products = [diagnostics_path, crossings_path, report_path]
    products.append(_plot_clean_curves(rows, output / "thermopower_sign_clean.pdf"))
    if crossings:
        products.append(_plot_crossings(crossings, output / "thermopower_crossings.pdf"))
    return products
