from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np

from .grids import disorder_quadrature
from .io import atomic_json


def _number(row: dict[str, str], field: str, default: float | None = None) -> float:
    value = row.get(field)
    if value in (None, ""):
        if default is None:
            raise ValueError(f"missing numeric field {field}")
        return float(default)
    return float(value)


def linearized_typical_log_multiplier(
    interaction: float,
    disorder_full_width: float,
    correlation_lambda: float,
    chemical_potential: float,
    hybridization_zero: complex,
    bandwidth: float = 1.0,
    w1: float = 0.5,
    quadrature_order: int = 256,
) -> float:
    """Linearized TMT multiplier at the Fermi level.

    Positive log multiplier means that an infinitesimal typical DOS grows under
    one Bethe/TMT iteration; negative means that it decays.  The two local
    potentials are (1-lambda)*epsilon and U+(1+lambda)*epsilon.
    """
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if not 0.0 <= w1 <= 1.0:
        raise ValueError("w1 must lie in [0, 1]")
    nodes, weights = disorder_quadrature(disorder_full_width, quadrature_order)
    r_value = float(chemical_potential) - float(np.real(hybridization_zero))
    lower = r_value - (1.0 - correlation_lambda) * nodes
    upper = r_value - interaction - (1.0 + correlation_lambda) * nodes
    denominator_floor = 1.0e-28 * max(float(bandwidth) ** 2, 1.0)
    factor = (
        (1.0 - w1) / np.maximum(lower * lower, denominator_floor)
        + w1 / np.maximum(upper * upper, denominator_floor)
    )
    hopping_squared = (float(bandwidth) / 4.0) ** 2
    return float(np.log(hopping_squared) + np.sum(weights * np.log(factor)))


def _write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({field for row in rows for field in row}) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _connected_components(
    records: list[dict],
    interactions: list[float] | None = None,
    disorders: list[float] | None = None,
) -> tuple[int, int, bool]:
    if interactions is None:
        interactions = sorted({float(row["interaction"]) for row in records})
    if disorders is None:
        disorders = sorted({float(row["disorder_full_width"]) for row in records})
    u_index = {value: index for index, value in enumerate(interactions)}
    d_index = {value: index for index, value in enumerate(disorders)}
    occupied = {
        (u_index[float(row["interaction"])], d_index[float(row["disorder_full_width"])])
        for row in records
        if bool(row["metal"])
    }
    components: list[int] = []
    while occupied:
        start = occupied.pop()
        queue = deque([start])
        size = 0
        while queue:
            i, j = queue.popleft()
            size += 1
            for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbor in occupied:
                    occupied.remove(neighbor)
                    queue.append(neighbor)
        components.append(size)
    complete = len(records) == len(interactions) * len(disorders)
    return len(components), max(components, default=0), complete


def phase_fraction_rows(records: list[dict]) -> list[dict]:
    interactions = sorted({float(row["interaction"]) for row in records})
    disorders = sorted({float(row["disorder_full_width"]) for row in records})
    expected = len(interactions) * len(disorders)
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in records:
        grouped[float(row["disorder_correlation_lambda"])].append(row)
    output: list[dict] = []
    for correlation_lambda, group in sorted(grouped.items()):
        metal = sum(bool(row["metal"]) for row in group)
        components, largest, complete = _connected_components(
            group, interactions, disorders
        )
        output.append(
            {
                "disorder_correlation_lambda": correlation_lambda,
                "points": len(group),
                "expected_grid_points": expected,
                "missing_points": expected - len(group),
                "metal_points": metal,
                "localized_points": len(group) - metal,
                "metal_fraction": metal / expected,
                "metal_components": components,
                "largest_metal_component_points": largest,
                "largest_metal_component_fraction": largest / expected,
                "regular_grid_complete": complete and len(group) == expected,
            }
        )
    return output


def _zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    crossings: list[float] = []
    for left in range(len(x) - 1):
        x0, x1, y0, y1 = x[left], x[left + 1], y[left], y[left + 1]
        if y0 == 0.0:
            crossings.append(float(x0))
        if y0 * y1 < 0.0:
            crossings.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    if len(y) and y[-1] == 0.0:
        crossings.append(float(x[-1]))
    return crossings


def reentrant_rows(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in records:
        grouped[
            (float(row["interaction"]), float(row["disorder_full_width"]))
        ].append(row)
    output: list[dict] = []
    for (interaction, disorder), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["disorder_correlation_lambda"]))
        lambdas = np.asarray(
            [float(row["disorder_correlation_lambda"]) for row in ordered]
        )
        values = np.asarray([float(row["log_lambda_typ"]) for row in ordered])
        signs = ["M" if value > 0.0 else "I" for value in values]
        compressed = [signs[0]] if signs else []
        for sign in signs[1:]:
            if sign != compressed[-1]:
                compressed.append(sign)
        crossings = _zero_crossings(lambdas, values)
        reentrant = (
            len(compressed) >= 3
            and compressed[0] == "I"
            and compressed[-1] == "I"
            and "M" in compressed[1:-1]
        )
        output.append(
            {
                "interaction": interaction,
                "disorder_full_width": disorder,
                "lambda_min": float(lambdas.min()),
                "lambda_max": float(lambdas.max()),
                "sampled_lambdas": len(lambdas),
                "phase_sequence": "-".join(compressed),
                "zero_crossings": len(crossings),
                "crossing_lambdas": ";".join(f"{value:.12g}" for value in crossings),
                "reentrant_AI_metal_AI": reentrant,
                "minimum_log_lambda_typ": float(values.min()),
                "maximum_log_lambda_typ": float(values.max()),
                "log_lambda_typ_range": float(np.ptp(values)),
            }
        )
    return output


def _load_records(
    summaries: Iterable[str | Path],
    points_roots: Iterable[str | Path],
    filling: float,
    temperature: float,
    bandwidth: float,
    w1: float,
    quadrature_order: int,
) -> list[dict]:
    summaries = [Path(path) for path in summaries]
    points_roots = [Path(path) for path in points_roots]
    if len(summaries) != len(points_roots):
        raise ValueError("summaries and points_roots must have the same length")
    records: dict[tuple[float, float, float], dict] = {}
    for dataset, (summary, points_root) in enumerate(zip(summaries, points_roots)):
        with summary.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("branch") != "typ":
                continue
            if not np.isclose(_number(row, "target_filling"), filling, atol=1e-12):
                continue
            if not np.isclose(_number(row, "temperature"), temperature, atol=1e-12):
                continue
            correlation_lambda = _number(row, "disorder_correlation_lambda", 0.0)
            interaction = _number(row, "interaction")
            disorder = _number(row, "disorder_full_width")
            point = points_root / f"point_{int(float(row['index'])):06d}" / "solution.npz"
            if not point.is_file():
                raise FileNotFoundError(f"missing saved solution: {point}")
            with np.load(point) as data:
                omega = np.asarray(data["omega"], dtype=float)
                hybridization = np.asarray(data["hybridization"], dtype=complex)
            zero = int(np.argmin(np.abs(omega)))
            log_multiplier = linearized_typical_log_multiplier(
                interaction,
                disorder,
                correlation_lambda,
                _number(row, "chemical_potential"),
                hybridization[zero],
                bandwidth,
                w1,
                quadrature_order,
            )
            key = (round(correlation_lambda, 12), round(interaction, 12), round(disorder, 12))
            records[key] = {
                "dataset": dataset,
                "index": int(float(row["index"])),
                "disorder_correlation_lambda": correlation_lambda,
                "interaction": interaction,
                "interaction_over_W": interaction / bandwidth,
                "disorder_full_width": disorder,
                "disorder_over_W": disorder / bandwidth,
                "target_filling": filling,
                "temperature": temperature,
                "chemical_potential": _number(row, "chemical_potential"),
                "R": _number(row, "chemical_potential") - float(hybridization[zero].real),
                "log_lambda_typ": log_multiplier,
                "lambda_typ": float(np.exp(np.clip(log_multiplier, -700.0, 700.0))),
                "metal": log_multiplier > 0.0,
                "rho_typ_zero": _number(row, "rho_typ_zero", float("nan")),
                "rho_arith_zero": _number(row, "rho_arith_zero", float("nan")),
                "sigma": _number(row, "sigma", float("nan")),
                "lower_disorder_scale_over_Delta": abs(1.0 - correlation_lambda),
                "upper_disorder_scale_over_Delta": abs(1.0 + correlation_lambda),
            }
    if not records:
        raise ValueError("no matching typical-medium rows were found")
    return list(records.values())


def _plot_fraction(rows: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["disorder_correlation_lambda"] for row in rows], dtype=float)
    fraction = np.asarray([row["metal_fraction"] for row in rows], dtype=float)
    largest = np.asarray([row["largest_metal_component_fraction"] for row in rows], dtype=float)
    components = np.asarray([row["metal_components"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(x, fraction, "o-", label=r"$f_{\rm metal}$")
    axes[0].plot(x, largest, "s--", label="largest connected component")
    axes[0].set(xlabel=r"correlation $\lambda$", ylabel="fraction of parameter grid", ylim=(-0.03, 1.03))
    axes[0].legend(frameon=False)
    axes[1].plot(x, components, "o-", color="#8c2d04")
    axes[1].set(xlabel=r"correlation $\lambda$", ylabel="number of metallic components")
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_maps(records: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    lambdas = sorted({float(row["disorder_correlation_lambda"]) for row in records})
    columns = min(3, len(lambdas))
    rows_count = int(np.ceil(len(lambdas) / columns))
    finite = np.asarray([abs(float(row["log_lambda_typ"])) for row in records])
    limit = max(float(np.percentile(finite, 98.0)), 1.0e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(rows_count, columns, figsize=(3.1 * columns, 2.7 * rows_count), squeeze=False)
    image = None
    for axis, correlation_lambda in zip(axes.flat, lambdas):
        chosen = [row for row in records if np.isclose(row["disorder_correlation_lambda"], correlation_lambda)]
        u_values = np.asarray(sorted({row["interaction_over_W"] for row in chosen}))
        d_values = np.asarray(sorted({row["disorder_over_W"] for row in chosen}))
        grid = np.full((len(u_values), len(d_values)), np.nan)
        ui = {value: index for index, value in enumerate(u_values)}
        di = {value: index for index, value in enumerate(d_values)}
        for row in chosen:
            grid[ui[row["interaction_over_W"]], di[row["disorder_over_W"]]] = row["log_lambda_typ"]
        image = axis.pcolormesh(d_values, u_values, np.ma.masked_invalid(grid), shading="nearest", cmap="RdBu_r", norm=norm)
        if len(d_values) > 1 and len(u_values) > 1 and np.nanmin(grid) <= 0.0 <= np.nanmax(grid):
            axis.contour(d_values, u_values, grid, levels=[0.0], colors="black", linewidths=0.8)
        axis.set_title(rf"$\lambda={correlation_lambda:g}$")
        axis.set_xlabel(r"$\Delta/W$")
        axis.set_ylabel(r"$U/W$")
        axis.tick_params(direction="in", top=True, right=True)
    for axis in axes.flat[len(lambdas):]:
        axis.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat[:len(lambdas)]), label=r"$\log\Lambda_{\rm typ}$", shrink=0.9)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.94, wspace=0.30, hspace=0.35)
    fig.savefig(output)
    plt.close(fig)
    return output


def _plot_profiles(records: list[dict], crossings: list[dict], output: Path, maximum: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(
        crossings,
        key=lambda row: (
            bool(row["reentrant_AI_metal_AI"]),
            int(row["zero_crossings"]),
            float(row["log_lambda_typ_range"]),
        ),
        reverse=True,
    )[:maximum]
    lookup: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in records:
        lookup[(float(row["interaction"]), float(row["disorder_full_width"]))].append(row)
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    for candidate in ranked:
        key = (float(candidate["interaction"]), float(candidate["disorder_full_width"]))
        curve = sorted(lookup[key], key=lambda row: float(row["disorder_correlation_lambda"]))
        axis.plot(
            [row["disorder_correlation_lambda"] for row in curve],
            [row["log_lambda_typ"] for row in curve],
            "o-",
            label=rf"$U/W={key[0]:g},\ \Delta/W={key[1]:g}$",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel=r"correlation $\lambda$", ylabel=r"$\log\Lambda_{\rm typ}$")
    axis.tick_params(direction="in", top=True, right=True)
    axis.grid(alpha=0.18)
    if ranked:
        axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def analyze_lambda_phase(
    summaries: Iterable[str | Path],
    points_roots: Iterable[str | Path],
    output_directory: str | Path,
    filling: float = 0.4,
    temperature: float = 0.05,
    bandwidth: float = 1.0,
    w1: float = 0.5,
    quadrature_order: int = 256,
    maximum_profiles: int = 6,
) -> list[Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records = _load_records(
        summaries, points_roots, filling, temperature, bandwidth, w1, quadrature_order
    )
    fractions = phase_fraction_rows(records)
    crossings = reentrant_rows(records)
    point_path = _write_rows(output / "lambda_phase_points.csv", records)
    fraction_path = _write_rows(output / "lambda_phase_fraction.csv", fractions)
    crossing_path = _write_rows(output / "lambda_reentrant_crossings.csv", crossings)
    maximum = max(fractions, key=lambda row: float(row["metal_fraction"]))
    report = {
        "target_filling": filling,
        "temperature": temperature,
        "sampled_lambdas": [row["disorder_correlation_lambda"] for row in fractions],
        "maximum_metal_fraction_lambda": maximum["disorder_correlation_lambda"],
        "maximum_metal_fraction": maximum["metal_fraction"],
        "reentrant_AI_metal_AI_points": sum(
            bool(row["reentrant_AI_metal_AI"]) for row in crossings
        ),
        "linearized_criterion": "metal iff log_lambda_typ > 0",
    }
    report_path = output / "lambda_phase_report.json"
    atomic_json(report_path, report)
    products = [point_path, fraction_path, crossing_path, report_path]
    try:
        products.extend(
            [
                _plot_fraction(fractions, output / "lambda_metal_fraction.pdf"),
                _plot_maps(records, output / "lambda_phase_maps.pdf"),
                _plot_profiles(
                    records,
                    crossings,
                    output / "lambda_reentrant_profiles.pdf",
                    maximum_profiles,
                ),
            ]
        )
    except ImportError as exc:
        raise RuntimeError("lambda phase plots require matplotlib") from exc
    return products
