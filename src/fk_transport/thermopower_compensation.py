from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .observables import minus_fermi_derivative
from .plotting import _grid, _publication_style, _save_publication_figure


def _number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary contains no rows")
    required = {
        "branch", "target_filling", "temperature", "interaction",
        "disorder_full_width", "L12",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"summary is missing columns: {', '.join(sorted(missing))}")
    return rows


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _row_crossings(x: np.ndarray, values: np.ndarray) -> list[float]:
    """Return linearly interpolated zero crossings, without spanning missing cells."""
    found: list[float] = []
    for index in range(x.size):
        value = values[index]
        if np.isfinite(value) and value == 0.0:
            found.append(float(x[index]))
        if index + 1 == x.size:
            continue
        left, right = values[index], values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right) or left * right >= 0.0:
            continue
        crossing = x[index] - left * (x[index + 1] - x[index]) / (right - left)
        found.append(float(crossing))
    unique: list[float] = []
    for value in sorted(found):
        if not unique or not np.isclose(value, unique[-1], rtol=0.0, atol=1.0e-12):
            unique.append(value)
    return unique


def extract_zero_crossings(
    rows: list[dict[str, str]], bandwidth: float = 1.0
) -> list[dict[str, float | int | str]]:
    """Extract L12=0 curves at fixed filling, temperature, branch, and U."""
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    fillings = sorted({_number(row, "target_filling") for row in rows})
    temperatures = sorted({_number(row, "temperature") for row in rows})
    output: list[dict[str, float | int | str]] = []
    for filling in fillings:
        for temperature in temperatures:
            for branch in ("arith", "typ"):
                disorder, interaction, values = _grid(
                    rows, branch, temperature, filling, "L12", positive_only=False
                )
                for row_index, interaction_value in enumerate(interaction):
                    crossings = _row_crossings(disorder, values[row_index])
                    for crossing_index, crossing in enumerate(crossings):
                        output.append(
                            {
                                "target_filling": filling,
                                "temperature": temperature,
                                "temperature_over_W": temperature / bandwidth,
                                "branch": branch,
                                "interaction": float(interaction_value),
                                "interaction_over_W": float(interaction_value) / bandwidth,
                                "crossing_index": crossing_index,
                                "disorder_full_width": crossing,
                                "disorder_over_W": crossing / bandwidth,
                            }
                        )
    return output


def match_branch_crossings(
    crossings: list[dict[str, float | int | str]],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Match the nearest arithmetic and typical zero at every sampled U."""
    grouped: dict[tuple[float, float, float], dict[str, list[dict]]] = defaultdict(
        lambda: {"arith": [], "typ": []}
    )
    for row in crossings:
        key = (
            float(row["target_filling"]),
            float(row["temperature"]),
            float(row["interaction"]),
        )
        grouped[key][str(row["branch"])].append(row)

    matched: list[dict[str, float | int | str]] = []
    for (filling, temperature, interaction), branches in sorted(grouped.items()):
        candidates = [
            (abs(float(a["disorder_full_width"]) - float(t["disorder_full_width"])), a, t)
            for a in branches["arith"]
            for t in branches["typ"]
        ]
        if not candidates:
            continue
        distance, arithmetic, typical = min(candidates, key=lambda item: item[0])
        matched.append(
            {
                "target_filling": filling,
                "temperature": temperature,
                "temperature_over_W": arithmetic["temperature_over_W"],
                "interaction": interaction,
                "interaction_over_W": arithmetic["interaction_over_W"],
                "disorder_arith": arithmetic["disorder_full_width"],
                "disorder_typ": typical["disorder_full_width"],
                "disorder_over_W_arith": arithmetic["disorder_over_W"],
                "disorder_over_W_typ": typical["disorder_over_W"],
                "delta_disorder": float(typical["disorder_full_width"])
                - float(arithmetic["disorder_full_width"]),
                "abs_delta_disorder": distance,
                "abs_delta_disorder_over_W": abs(
                    float(typical["disorder_over_W"])
                    - float(arithmetic["disorder_over_W"])
                ),
            }
        )

    summaries: list[dict[str, float | int | str]] = []
    by_temperature: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in matched:
        by_temperature[(float(row["target_filling"]), float(row["temperature"]))].append(row)
    for (filling, temperature), selected in sorted(by_temperature.items()):
        distances = np.asarray(
            [float(row["abs_delta_disorder_over_W"]) for row in selected], dtype=float
        )
        signed = np.asarray(
            [float(row["delta_disorder"]) for row in selected], dtype=float
        )
        summaries.append(
            {
                "target_filling": filling,
                "temperature": temperature,
                "temperature_over_W": selected[0]["temperature_over_W"],
                "matched_interactions": distances.size,
                "mean_abs_separation_over_W": float(np.mean(distances)),
                "rms_separation_over_W": float(np.sqrt(np.mean(distances**2))),
                "median_abs_separation_over_W": float(np.median(distances)),
                "maximum_abs_separation_over_W": float(np.max(distances)),
                "mean_signed_separation": float(np.mean(signed)),
            }
        )
    return matched, summaries


def _plot_contours(
    crossings: list[dict[str, float | int | str]], filling: float, output: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temperatures = sorted(
        {float(row["temperature_over_W"]) for row in crossings if float(row["target_filling"]) == filling}
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, max(len(temperatures), 1)))
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True, sharey=True)
        for axis, branch in zip(axes, ("arith", "typ")):
            for color, temperature in zip(colors, temperatures):
                selected = [
                    row for row in crossings
                    if row["branch"] == branch
                    and float(row["target_filling"]) == filling
                    and np.isclose(float(row["temperature_over_W"]), temperature)
                    and int(row["crossing_index"]) == 0
                ]
                selected.sort(key=lambda row: float(row["interaction_over_W"]))
                if selected:
                    axis.plot(
                        [float(row["disorder_over_W"]) for row in selected],
                        [float(row["interaction_over_W"]) for row in selected],
                        color=color,
                        marker="o",
                        markersize=2.0,
                        linewidth=1.2,
                        label=rf"$T/W={temperature:g}$",
                    )
            axis.set_title(rf"{branch}, $n_c={filling:g}$")
            axis.set_xlabel(r"disorder $\Delta/W$")
            axis.tick_params(which="both", direction="in", top=True, right=True)
        axes[0].set_ylabel(r"interaction $U/W$")
        axes[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_branch_comparison(
    crossings: list[dict[str, float | int | str]], filling: float, output: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temperatures = sorted(
        {float(row["temperature_over_W"]) for row in crossings if float(row["target_filling"]) == filling}
    )
    columns = 2
    rows_count = max(1, int(np.ceil(len(temperatures) / columns)))
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(
            rows_count, columns, figsize=(7.4, 2.8 * rows_count), squeeze=False,
            sharex=True, sharey=True,
        )
        for panel, (axis, temperature) in enumerate(zip(axes.flat, temperatures)):
            for branch, style in (("arith", "-"), ("typ", "--")):
                selected = [
                    row for row in crossings
                    if row["branch"] == branch
                    and float(row["target_filling"]) == filling
                    and np.isclose(float(row["temperature_over_W"]), temperature)
                    and int(row["crossing_index"]) == 0
                ]
                selected.sort(key=lambda row: float(row["interaction_over_W"]))
                axis.plot(
                    [float(row["disorder_over_W"]) for row in selected],
                    [float(row["interaction_over_W"]) for row in selected],
                    style,
                    linewidth=1.4,
                    label=branch,
                )
            axis.set_title(rf"$T/W={temperature:g}$")
            axis.tick_params(which="both", direction="in", top=True, right=True)
            axis.text(0.03, 0.93, f"({chr(97 + panel)})", transform=axis.transAxes,
                      fontweight="bold", va="top")
        for axis in axes.flat[len(temperatures):]:
            axis.set_visible(False)
        for axis in axes[-1, :]:
            if axis.get_visible():
                axis.set_xlabel(r"disorder $\Delta/W$")
        for axis in axes[:, 0]:
            axis.set_ylabel(r"interaction $U/W$")
        if temperatures:
            axes.flat[0].legend(frameon=False)
        fig.suptitle(rf"$L_{{12}}=0$, $n_c={filling:g}$")
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_separation(
    summaries: list[dict[str, float | int | str]], filling: float, output: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in summaries if float(row["target_filling"]) == filling]
    selected.sort(key=lambda row: float(row["temperature_over_W"]))
    with plt.rc_context(_publication_style()):
        fig, axis = plt.subplots(figsize=(4.2, 3.1))
        temperature = [float(row["temperature_over_W"]) for row in selected]
        axis.plot(
            temperature,
            [float(row["rms_separation_over_W"]) for row in selected],
            "o-", label="RMS",
        )
        axis.plot(
            temperature,
            [float(row["mean_abs_separation_over_W"]) for row in selected],
            "s--", label="mean absolute",
        )
        axis.set_xlabel(r"temperature $T/W$")
        axis.set_ylabel(r"arith--typ contour separation $/W$")
        axis.set_title(rf"$L_{{12}}=0$, $n_c={filling:g}$")
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.legend(frameon=False)
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _transport_distribution(
    omega: np.ndarray, tau: np.ndarray, temperature: float
) -> tuple[np.ndarray, float, float, float]:
    weight = minus_fermi_derivative(omega, temperature)
    density = weight * np.maximum(tau, 0.0)
    normalization = float(np.trapz(density, omega))
    if normalization <= 0.0:
        raise ValueError("transport distribution has non-positive normalization")
    probability = density / normalization
    negative = omega < 0.0
    positive = omega > 0.0
    l12_negative = float(np.trapz((omega * density)[negative], omega[negative]))
    l12_positive = float(np.trapz((omega * density)[positive], omega[positive]))
    return probability, normalization, l12_negative, l12_positive


def spectral_diagnostics(
    summary_rows: list[dict[str, str]],
    matched: list[dict[str, float | int | str]],
    points_root: str | Path,
) -> list[dict[str, float | int | str]]:
    """Evaluate cancellation and arith/typ shape distance near each matched contour."""
    points = Path(points_root)
    lookup: dict[tuple[str, float, float, float], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        key = (
            row["branch"],
            _number(row, "target_filling"),
            _number(row, "temperature"),
            _number(row, "interaction"),
        )
        lookup[key].append(row)

    diagnostics: list[dict[str, float | int | str]] = []
    for contour in matched:
        filling = float(contour["target_filling"])
        temperature = float(contour["temperature"])
        interaction = float(contour["interaction"])
        midpoint = 0.5 * (
            float(contour["disorder_arith"]) + float(contour["disorder_typ"])
        )
        selected: dict[str, dict[str, str]] = {}
        distributions: dict[str, tuple[np.ndarray, np.ndarray, float, float, float]] = {}
        for branch in ("arith", "typ"):
            candidates = lookup.get((branch, filling, temperature, interaction), [])
            if not candidates:
                break
            row = min(candidates, key=lambda item: abs(_number(item, "disorder_full_width") - midpoint))
            index = int(float(row["index"]))
            solution_path = points / f"point_{index:06d}" / "solution.npz"
            if not solution_path.is_file():
                break
            with np.load(solution_path) as data:
                omega = np.asarray(data["omega"], dtype=float)
                tau = np.asarray(data["tau"], dtype=float)
            probability, l11, negative, positive = _transport_distribution(
                omega, tau, temperature
            )
            selected[branch] = row
            distributions[branch] = (omega, probability, l11, negative, positive)
        if len(distributions) != 2:
            continue
        omega_a, probability_a, l11_a, negative_a, positive_a = distributions["arith"]
        omega_t, probability_t, l11_t, negative_t, positive_t = distributions["typ"]
        if not np.array_equal(omega_a, omega_t):
            probability_t = np.interp(omega_a, omega_t, probability_t, left=0.0, right=0.0)
        shape_distance = 0.5 * float(np.trapz(np.abs(probability_a - probability_t), omega_a))
        diagnostics.append(
            {
                **contour,
                "sampled_disorder_arith": _number(selected["arith"], "disorder_full_width"),
                "sampled_disorder_typ": _number(selected["typ"], "disorder_full_width"),
                "point_index_arith": int(float(selected["arith"]["index"])),
                "point_index_typ": int(float(selected["typ"]["index"])),
                "L11_arith": l11_a,
                "L11_typ": l11_t,
                "L12_negative_arith": negative_a,
                "L12_positive_arith": positive_a,
                "L12_balance_arith": negative_a + positive_a,
                "L12_negative_typ": negative_t,
                "L12_positive_typ": positive_t,
                "L12_balance_typ": negative_t + positive_t,
                "transport_distribution_L1_distance": shape_distance,
            }
        )
    return diagnostics


def _plot_spectral_summary(
    diagnostics: list[dict[str, float | int | str]], filling: float, output: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in diagnostics:
        if float(row["target_filling"]) == filling:
            grouped[float(row["temperature_over_W"])].append(
                float(row["transport_distribution_L1_distance"])
            )
    temperatures = sorted(grouped)
    means = [float(np.mean(grouped[temperature])) for temperature in temperatures]
    p90 = [float(np.percentile(grouped[temperature], 90.0)) for temperature in temperatures]
    with plt.rc_context(_publication_style()):
        fig, axis = plt.subplots(figsize=(4.2, 3.1))
        axis.plot(temperatures, means, "o-", label="mean")
        axis.plot(temperatures, p90, "s--", label="90th percentile")
        axis.set_xlabel(r"temperature $T/W$")
        axis.set_ylabel(r"$\frac{1}{2}\int |P_{\rm arith}-P_{\rm typ}|\,d\omega$")
        axis.set_title(rf"transport-shape distance near $L_{{12}}=0$, $n_c={filling:g}$")
        axis.set_ylim(bottom=0.0)
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.legend(frameon=False)
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def analyze_thermopower_compensation(
    summary_path: str | Path,
    output_directory: str | Path,
    bandwidth: float = 1.0,
    points_root: str | Path | None = None,
) -> list[Path]:
    rows = _read_rows(summary_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    crossings = extract_zero_crossings(rows, bandwidth)
    matched, summaries = match_branch_crossings(crossings)
    outputs = [
        _write_rows(output / "thermopower_zero_crossings.csv", crossings),
        _write_rows(output / "thermopower_zero_crossing_pairs.csv", matched),
        _write_rows(output / "thermopower_zero_crossing_separation.csv", summaries),
    ]
    fillings = sorted({_number(row, "target_filling") for row in rows})
    for filling in fillings:
        tag = f"{filling:.6g}".replace(".", "p")
        outputs.extend(
            [
                _plot_contours(
                    crossings, filling, output / f"compensation_contours_n_{tag}.png"
                ),
                _plot_branch_comparison(
                    crossings, filling, output / f"compensation_branch_comparison_n_{tag}.png"
                ),
                _plot_separation(
                    summaries, filling, output / f"compensation_separation_n_{tag}.png"
                ),
            ]
        )
    if points_root is not None:
        diagnostics = spectral_diagnostics(rows, matched, points_root)
        outputs.append(
            _write_rows(output / "thermopower_compensation_spectral_diagnostics.csv", diagnostics)
        )
        if diagnostics:
            for filling in fillings:
                tag = f"{filling:.6g}".replace(".", "p")
                outputs.append(
                    _plot_spectral_summary(
                        diagnostics,
                        filling,
                        output / f"transport_shape_distance_n_{tag}.png",
                    )
                )
    return outputs
