from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .observables import minus_fermi_derivative
from .plotting import _publication_style, _save_publication_figure
from .thermopower_compensation import _number, _read_rows, extract_zero_crossings


def _integral(values: np.ndarray, omega: np.ndarray) -> float:
    return float(np.trapz(values, omega))


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _tag(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")


def _load_arrays(
    points_root: Path, index: int, branch: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = points_root / f"point_{index:06d}" / "solution.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    spectral_field = "rho_arith" if branch == "arith" else "rho_typ"
    with np.load(path) as data:
        required = {"omega", "tau", spectral_field}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} does not contain {', '.join(sorted(missing))}")
        omega = np.asarray(data["omega"], dtype=float)
        tau = np.maximum(np.asarray(data["tau"], dtype=float), 0.0)
        rho = np.maximum(np.asarray(data[spectral_field], dtype=float), 0.0)
    return omega, tau, rho


def _moments_from_arrays(
    omega: np.ndarray, tau: np.ndarray, rho: np.ndarray, temperature: float
) -> dict[str, float]:
    weight = minus_fermi_derivative(omega, temperature)
    transport_weight = _integral(weight * tau, omega)
    transport_moment = _integral(weight * omega * tau, omega)
    spectral_weight = _integral(weight * rho, omega)
    spectral_moment = _integral(weight * omega * rho, omega)
    zero_index = int(np.argmin(np.abs(omega)))
    return {
        "rho_zero": float(rho[zero_index]),
        "spectral_thermal_weight": spectral_weight,
        "spectral_first_moment": spectral_moment,
        "spectral_centroid": spectral_moment / spectral_weight if spectral_weight > 0.0 else float("nan"),
        "transport_thermal_weight": transport_weight,
        "transport_first_moment": transport_moment,
        "transport_centroid": transport_moment / transport_weight if transport_weight > 0.0 else float("nan"),
    }


def calculate_mechanism_diagnostics(
    summary_paths: list[str | Path],
    points_roots: list[str | Path],
    temperatures: list[float] | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, str]]]:
    if len(summary_paths) != len(points_roots):
        raise ValueError("--summaries and --points-roots must contain the same number of paths")
    diagnostics: list[dict[str, float | int | str]] = []
    all_summary_rows: list[dict[str, str]] = []
    for dataset_index, (summary_path, points_root_value) in enumerate(
        zip(summary_paths, points_roots)
    ):
        rows = _read_rows(summary_path)
        if temperatures:
            rows = [
                row for row in rows
                if any(np.isclose(_number(row, "temperature"), value) for value in temperatures)
            ]
        if not rows:
            continue
        all_summary_rows.extend(rows)
        points_root = Path(points_root_value)
        by_point: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_point[(int(_number(row, "index")), row["branch"])].append(row)
        for (index, branch), point_rows in by_point.items():
            omega, tau, rho = _load_arrays(points_root, index, branch)
            for row in point_rows:
                temperature = _number(row, "temperature")
                moments = _moments_from_arrays(omega, tau, rho, temperature)
                sigma = _number(row, "sigma")
                thermopower = _number(row, "thermopower")
                diagnostics.append(
                    {
                        "dataset_index": dataset_index,
                        "summary_path": str(summary_path),
                        "points_root": str(points_root),
                        "index": index,
                        "branch": branch,
                        "target_filling": _number(row, "target_filling"),
                        "temperature": temperature,
                        "interaction": _number(row, "interaction"),
                        "disorder_full_width": _number(row, "disorder_full_width"),
                        "sigma": sigma,
                        "thermopower": thermopower,
                        "L12": _number(row, "L12"),
                        "power_factor": thermopower**2 * sigma,
                        **moments,
                        "spectral_centroid_over_T": moments["spectral_centroid"] / temperature,
                        "transport_centroid_over_T": moments["transport_centroid"] / temperature,
                    }
                )
    lookup: dict[tuple[float, float, float, float], dict[str, dict]] = defaultdict(dict)
    for row in diagnostics:
        key = tuple(
            round(float(row[field]), 12)
            for field in (
                "target_filling", "temperature", "interaction", "disorder_full_width"
            )
        )
        lookup[key][str(row["branch"])] = row
    for branches in lookup.values():
        if set(branches) != {"arith", "typ"}:
            continue
        arith, typ = branches["arith"], branches["typ"]
        sigma_a = float(arith["sigma"])
        rho_a = float(arith["rho_zero"])
        sigma_ratio = float(typ["sigma"]) / sigma_a if sigma_a > 0.0 else float("nan")
        rho_ratio = float(typ["rho_zero"]) / rho_a if rho_a > 0.0 else float("nan")
        for row in (arith, typ):
            row["sigma_typ_over_arith"] = sigma_ratio
            row["rho_zero_typ_over_arith"] = rho_ratio
    return diagnostics, all_summary_rows


def _plot_centroid_correlations(rows: list[dict], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    fillings = sorted({float(row["target_filling"]) for row in rows})
    for filling in fillings:
        selected = [
            row for row in rows
            if np.isclose(float(row["target_filling"]), filling)
            and np.isfinite(float(row["spectral_centroid_over_T"]))
            and np.isfinite(float(row["transport_centroid_over_T"]))
        ]
        with plt.rc_context(_publication_style()):
            fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True, sharey=True)
            for axis, branch in zip(axes, ("arith", "typ")):
                branch_rows = [row for row in selected if row["branch"] == branch]
                x = np.asarray([float(row["spectral_centroid_over_T"]) for row in branch_rows])
                y = np.asarray([float(row["transport_centroid_over_T"]) for row in branch_rows])
                color = np.asarray([float(row["disorder_full_width"]) for row in branch_rows])
                scatter = axis.scatter(x, y, c=color, s=9, cmap="viridis", alpha=0.75)
                if x.size:
                    limit = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0e-6)
                    axis.plot([-limit, limit], [-limit, limit], color="0.4", linestyle=":", linewidth=0.8)
                    same_sign = float(np.mean(np.signbit(x) == np.signbit(y)))
                    correlation = (
                        float(np.corrcoef(x, y)[0, 1])
                        if x.size > 1 and np.std(x) > 0.0 and np.std(y) > 0.0
                        else float("nan")
                    )
                    axis.set_title(rf"{branch}: $r={correlation:.2f}$, sign={same_sign:.2f}")
                axis.axhline(0.0, color="0.7", linewidth=0.6)
                axis.axvline(0.0, color="0.7", linewidth=0.6)
                axis.set_xlabel(r"spectral centroid $\bar\omega_\rho/T$")
                axis.tick_params(which="both", direction="in", top=True, right=True)
            axes[0].set_ylabel(r"transport centroid $\bar\omega_\tau/T=-S$")
            if selected:
                fig.colorbar(scatter, ax=axes, label=r"disorder $\Delta/W$")
            fig.suptitle(rf"spectral--transport asymmetry, $n_c={filling:g}$")
            path = output / f"centroid_correlation_n_{_tag(filling)}.pdf"
            _save_publication_figure(fig, path)
            plt.close(fig)
            outputs.append(path)
    return outputs


def _plot_contours_by_filling(
    summary_rows: list[dict[str, str]], output: Path, bandwidth: float
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crossings = extract_zero_crossings(summary_rows, bandwidth)
    temperatures = sorted({float(row["temperature_over_W"]) for row in crossings})
    fillings = sorted({_number(row, "target_filling") for row in summary_rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, max(len(fillings), 1)))
    outputs: list[Path] = []
    for temperature in temperatures:
        with plt.rc_context(_publication_style()):
            fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True, sharey=True)
            for axis, branch in zip(axes, ("arith", "typ")):
                for filling, color in zip(fillings, colors):
                    selected = [
                        row for row in crossings
                        if row["branch"] == branch
                        and np.isclose(float(row["target_filling"]), filling)
                        and np.isclose(float(row["temperature_over_W"]), temperature)
                        and int(row["crossing_index"]) == 0
                    ]
                    selected.sort(key=lambda row: float(row["interaction_over_W"]))
                    if selected:
                        axis.plot(
                            [float(row["disorder_over_W"]) for row in selected],
                            [float(row["interaction_over_W"]) for row in selected],
                            "o-", color=color, markersize=2.5, linewidth=1.1,
                            label=rf"$n_c={filling:g}$",
                        )
                axis.set_title(branch)
                axis.set_xlabel(r"disorder $\Delta/W$")
                axis.tick_params(which="both", direction="in", top=True, right=True)
            axes[0].set_ylabel(r"interaction $U/W$")
            axes[1].legend(frameon=False)
            fig.suptitle(rf"thermoelectric compensation contours, $T/W={temperature:g}$")
            path = output / f"compensation_contours_fillings_T_{_tag(temperature)}.pdf"
            _save_publication_figure(fig, path)
            plt.close(fig)
            outputs.append(path)
    return outputs


def _nearest_available(values: list[float], requested: list[float]) -> list[float]:
    return sorted({min(values, key=lambda value: abs(value - target)) for target in requested})


def _plot_linecuts(
    rows: list[dict], output: Path, interactions: list[float], bandwidth: float
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    fillings = sorted({float(row["target_filling"]) for row in rows})
    for filling in fillings:
        filling_rows = [row for row in rows if np.isclose(float(row["target_filling"]), filling)]
        temperatures = sorted({float(row["temperature"]) for row in filling_rows})
        available_u = sorted({float(row["interaction"]) for row in filling_rows})
        for temperature in temperatures:
            for interaction in _nearest_available(available_u, interactions):
                selected = [
                    row for row in filling_rows
                    if np.isclose(float(row["temperature"]), temperature)
                    and np.isclose(float(row["interaction"]), interaction)
                ]
                if not selected:
                    continue
                with plt.rc_context(_publication_style()):
                    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7), sharex=True)
                    for branch, color in (("arith", "#2477b3"), ("typ", "#d94b2b")):
                        branch_rows = sorted(
                            (row for row in selected if row["branch"] == branch),
                            key=lambda row: float(row["disorder_full_width"]),
                        )
                        x = np.asarray([float(row["disorder_full_width"]) / bandwidth for row in branch_rows])
                        axes[0, 0].plot(x, [float(row["L12"]) for row in branch_rows], "o-", color=color, markersize=2, label=branch)
                        axes[0, 1].plot(x, [float(row["thermopower"]) for row in branch_rows], "o-", color=color, markersize=2)
                        axes[1, 0].semilogy(x, np.maximum([float(row["sigma"]) for row in branch_rows], 1.0e-300), "o-", color=color, markersize=2)
                    paired = sorted(
                        (row for row in selected if row["branch"] == "typ"),
                        key=lambda row: float(row["disorder_full_width"]),
                    )
                    x_pair = [float(row["disorder_full_width"]) / bandwidth for row in paired]
                    axes[1, 1].semilogy(
                        x_pair,
                        np.maximum([float(row.get("sigma_typ_over_arith", np.nan)) for row in paired], 1.0e-300),
                        "o-", label=r"$\sigma_{typ}/\sigma_{arith}$",
                    )
                    axes[1, 1].semilogy(
                        x_pair,
                        np.maximum([float(row.get("rho_zero_typ_over_arith", np.nan)) for row in paired], 1.0e-300),
                        "s--", label=r"$\rho_{typ}(0)/\rho_{arith}(0)$",
                    )
                    axes[0, 0].axhline(0.0, color="0.45", linewidth=0.7)
                    axes[0, 1].axhline(0.0, color="0.45", linewidth=0.7)
                    axes[0, 0].set_ylabel(r"$L_{12}$")
                    axes[0, 1].set_ylabel(r"$S$")
                    axes[1, 0].set_ylabel(r"$\sigma=L_{11}$")
                    axes[1, 1].set_ylabel("typical/arithmetic ratio")
                    axes[0, 0].legend(frameon=False)
                    axes[1, 1].legend(frameon=False, fontsize=7)
                    for axis in axes.flat:
                        axis.set_xlabel(r"disorder $\Delta/W$")
                        axis.tick_params(which="both", direction="in", top=True, right=True)
                    fig.suptitle(
                        rf"$n_c={filling:g}$, $U/W={interaction / bandwidth:g}$, "
                        rf"$T/W={temperature / bandwidth:g}$"
                    )
                    fig.tight_layout()
                    path = output / (
                        f"mechanism_linecut_n_{_tag(filling)}_U_{_tag(interaction / bandwidth)}"
                        f"_T_{_tag(temperature / bandwidth)}.pdf"
                    )
                    _save_publication_figure(fig, path)
                    plt.close(fig)
                    outputs.append(path)
    return outputs


def _reliability_summary(
    rows: list[dict], sigma_floors: list[float]
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["branch"] == "typ":
            grouped[(float(row["target_filling"]), float(row["temperature"]))].append(row)
    output: list[dict[str, float | int | str]] = []
    for (filling, temperature), group in sorted(grouped.items()):
        for floor in sigma_floors:
            reliable = [row for row in group if float(row["sigma"]) >= floor]
            if not reliable:
                continue
            maximum_s = max(reliable, key=lambda row: abs(float(row["thermopower"])))
            maximum_pf = max(reliable, key=lambda row: float(row["power_factor"]))
            output.append(
                {
                    "target_filling": filling,
                    "temperature": temperature,
                    "sigma_floor": floor,
                    "max_abs_S_typ": abs(float(maximum_s["thermopower"])),
                    "max_abs_S_interaction": maximum_s["interaction"],
                    "max_abs_S_disorder": maximum_s["disorder_full_width"],
                    "max_power_factor_typ": maximum_pf["power_factor"],
                    "max_power_factor_interaction": maximum_pf["interaction"],
                    "max_power_factor_disorder": maximum_pf["disorder_full_width"],
                    "reliable_points": len(reliable),
                }
            )
    return output


def _plot_reliability(rows: list[dict], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    temperatures = sorted({float(row["temperature"]) for row in rows})
    floors = sorted({float(row["sigma_floor"]) for row in rows}, reverse=True)
    for temperature in temperatures:
        selected = [row for row in rows if np.isclose(float(row["temperature"]), temperature)]
        with plt.rc_context(_publication_style()):
            fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
            for floor in floors:
                line = sorted(
                    (row for row in selected if np.isclose(float(row["sigma_floor"]), floor)),
                    key=lambda row: float(row["target_filling"]),
                )
                label = rf"$\sigma_{{typ}}\geq {floor:.0e}$"
                axes[0].plot(
                    [float(row["target_filling"]) for row in line],
                    [float(row["max_abs_S_typ"]) for row in line], "o-", label=label,
                )
                axes[1].semilogy(
                    [float(row["target_filling"]) for row in line],
                    [float(row["max_power_factor_typ"]) for row in line], "o-", label=label,
                )
            axes[0].set_ylabel(r"reliable maximum $|S_{typ}|$")
            axes[1].set_ylabel(r"maximum $S_{typ}^2\sigma_{typ}$")
            for axis in axes:
                axis.set_xlabel(r"filling $n_c$")
                axis.tick_params(which="both", direction="in", top=True, right=True)
            axes[0].legend(frameon=False, fontsize=7)
            fig.suptitle(rf"reliability test, $T/W={temperature:g}$")
            fig.tight_layout()
            path = output / f"reliability_maxima_T_{_tag(temperature)}.pdf"
            _save_publication_figure(fig, path)
            plt.close(fig)
            outputs.append(path)
    return outputs


def analyze_transport_mechanism(
    summary_paths: list[str | Path],
    points_roots: list[str | Path],
    output_directory: str | Path,
    interactions: list[float],
    sigma_floors: list[float],
    bandwidth: float = 1.0,
    temperatures: list[float] | None = None,
    skip_linecuts: bool = False,
) -> list[Path]:
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics, summary_rows = calculate_mechanism_diagnostics(
        summary_paths, points_roots, temperatures
    )
    if not diagnostics:
        raise ValueError("no rows match the requested datasets and temperatures")
    reliability = _reliability_summary(diagnostics, sigma_floors)
    outputs = [
        _write_rows(output / "mechanism_diagnostics.csv", diagnostics),
        _write_rows(output / "reliability_summary.csv", reliability),
    ]
    outputs.extend(_plot_centroid_correlations(diagnostics, output))
    outputs.extend(_plot_contours_by_filling(summary_rows, output, bandwidth))
    if not skip_linecuts:
        outputs.extend(_plot_linecuts(diagnostics, output, interactions, bandwidth))
    outputs.extend(_plot_reliability(reliability, output))
    return outputs
