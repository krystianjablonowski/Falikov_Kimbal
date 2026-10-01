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


def _write_rows(path: Path, rows: list[dict]) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _load_solution(points_root: Path, index: int) -> dict[str, np.ndarray]:
    path = points_root / f"point_{index:06d}" / "solution.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {"omega", "tau", "rho_arith", "rho_typ"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} does not contain {', '.join(sorted(missing))}")
        return {name: np.asarray(data[name], dtype=float) for name in required}


def _moments(omega: np.ndarray, curve: np.ndarray, temperature: float) -> tuple[float, float, float]:
    weight = minus_fermi_derivative(omega, temperature)
    m0 = _integral(weight * curve, omega)
    m1 = _integral(weight * omega * curve, omega)
    centroid = m1 / m0 if m0 > 0.0 else float("nan")
    return m0, m1, centroid


def calculate_publication_diagnostics(
    summary_paths: list[str | Path],
    points_roots: list[str | Path],
    temperatures: list[float] | None = None,
) -> tuple[list[dict], list[dict], list[dict[str, str]]]:
    """Reconstruct Kubo/Kelvin and arith/typ covariance identities from saved spectra."""
    if len(summary_paths) != len(points_roots):
        raise ValueError("--summaries and --points-roots must have equal lengths")
    point_rows: list[dict] = []
    summary_rows: list[dict[str, str]] = []
    arrays: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for dataset, (summary_path, root_value) in enumerate(zip(summary_paths, points_roots)):
        rows = _read_rows(Path(summary_path))
        if temperatures:
            rows = [
                row for row in rows
                if any(np.isclose(_number(row, "temperature"), value) for value in temperatures)
            ]
        summary_rows.extend(rows)
        root = Path(root_value)
        for row in rows:
            index = int(_number(row, "index"))
            key = (dataset, index)
            if key not in arrays:
                arrays[key] = _load_solution(root, index)
            data = arrays[key]
            omega = data["omega"]
            temperature = _number(row, "temperature")
            tau = np.maximum(data["tau"], 0.0)
            rho_arith = np.maximum(data["rho_arith"], 0.0)
            rho_typ = np.maximum(data["rho_typ"], 0.0)
            m0, m1, transport_centroid = _moments(omega, tau, temperature)
            n0, n1, spectral_centroid = _moments(omega, rho_arith, temperature)
            kubo = -transport_centroid / temperature
            kelvin = -spectral_centroid / temperature
            zero = int(np.argmin(np.abs(omega)))
            point_rows.append(
                {
                    "dataset": dataset,
                    "index": index,
                    "branch": row["branch"],
                    "target_filling": _number(row, "target_filling"),
                    "temperature": temperature,
                    "interaction": _number(row, "interaction"),
                    "disorder_full_width": _number(row, "disorder_full_width"),
                    "M0_transport": m0,
                    "M1_transport": m1,
                    "N0_spectral": n0,
                    "N1_spectral": n1,
                    "S_kubo_reconstructed": kubo,
                    "S_kubo_summary": _number(row, "thermopower"),
                    "S_kelvin": kelvin,
                    "S_kubo_minus_kelvin": kubo - kelvin,
                    "lorenz_over_L0": _number(row, "lorenz_over_L0"),
                    "rho_arith_zero": float(rho_arith[zero]),
                    "rho_typ_zero": float(rho_typ[zero]),
                    "rho_typ_over_arith_zero": (
                        float(rho_typ[zero] / rho_arith[zero])
                        if rho_arith[zero] > 0.0 else float("nan")
                    ),
                }
            )

    grouped: dict[tuple[float, ...], dict[str, dict]] = defaultdict(dict)
    for row in point_rows:
        key = (
            int(row["dataset"]),
            round(float(row["target_filling"]), 12),
            round(float(row["temperature"]), 12),
            round(float(row["interaction"]), 12),
            round(float(row["disorder_full_width"]), 12),
        )
        grouped[key][str(row["branch"])] = row
    covariance_rows: list[dict] = []
    for branches in grouped.values():
        if set(branches) != {"arith", "typ"}:
            continue
        arith, typ = branches["arith"], branches["typ"]
        data_a = arrays[(int(arith["dataset"]), int(arith["index"]))]
        data_t = arrays[(int(typ["dataset"]), int(typ["index"]))]
        omega = data_a["omega"]
        tau_a = np.maximum(data_a["tau"], 0.0)
        tau_t = np.maximum(data_t["tau"], 0.0)
        if not np.array_equal(omega, data_t["omega"]):
            tau_t = np.interp(omega, data_t["omega"], tau_t, left=0.0, right=0.0)
        temperature = float(arith["temperature"])
        fermi_weight = minus_fermi_derivative(omega, temperature)
        normalization = _integral(fermi_weight * tau_a, omega)
        if normalization <= 0.0:
            continue
        probability = fermi_weight * tau_a / normalization
        floor = max(float(np.max(tau_a)) * 1.0e-300, np.finfo(float).tiny)
        ratio = tau_t / np.maximum(tau_a, floor)
        mean_omega = _integral(probability * omega, omega)
        mean_ratio = _integral(probability * ratio, omega)
        covariance = _integral(probability * (omega - mean_omega) * (ratio - mean_ratio), omega)
        prediction = -covariance / (temperature * mean_ratio) if mean_ratio > 0.0 else float("nan")
        measured = float(typ["S_kubo_reconstructed"]) - float(arith["S_kubo_reconstructed"])
        covariance_rows.append(
            {
                "target_filling": arith["target_filling"],
                "temperature": temperature,
                "interaction": arith["interaction"],
                "disorder_full_width": arith["disorder_full_width"],
                "mean_transport_ratio": mean_ratio,
                "covariance_omega_ratio": covariance,
                "predicted_delta_S": prediction,
                "measured_delta_S": measured,
                "identity_absolute_error": abs(prediction - measured),
                "sigma_typ_over_arith": mean_ratio,
            }
        )
    return point_rows, covariance_rows, summary_rows


def _crossing(x: np.ndarray, values: np.ndarray, level: float) -> float | None:
    valid = np.isfinite(values) & (values > 0.0)
    x, values = x[valid], values[valid]
    if x.size < 2:
        return None
    shifted = np.log10(values) - np.log10(level)
    for index in range(x.size - 1):
        if shifted[index] == 0.0:
            return float(x[index])
        if shifted[index] * shifted[index + 1] < 0.0:
            fraction = -shifted[index] / (shifted[index + 1] - shifted[index])
            return float(x[index] + fraction * (x[index + 1] - x[index]))
    return None


def fit_large_u_boundaries(
    point_rows: list[dict], threshold: float, u_min: float
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for row in point_rows:
        if row["branch"] == "typ":
            grouped[(float(row["target_filling"]), float(row["temperature"]), float(row["interaction"]))].append(row)
    boundaries: list[dict] = []
    for (filling, temperature, interaction), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: float(item["disorder_full_width"]))
        crossing = _crossing(
            np.asarray([float(row["disorder_full_width"]) for row in rows]),
            np.asarray([float(row["rho_typ_over_arith_zero"]) for row in rows]),
            threshold,
        )
        if crossing is not None:
            boundaries.append(
                {"target_filling": filling, "temperature": temperature, "interaction": interaction,
                 "inverse_u_squared": 1.0 / interaction**2 if interaction > 0.0 else float("nan"),
                 "critical_disorder": crossing, "ratio_threshold": threshold}
            )
    fits: list[dict] = []
    by_group: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in boundaries:
        if float(row["interaction"]) >= u_min:
            by_group[(float(row["target_filling"]), float(row["temperature"]))].append(row)
    for (filling, temperature), rows in sorted(by_group.items()):
        if len(rows) < 3:
            continue
        x = np.asarray([float(row["inverse_u_squared"]) for row in rows])
        y = np.asarray([float(row["critical_disorder"]) for row in rows])
        coefficient, intercept = np.polyfit(x, y, 1)
        residual = y - (intercept + coefficient * x)
        fits.append(
            {"target_filling": filling, "temperature": temperature, "u_min": u_min,
             "ratio_threshold": threshold, "points": len(rows), "delta_infinity": float(intercept),
             "coefficient_over_u2": float(coefficient),
             "rmse": float(np.sqrt(np.mean(residual**2))),
             "r_squared": float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)) if np.std(y) > 0.0 else float("nan")}
        )
    return boundaries, fits


def _plot_covariance(rows: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = [row for row in rows if np.isfinite(float(row["predicted_delta_S"])) and np.isfinite(float(row["measured_delta_S"]))]
    with plt.rc_context(_publication_style()):
        fig, axis = plt.subplots(figsize=(4.0, 3.4))
        x = np.asarray([float(row["predicted_delta_S"]) for row in finite])
        y = np.asarray([float(row["measured_delta_S"]) for row in finite])
        color = np.asarray([float(row["target_filling"]) for row in finite])
        axis.scatter(x, y, c=color, cmap="viridis", s=7, alpha=0.65)
        if x.size:
            limit = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0e-12)
            axis.plot([-limit, limit], [-limit, limit], "k:", linewidth=0.8)
        axis.set_xlabel(r"covariance prediction for $S_{typ}-S_{arith}$")
        axis.set_ylabel(r"measured $S_{typ}-S_{arith}$")
        fig.tight_layout()
        output = _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_kubo_kelvin(rows: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = [row for row in rows if np.isfinite(float(row["S_kubo_reconstructed"])) and np.isfinite(float(row["S_kelvin"]))]
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
        for axis, branch in zip(axes, ("arith", "typ")):
            selected = [row for row in finite if row["branch"] == branch]
            x = np.asarray([float(row["S_kelvin"]) for row in selected])
            y = np.asarray([float(row["S_kubo_reconstructed"]) for row in selected])
            c = np.asarray([float(row["disorder_full_width"]) for row in selected])
            axis.scatter(x, y, c=c, cmap="plasma", s=7, alpha=0.65)
            if x.size:
                limit = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0e-12)
                axis.plot([-limit, limit], [-limit, limit], "k:", linewidth=0.8)
            axis.axhline(0.0, color="0.6", linewidth=0.6)
            axis.axvline(0.0, color="0.6", linewidth=0.6)
            axis.set_title(branch)
            axis.set_xlabel(r"Kelvin $S_K$")
        axes[0].set_ylabel(r"Kubo $S$")
        fig.tight_layout()
        output = _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_large_u(boundaries: list[dict], fits: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(_publication_style()):
        fig, axis = plt.subplots(figsize=(4.4, 3.5))
        groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
        for row in boundaries:
            groups[(float(row["target_filling"]), float(row["temperature"]))].append(row)
        fit_lookup = {(float(row["target_filling"]), float(row["temperature"])): row for row in fits}
        for key, rows in sorted(groups.items()):
            rows = [row for row in rows if np.isfinite(float(row["inverse_u_squared"]))]
            rows.sort(key=lambda row: float(row["inverse_u_squared"]))
            x = np.asarray([float(row["inverse_u_squared"]) for row in rows])
            y = np.asarray([float(row["critical_disorder"]) for row in rows])
            line, = axis.plot(x, y, "o", markersize=2.5, label=rf"$n={key[0]:g},T={key[1]:g}$")
            fit = fit_lookup.get(key)
            if fit and x.size:
                xx = np.linspace(0.0, float(np.max(x)), 100)
                yy = float(fit["delta_infinity"]) + float(fit["coefficient_over_u2"]) * xx
                axis.plot(xx, yy, color=line.get_color(), linewidth=1.0)
        axis.set_xlabel(r"$1/(U/W)^2$")
        axis.set_ylabel(r"localization boundary $\Delta_c/W$")
        axis.legend(frameon=False, fontsize=6, ncol=2)
        fig.tight_layout()
        output = _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_phase_lines(crossings: list[dict], boundaries: list[dict], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    products: list[Path] = []
    groups = sorted({(float(row["target_filling"]), float(row["temperature"])) for row in boundaries})
    for filling, temperature in groups:
        with plt.rc_context(_publication_style()):
            fig, axis = plt.subplots(figsize=(4.5, 3.6))
            for branch, style in (("arith", "-"), ("typ", "--")):
                selected = [
                    row for row in crossings
                    if row["branch"] == branch
                    and np.isclose(float(row["target_filling"]), filling)
                    and np.isclose(float(row["temperature"]), temperature)
                    and int(row["crossing_index"]) == 0
                ]
                selected.sort(key=lambda row: float(row["interaction"]))
                if selected:
                    axis.plot(
                        [float(row["disorder_full_width"]) for row in selected],
                        [float(row["interaction"]) for row in selected],
                        style, linewidth=1.3, label=rf"$S_{{{branch}}}=0$",
                    )
            localized = [
                row for row in boundaries
                if np.isclose(float(row["target_filling"]), filling)
                and np.isclose(float(row["temperature"]), temperature)
            ]
            localized.sort(key=lambda row: float(row["interaction"]))
            if localized:
                axis.plot(
                    [float(row["critical_disorder"]) for row in localized],
                    [float(row["interaction"]) for row in localized],
                    "ko-", markersize=2.5, linewidth=1.0,
                    label=rf"$\rho_{{typ}}(0)/\rho_{{arith}}(0)={localized[0]['ratio_threshold']:.0e}$",
                )
            axis.plot([0.0, 3.2], [0.0, 3.2], color="0.65", linestyle=":", linewidth=0.8,
                      label=r"$U=\Delta$")
            axis.set_xlabel(r"disorder $\Delta/W$")
            axis.set_ylabel(r"interaction $U/W$")
            axis.set_title(rf"$n_c={filling:g}$, $T/W={temperature:g}$")
            axis.legend(frameon=False, fontsize=7)
            axis.set_xlim(left=0.0)
            axis.set_ylim(bottom=0.0)
            fig.tight_layout()
            tag = f"n_{filling:.6g}_T_{temperature:.6g}".replace(".", "p")
            path = _save_publication_figure(fig, output / f"phase_lines_{tag}.pdf")
            plt.close(fig)
            products.append(path)
    return products


def analyze_publication_tests(
    summary_paths: list[str | Path], points_roots: list[str | Path], output_directory: str | Path,
    bandwidth: float = 1.0, temperatures: list[float] | None = None,
    localization_threshold: float = 1.0e-4, u_min: float = 1.5,
) -> list[Path]:
    if bandwidth <= 0.0 or localization_threshold <= 0.0 or u_min <= 0.0:
        raise ValueError("bandwidth, localization threshold, and u_min must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    points, covariance, summaries = calculate_publication_diagnostics(summary_paths, points_roots, temperatures)
    boundaries, fits = fit_large_u_boundaries(points, localization_threshold, u_min * bandwidth)
    crossings = extract_zero_crossings(summaries, bandwidth)
    products = [
        _write_rows(output / "publication_point_diagnostics.csv", points),
        _write_rows(output / "arith_typ_covariance_test.csv", covariance),
        _write_rows(output / "localization_boundaries.csv", boundaries),
        _write_rows(output / "large_u_localization_fits.csv", fits),
        _write_rows(output / "thermopower_zero_crossings.csv", crossings),
        _plot_covariance(covariance, output / "arith_typ_covariance_test.pdf"),
        _plot_kubo_kelvin(points, output / "kubo_kelvin_comparison.pdf"),
        _plot_large_u(boundaries, fits, output / "large_u_localization_scaling.pdf"),
    ]
    products.extend(_plot_phase_lines(crossings, boundaries, output))
    return products
