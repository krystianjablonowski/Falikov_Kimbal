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
    conductivity_relative_floor: float = 1.0e-4,
) -> tuple[list[dict], list[dict], list[dict[str, str]]]:
    """Reconstruct Kubo/Kelvin and arith/typ covariance identities from saved spectra."""
    if len(summary_paths) != len(points_roots):
        raise ValueError("--summaries and --points-roots must have equal lengths")
    point_rows: list[dict] = []
    summary_rows: list[dict[str, str]] = []
    roots = [Path(value) for value in points_roots]
    for dataset, (summary_path, root_value) in enumerate(zip(summary_paths, points_roots)):
        rows = _read_rows(Path(summary_path))
        if temperatures:
            rows = [
                row for row in rows
                if any(np.isclose(_number(row, "temperature"), value) for value in temperatures)
            ]
        dataset_rows = [{**row, "dataset": str(dataset)} for row in rows]
        summary_rows.extend(dataset_rows)
        root = Path(root_value)
        for row in rows:
            index = int(_number(row, "index"))
            data = _load_solution(root, index)
            omega = data["omega"]
            temperature = _number(row, "temperature")
            tau = np.maximum(data["tau"], 0.0)
            rho_arith = np.maximum(data["rho_arith"], 0.0)
            rho_typ = np.maximum(data["rho_typ"], 0.0)
            m0, m1, transport_centroid = _moments(omega, tau, temperature)
            n0, n1, spectral_centroid = _moments(omega, rho_arith, temperature)
            kubo = -transport_centroid / temperature
            # For the fixed spectrum used by this code, the Kelvin estimate is
            # (d mu/dT)_n = -<omega>_rho/T.  Prefer the value already exported
            # by combined_observables and retain the moment reconstruction as a
            # transparent fallback for older summaries.
            exported_kelvin = _number(row, "dmu_dT_fixed_density")
            kelvin = exported_kelvin if np.isfinite(exported_kelvin) else -spectral_centroid / temperature
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
                    "S_kubo_reconstruction_error": kubo - _number(row, "thermopower"),
                    "S_kelvin": kelvin,
                    "S_kelvin_source": (
                        "summary_dmu_dT_fixed_density"
                        if np.isfinite(exported_kelvin) else "fixed_spectrum_DOS_moment"
                    ),
                    "S_kubo_minus_kelvin": kubo - kelvin,
                    "sigma": _number(row, "sigma"),
                    "lorenz_over_L0": _number(row, "lorenz_over_L0"),
                    "rho_arith_zero": float(rho_arith[zero]),
                    "rho_typ_zero": float(rho_typ[zero]),
                    "rho_typ_over_arith_zero": (
                        float(rho_typ[zero] / rho_arith[zero])
                        if rho_arith[zero] > 0.0 else float("nan")
                    ),
                }
            )

    reliability_groups: dict[tuple[int, float, float, str], list[dict]] = defaultdict(list)
    for row in point_rows:
        reliability_groups[(
            int(row["dataset"]), round(float(row["target_filling"]), 12),
            round(float(row["temperature"]), 12), str(row["branch"]),
        )].append(row)
    for rows in reliability_groups.values():
        finite_sigma = [float(row["sigma"]) for row in rows if np.isfinite(float(row["sigma"]))]
        maximum = max(finite_sigma, default=float("nan"))
        cutoff = conductivity_relative_floor * maximum if maximum > 0.0 else float("nan")
        for row in rows:
            row["sigma_reliability_cutoff"] = cutoff
            row["transport_reliable"] = bool(
                np.isfinite(float(row["sigma"])) and np.isfinite(cutoff)
                and float(row["sigma"]) >= cutoff
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
        data_a = _load_solution(roots[int(arith["dataset"])], int(arith["index"]))
        data_t = _load_solution(roots[int(typ["dataset"])], int(typ["index"]))
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
                "transport_reliable": bool(arith["transport_reliable"] and typ["transport_reliable"]),
            }
        )
    return point_rows, covariance_rows, summary_rows


def boundary_coverage(point_rows: list[dict], threshold: float) -> list[dict]:
    """Report why a threshold contour is present or absent at each fixed U."""
    grouped: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for row in point_rows:
        if row["branch"] == "typ":
            grouped[(float(row["target_filling"]), float(row["temperature"]),
                     float(row["interaction"]))].append(row)
    output: list[dict] = []
    for (filling, temperature, interaction), rows in sorted(grouped.items()):
        values = np.asarray([float(row["rho_typ_over_arith_zero"]) for row in rows])
        valid = values[np.isfinite(values) & (values > 0.0)]
        if valid.size < 2:
            status = "insufficient_data"
        elif float(np.min(valid)) > threshold:
            status = "threshold_not_reached"
        elif float(np.max(valid)) < threshold:
            status = "already_below_threshold"
        else:
            status = "crossed"
        output.append({
            "target_filling": filling, "temperature": temperature,
            "interaction": interaction, "ratio_threshold": threshold,
            "valid_disorder_points": int(valid.size),
            "minimum_ratio": float(np.min(valid)) if valid.size else float("nan"),
            "maximum_ratio": float(np.max(valid)) if valid.size else float("nan"),
            "status": status,
        })
    return output


def _crossing_candidates(
    x: np.ndarray, values: np.ndarray, level: float, descending_only: bool = True
) -> list[float]:
    valid = np.isfinite(values) & (values > 0.0)
    x, values = x[valid], values[valid]
    if x.size < 2:
        return []
    order = np.argsort(x)
    x, values = x[order], values[order]
    shifted = np.log10(values) - np.log10(level)
    found: list[float] = []
    for index in range(x.size - 1):
        if shifted[index] == 0.0:
            found.append(float(x[index]))
        if shifted[index] * shifted[index + 1] < 0.0:
            if descending_only and not (shifted[index] > 0.0 > shifted[index + 1]):
                continue
            fraction = -shifted[index] / (shifted[index + 1] - shifted[index])
            found.append(float(x[index] + fraction * (x[index + 1] - x[index])))
    return found


def _crossing(x: np.ndarray, values: np.ndarray, level: float) -> float | None:
    candidates = _crossing_candidates(x, values, level)
    return candidates[0] if candidates else None


def fit_large_u_boundaries(
    point_rows: list[dict], threshold: float, u_min: float
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for row in point_rows:
        if row["branch"] == "typ":
            grouped[(float(row["target_filling"]), float(row["temperature"]), float(row["interaction"]))].append(row)
    candidate_groups: dict[tuple[float, float], list[tuple[float, list[float]]]] = defaultdict(list)
    for (filling, temperature, interaction), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: float(item["disorder_full_width"]))
        candidates = _crossing_candidates(
            np.asarray([float(row["disorder_full_width"]) for row in rows]),
            np.asarray([float(row["rho_typ_over_arith_zero"]) for row in rows]), threshold,
        )
        candidate_groups[(filling, temperature)].append((interaction, candidates))

    boundaries: list[dict] = []
    # Follow one continuous descending-threshold branch from large to small U.
    # This prevents switching between unrelated crossings when the ratio is
    # weakly non-monotone because of a coarse grid or the spectral floor.
    for (filling, temperature), samples in sorted(candidate_groups.items()):
        previous: float | None = None
        for interaction, candidates in sorted(samples, reverse=True):
            if not candidates:
                previous = None
                continue
            crossing = min(candidates, key=lambda value: abs(value - previous)) if previous is not None else candidates[0]
            jump = abs(crossing - previous) if previous is not None else float("nan")
            boundaries.append(
                {"target_filling": filling, "temperature": temperature, "interaction": interaction,
                 "inverse_u_squared": 1.0 / interaction**2 if interaction > 0.0 else float("nan"),
                 "critical_disorder": crossing, "ratio_threshold": threshold,
                 "crossing_candidates": len(candidates), "jump_from_previous_u": jump}
            )
            previous = crossing
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
    finite = [row for row in rows if row.get("transport_reliable", True)
              and np.isfinite(float(row["predicted_delta_S"]))
              and np.isfinite(float(row["measured_delta_S"]))]
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
        axis.set_title("covariance identity check")
        if not finite:
            axis.text(0.5, 0.5, "no transport-reliable points", ha="center", va="center",
                      transform=axis.transAxes)
        fig.tight_layout()
        output = _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_kubo_kelvin(rows: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = [row for row in rows if row.get("transport_reliable", True)
              and np.isfinite(float(row["S_kubo_reconstructed"]))
              and np.isfinite(float(row["S_kelvin"]))]
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
            if not selected:
                axis.text(0.5, 0.5, "no transport-reliable points", ha="center", va="center",
                          transform=axis.transAxes)
        axes[0].set_ylabel(r"Kubo $S$")
        fig.tight_layout()
        output = _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_large_u(boundaries: list[dict], fits: list[dict], coverage: list[dict], output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(_publication_style()):
        fig, axis = plt.subplots(figsize=(4.4, 3.5))
        groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
        fit_lookup = {(float(row["target_filling"]), float(row["temperature"])): row for row in fits}
        for row in boundaries:
            key = (float(row["target_filling"]), float(row["temperature"]))
            fit = fit_lookup.get(key)
            if fit and float(row["interaction"]) >= float(fit["u_min"]):
                groups[key].append(row)
        for key, rows in sorted(groups.items()):
            rows = [row for row in rows if np.isfinite(float(row["inverse_u_squared"]))]
            rows.sort(key=lambda row: float(row["inverse_u_squared"]))
            x = np.asarray([float(row["inverse_u_squared"]) for row in rows])
            y = np.asarray([float(row["critical_disorder"]) for row in rows])
            line, = axis.plot(x, y, "o", markersize=2.5, label=rf"$n={key[0]:g},T={key[1]:g}$")
            fit = fit_lookup.get(key)
            if fit and x.size:
                # Never extrapolate the large-U fit into the small-U regime.
                xx = np.linspace(0.0, 1.0 / float(fit["u_min"])**2, 100)
                yy = float(fit["delta_infinity"]) + float(fit["coefficient_over_u2"]) * xx
                axis.plot(xx, yy, color=line.get_color(), linewidth=1.0)
        axis.set_xlabel(r"$1/(U/W)^2$")
        axis.set_ylabel(r"localization boundary $\Delta_c/W$")
        if groups:
            axis.legend(frameon=False, fontsize=6, ncol=2)
        else:
            statuses = defaultdict(int)
            for row in coverage:
                statuses[str(row["status"])] += 1
            explanation = "no threshold crossings"
            if statuses:
                explanation += "\n" + "\n".join(f"{key}: {value}" for key, value in sorted(statuses.items()))
            axis.text(0.5, 0.5, explanation, ha="center", va="center", transform=axis.transAxes)
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
                disorder = np.asarray([float(row["critical_disorder"]) for row in localized])
                interaction = np.asarray([float(row["interaction"]) for row in localized])
                axis.plot(disorder, interaction, "ko", markersize=2.5,
                          label=rf"$\rho_{{typ}}(0)/\rho_{{arith}}(0)={localized[0]['ratio_threshold']:.0e}$")
                if interaction.size > 1:
                    typical_du = float(np.median(np.diff(interaction)))
                    for index in range(interaction.size - 1):
                        adjacent_u = interaction[index + 1] - interaction[index] <= 1.5 * typical_du
                        modest_jump = abs(disorder[index + 1] - disorder[index]) <= 0.5
                        if adjacent_u and modest_jump:
                            axis.plot(disorder[index:index + 2], interaction[index:index + 2], "k-", linewidth=1.0)
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
    conductivity_relative_floor: float = 1.0e-4,
) -> list[Path]:
    if (bandwidth <= 0.0 or localization_threshold <= 0.0 or u_min <= 0.0
            or conductivity_relative_floor <= 0.0):
        raise ValueError("bandwidth, localization threshold, and u_min must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    points, covariance, summaries = calculate_publication_diagnostics(
        summary_paths, points_roots, temperatures, conductivity_relative_floor
    )
    boundaries, fits = fit_large_u_boundaries(points, localization_threshold, u_min * bandwidth)
    coverage = boundary_coverage(points, localization_threshold)
    reliability = {
        (int(row["dataset"]), str(row["branch"]), round(float(row["target_filling"]), 12),
         round(float(row["temperature"]), 12), round(float(row["interaction"]), 12),
         round(float(row["disorder_full_width"]), 12)): bool(row["transport_reliable"])
        for row in points
    }
    reliable_summaries = []
    for row in summaries:
        key = (
            int(float(row["dataset"])), str(row["branch"]),
            round(_number(row, "target_filling"), 12), round(_number(row, "temperature"), 12),
            round(_number(row, "interaction"), 12), round(_number(row, "disorder_full_width"), 12),
        )
        copy = dict(row)
        if not reliability.get(key, False):
            copy["L12"] = "nan"
        reliable_summaries.append(copy)
    crossings = extract_zero_crossings(reliable_summaries, bandwidth)
    products = [
        _write_rows(output / "publication_point_diagnostics.csv", points),
        _write_rows(output / "arith_typ_covariance_test.csv", covariance),
        _write_rows(output / "localization_boundaries.csv", boundaries),
        _write_rows(output / "localization_boundary_coverage.csv", coverage),
        _write_rows(output / "large_u_localization_fits.csv", fits),
        _write_rows(output / "thermopower_zero_crossings.csv", crossings),
        _plot_covariance(covariance, output / "arith_typ_covariance_test.pdf"),
        _plot_kubo_kelvin(points, output / "kubo_kelvin_comparison.pdf"),
        _plot_large_u(boundaries, fits, coverage, output / "large_u_localization_scaling.pdf"),
    ]
    products.extend(_plot_phase_lines(crossings, boundaries, output))
    return products
