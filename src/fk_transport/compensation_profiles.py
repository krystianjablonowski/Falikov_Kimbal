from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .observables import minus_fermi_derivative
from .plotting import _publication_style, _save_publication_figure
from .thermopower_compensation import (
    _number,
    _read_rows,
    extract_zero_crossings,
    match_branch_crossings,
)


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _nearest_row(rows: list[dict[str, str]], disorder: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(_number(row, "disorder_full_width") - disorder))


def select_profile_points(
    rows: list[dict[str, str]],
    interactions: list[float],
    bandwidth: float = 1.0,
    metal_ratio: float = 0.5,
    edge_ratio: float = 1.0e-4,
    sigma_floor: float = 1.0e-8,
) -> list[dict[str, float | int | str]]:
    """Select common arith/typ grid points that expose thermoelectric filtering."""
    if bandwidth <= 0.0 or metal_ratio <= 0.0 or edge_ratio <= 0.0 or sigma_floor < 0.0:
        raise ValueError("bandwidth and ratio targets must be positive; sigma_floor cannot be negative")
    crossings = extract_zero_crossings(rows, bandwidth)
    matched, _ = match_branch_crossings(crossings)
    crossing_lookup = {
        (
            float(row["target_filling"]),
            float(row["temperature"]),
            float(row["interaction"]),
        ): row
        for row in matched
    }
    fillings = sorted({_number(row, "target_filling") for row in rows})
    temperatures = sorted({_number(row, "temperature") for row in rows})
    selected: list[dict[str, float | int | str]] = []
    for filling in fillings:
        available_u = sorted({_number(row, "interaction") for row in rows})
        chosen_u = sorted({min(available_u, key=lambda value: abs(value - requested)) for requested in interactions})
        for temperature in temperatures:
            for interaction in chosen_u:
                branch_rows = {
                    branch: [
                        row for row in rows
                        if row["branch"] == branch
                        and np.isclose(_number(row, "target_filling"), filling)
                        and np.isclose(_number(row, "temperature"), temperature)
                        and np.isclose(_number(row, "interaction"), interaction)
                    ]
                    for branch in ("arith", "typ")
                }
                arithmetic = {
                    round(_number(row, "disorder_full_width"), 12): row
                    for row in branch_rows["arith"]
                }
                typical = {
                    round(_number(row, "disorder_full_width"), 12): row
                    for row in branch_rows["typ"]
                }
                common = sorted(set(arithmetic).intersection(typical))
                candidates = []
                for disorder_key in common:
                    arith, typ = arithmetic[disorder_key], typical[disorder_key]
                    sigma_arith, sigma_typ = _number(arith, "sigma"), _number(typ, "sigma")
                    ratio = sigma_typ / sigma_arith if sigma_arith > 0.0 else float("nan")
                    candidates.append((float(disorder_key), arith, typ, ratio))
                finite_ratio = [item for item in candidates if np.isfinite(item[3]) and item[3] > 0.0]
                if not finite_ratio:
                    continue
                choices: list[tuple[str, tuple[float, dict, dict, float]]] = []
                choices.append(("metallic", min(
                    finite_ratio, key=lambda item: abs(np.log10(item[3] / metal_ratio))
                )))
                reliable = [item for item in candidates if _number(item[2], "sigma") >= sigma_floor]
                if reliable:
                    choices.append(("max_abs_S_typ", max(
                        reliable, key=lambda item: abs(_number(item[2], "thermopower"))
                    )))
                crossing = crossing_lookup.get((filling, temperature, interaction))
                if crossing is not None:
                    midpoint = 0.5 * (
                        float(crossing["disorder_arith"]) + float(crossing["disorder_typ"])
                    )
                    choices.append(("compensation", min(
                        candidates, key=lambda item: abs(item[0] - midpoint)
                    )))
                choices.append(("localized_edge", min(
                    finite_ratio, key=lambda item: abs(np.log10(item[3] / edge_ratio))
                )))
                for kind, (disorder, arith, typ, ratio) in choices:
                    selected.append(
                        {
                            "selection": kind,
                            "target_filling": filling,
                            "temperature": temperature,
                            "temperature_over_W": temperature / bandwidth,
                            "interaction": interaction,
                            "interaction_over_W": interaction / bandwidth,
                            "disorder_full_width": disorder,
                            "disorder_over_W": disorder / bandwidth,
                            "sigma_arith": _number(arith, "sigma"),
                            "sigma_typ": _number(typ, "sigma"),
                            "sigma_typ_over_arith": ratio,
                            "S_arith": _number(arith, "thermopower"),
                            "S_typ": _number(typ, "thermopower"),
                            "L12_arith": _number(arith, "L12"),
                            "L12_typ": _number(typ, "L12"),
                            "point_index_arith": int(_number(arith, "index")),
                            "point_index_typ": int(_number(typ, "index")),
                        }
                    )
    return selected


def _load_profile(points_root: Path, index: int, temperature: float) -> dict[str, np.ndarray | float]:
    path = points_root / f"point_{index:06d}" / "solution.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        omega = np.asarray(data["omega"], dtype=float)
        tau = np.maximum(np.asarray(data["tau"], dtype=float), 0.0)
    weight = minus_fermi_derivative(omega, temperature)
    density = weight * tau
    l11 = float(np.trapz(density, omega))
    probability = density / l11 if l11 > 0.0 else np.full_like(density, np.nan)
    integrand = omega * density
    absolute_moment = float(np.trapz(np.abs(integrand), omega))
    normalized_integrand = integrand / absolute_moment if absolute_moment > 0.0 else integrand
    cumulative = np.zeros_like(omega)
    cumulative[1:] = np.cumsum(
        0.5 * (normalized_integrand[1:] + normalized_integrand[:-1]) * np.diff(omega)
    )
    return {
        "omega": omega,
        "tau": tau,
        "probability": probability,
        "normalized_integrand": normalized_integrand,
        "cumulative": cumulative,
    }


def _plot_group(
    selections: list[dict[str, float | int | str]],
    points_root: Path,
    output: Path,
    omega_window: float | None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ("metallic", "max_abs_S_typ", "compensation", "localized_edge")
    by_kind = {str(row["selection"]): row for row in selections}
    chosen = [by_kind[kind] for kind in order if kind in by_kind]
    temperature = float(chosen[0]["temperature"])
    filling = float(chosen[0]["target_filling"])
    interaction = float(chosen[0]["interaction_over_W"])
    window = float(omega_window) if omega_window is not None else max(0.25, 10.0 * temperature)
    colors = {"arith": "#2477b3", "typ": "#d94b2b"}
    labels = {
        "metallic": "metallic side",
        "max_abs_S_typ": r"maximum $|S_{\rm typ}|$",
        "compensation": r"near $L_{12}=0$",
        "localized_edge": "localized edge",
    }
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(4, len(chosen), figsize=(3.0 * len(chosen), 8.6), squeeze=False)
        for column, row in enumerate(chosen):
            for branch in ("arith", "typ"):
                profile = _load_profile(
                    points_root, int(row[f"point_index_{branch}"]), temperature
                )
                omega = profile["omega"]
                mask = np.abs(omega) <= window
                axes[0, column].semilogy(
                    omega[mask], np.maximum(profile["tau"][mask], 1.0e-300),
                    color=colors[branch], label=branch,
                )
                axes[1, column].plot(
                    omega[mask], profile["probability"][mask], color=colors[branch]
                )
                axes[2, column].plot(
                    omega[mask], profile["normalized_integrand"][mask], color=colors[branch]
                )
                axes[3, column].plot(
                    omega[mask], profile["cumulative"][mask], color=colors[branch]
                )
            for axis in axes[:, column]:
                axis.axvline(0.0, color="0.35", linewidth=0.7, linestyle=":")
                axis.tick_params(which="both", direction="in", top=True, right=True)
            axes[0, column].set_title(
                labels[str(row["selection"])]
                + "\n"
                + rf"$\Delta/W={float(row['disorder_over_W']):g}$, "
                + rf"$S_{{\rm typ}}={float(row['S_typ']):.2g}$"
            )
            axes[3, column].set_xlabel(r"energy $\omega/W$")
        axes[0, 0].set_ylabel(r"$\tau(\omega)$")
        axes[1, 0].set_ylabel(r"$P(\omega)$")
        axes[2, 0].set_ylabel(r"$\omega(-f')\tau/\!\int|I_{12}|$")
        axes[3, 0].set_ylabel(r"cumulative normalized $L_{12}$")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(
            rf"$n_c={filling:g}$, $U/W={interaction:g}$, $T/W={temperature:g}$"
        )
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def plot_compensation_profiles(
    summary_path: str | Path,
    points_root: str | Path,
    output_directory: str | Path,
    interactions: list[float],
    bandwidth: float = 1.0,
    metal_ratio: float = 0.5,
    edge_ratio: float = 1.0e-4,
    sigma_floor: float = 1.0e-8,
    omega_window: float | None = None,
) -> list[Path]:
    rows = _read_rows(summary_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_profile_points(
        rows, interactions, bandwidth, metal_ratio, edge_ratio, sigma_floor
    )
    outputs = [_write_rows(output / "compensation_profile_points.csv", selected)]
    groups: dict[tuple[float, float, float], list[dict]] = {}
    for row in selected:
        key = (
            float(row["target_filling"]),
            float(row["temperature_over_W"]),
            float(row["interaction_over_W"]),
        )
        groups.setdefault(key, []).append(row)
    for (filling, temperature, interaction), group in sorted(groups.items()):
        filling_tag = f"{filling:.6g}".replace(".", "p")
        temperature_tag = f"{temperature:.6g}".replace(".", "p")
        interaction_tag = f"{interaction:.6g}".replace(".", "p")
        outputs.append(
            _plot_group(
                group,
                Path(points_root),
                output / (
                    f"transport_profiles_n_{filling_tag}_U_{interaction_tag}_T_{temperature_tag}.png"
                ),
                omega_window,
            )
        )
    return outputs
