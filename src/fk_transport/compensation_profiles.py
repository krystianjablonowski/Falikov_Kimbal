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


def _bracket(
    candidates: list[tuple[float, dict, dict, float]], target: float
) -> tuple[tuple[float, dict, dict, float], tuple[float, dict, dict, float], float]:
    """Return adjacent common-grid points and the linear weight at target."""
    ordered = sorted(candidates, key=lambda item: item[0])
    lower = max((item for item in ordered if item[0] <= target), default=ordered[0], key=lambda item: item[0])
    upper = min((item for item in ordered if item[0] >= target), default=ordered[-1], key=lambda item: item[0])
    span = upper[0] - lower[0]
    alpha = 0.0 if span == 0.0 else (target - lower[0]) / span
    return lower, upper, float(np.clip(alpha, 0.0, 1.0))


def _interpolated_selection(
    kind: str,
    target: float,
    candidates: list[tuple[float, dict, dict, float]],
    filling: float,
    temperature: float,
    interaction: float,
    bandwidth: float,
) -> dict[str, float | int | str]:
    low, high, alpha = _bracket(candidates, target)

    def blend(branch_index: int, field: str) -> float:
        return (1.0 - alpha) * _number(low[branch_index], field) + alpha * _number(
            high[branch_index], field
        )

    sigma_arith = blend(1, "sigma")
    sigma_typ = blend(2, "sigma")
    l12_arith = blend(1, "L12")
    l12_typ = blend(2, "L12")
    s_arith = -l12_arith / (temperature * sigma_arith) if sigma_arith > 0.0 else float("nan")
    s_typ = -l12_typ / (temperature * sigma_typ) if sigma_typ > 0.0 else float("nan")
    nearest = low if alpha <= 0.5 else high
    return {
        "selection": kind,
        "target_filling": filling,
        "temperature": temperature,
        "temperature_over_W": temperature / bandwidth,
        "interaction": interaction,
        "interaction_over_W": interaction / bandwidth,
        "disorder_full_width": target,
        "disorder_over_W": target / bandwidth,
        "interpolation_disorder_low": low[0],
        "interpolation_disorder_high": high[0],
        "interpolation_alpha": alpha,
        "sigma_arith": sigma_arith,
        "sigma_typ": sigma_typ,
        "sigma_typ_over_arith": sigma_typ / sigma_arith if sigma_arith > 0.0 else float("nan"),
        "S_arith": s_arith,
        "S_typ": s_typ,
        "L12_arith": l12_arith,
        "L12_typ": l12_typ,
        "point_index_arith": int(_number(nearest[1], "index")),
        "point_index_typ": int(_number(nearest[2], "index")),
        "point_index_arith_low": int(_number(low[1], "index")),
        "point_index_arith_high": int(_number(high[1], "index")),
        "point_index_typ_low": int(_number(low[2], "index")),
        "point_index_typ_high": int(_number(high[2], "index")),
    }


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
                choices: list[tuple[str, float]] = []
                choices.append(("metallic", min(
                    finite_ratio, key=lambda item: abs(np.log10(item[3] / metal_ratio))
                )[0]))
                reliable = [item for item in candidates if _number(item[2], "sigma") >= sigma_floor]
                if reliable:
                    choices.append(("max_abs_S_typ", max(
                        reliable, key=lambda item: abs(_number(item[2], "thermopower"))
                    )[0]))
                crossing = crossing_lookup.get((filling, temperature, interaction))
                if crossing is not None:
                    choices.extend(
                        [
                            ("compensation_arith", float(crossing["disorder_arith"])),
                            ("compensation_typ", float(crossing["disorder_typ"])),
                        ]
                    )
                choices.append(("localized_edge", min(
                    finite_ratio, key=lambda item: abs(np.log10(item[3] / edge_ratio))
                )[0]))
                for kind, disorder in choices:
                    selected.append(
                        _interpolated_selection(
                            kind, disorder, candidates, filling, temperature,
                            interaction, bandwidth,
                        )
                    )
    return selected


def _load_profile(
    points_root: Path,
    index: int,
    temperature: float,
    high_index: int | None = None,
    alpha: float = 0.0,
) -> dict[str, np.ndarray | float]:
    def load(point_index: int) -> tuple[np.ndarray, np.ndarray]:
        path = points_root / f"point_{point_index:06d}" / "solution.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            return (
                np.asarray(data["omega"], dtype=float),
                np.maximum(np.asarray(data["tau"], dtype=float), 0.0),
            )

    omega, tau = load(index)
    if high_index is not None and high_index != index and alpha > 0.0:
        omega_high, tau_high = load(high_index)
        if not np.array_equal(omega, omega_high):
            tau_high = np.interp(omega, omega_high, tau_high)
        tau = (1.0 - alpha) * tau + alpha * tau_high
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

    order = (
        "metallic", "max_abs_S_typ", "compensation_arith",
        "compensation_typ", "localized_edge",
    )
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
        "compensation_arith": r"$L_{12}^{\rm arith}=0$",
        "compensation_typ": r"$L_{12}^{\rm typ}=0$",
        "localized_edge": "localized edge",
    }
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(4, len(chosen), figsize=(3.0 * len(chosen), 8.6), squeeze=False)
        for column, row in enumerate(chosen):
            for branch in ("arith", "typ"):
                profile = _load_profile(
                    points_root,
                    int(row[f"point_index_{branch}_low"]),
                    temperature,
                    int(row[f"point_index_{branch}_high"]),
                    float(row["interpolation_alpha"]),
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
            title = labels[str(row["selection"])] + "\n" + rf"$\Delta/W={float(row['disorder_over_W']):g}$"
            if str(row["selection"]).startswith("compensation"):
                title += rf", $S_a={float(row['S_arith']):.2g}$, $S_t={float(row['S_typ']):.2g}$"
            else:
                title += rf", $S_{{\rm typ}}={float(row['S_typ']):.2g}$"
            axes[0, column].set_title(title)
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
                    f"transport_profiles_n_{filling_tag}_U_{interaction_tag}_T_{temperature_tag}.pdf"
                ),
                omega_window,
            )
        )
    return outputs
