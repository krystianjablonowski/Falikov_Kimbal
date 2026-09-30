from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .compensation_profiles import select_profile_points
from .observables import minus_fermi_derivative
from .plotting import _publication_style, _save_publication_figure
from .thermopower_compensation import _read_rows


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _integral(values: np.ndarray, omega: np.ndarray) -> float:
    return float(np.trapz(values, omega))


def _load_spectral_profile(
    points_root: Path,
    index: int,
    branch: str,
    temperature: float,
    high_index: int | None = None,
    alpha: float = 0.0,
) -> dict[str, np.ndarray | float]:
    field = "rho_arith" if branch == "arith" else "rho_typ"

    def load(point_index: int) -> tuple[np.ndarray, np.ndarray]:
        path = points_root / f"point_{point_index:06d}" / "solution.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            missing = {"omega", field}.difference(data.files)
            if missing:
                raise KeyError(f"{path} does not contain {', '.join(sorted(missing))}")
            return (
                np.asarray(data["omega"], dtype=float),
                np.maximum(np.asarray(data[field], dtype=float), 0.0),
            )

    omega, rho = load(index)
    if high_index is not None and high_index != index and alpha > 0.0:
        omega_high, rho_high = load(high_index)
        if not np.array_equal(omega, omega_high):
            rho_high = np.interp(omega, omega_high, rho_high)
        rho = (1.0 - alpha) * rho + alpha * rho_high

    weight = minus_fermi_derivative(omega, temperature)
    density = weight * rho
    normalization = _integral(density, omega)
    probability = (
        density / normalization
        if normalization > 0.0
        else np.full_like(density, np.nan)
    )
    moment_integrand = omega * density
    absolute_moment = _integral(np.abs(moment_integrand), omega)
    normalized_integrand = (
        moment_integrand / absolute_moment
        if absolute_moment > 0.0
        else np.zeros_like(moment_integrand)
    )
    cumulative = np.zeros_like(omega)
    cumulative[1:] = np.cumsum(
        0.5 * (normalized_integrand[1:] + normalized_integrand[:-1])
        * np.diff(omega)
    )
    zero_index = int(np.argmin(np.abs(omega)))
    negative = omega < 0.0
    positive = omega > 0.0
    return {
        "omega": omega,
        "rho": rho,
        "probability": probability,
        "normalized_integrand": normalized_integrand,
        "cumulative": cumulative,
        "rho_zero": float(rho[zero_index]),
        "thermal_spectral_weight": normalization,
        "spectral_first_moment": _integral(moment_integrand, omega),
        "spectral_first_moment_negative": _integral(
            moment_integrand[negative], omega[negative]
        ),
        "spectral_first_moment_positive": _integral(
            moment_integrand[positive], omega[positive]
        ),
    }


def _plot_group(
    selections: list[dict[str, float | int | str]],
    points_root: Path,
    output: Path,
    omega_window: float | None,
) -> tuple[Path, list[dict[str, float | int | str]]]:
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
    diagnostics: list[dict[str, float | int | str]] = []
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(4, len(chosen), figsize=(3.0 * len(chosen), 8.6), squeeze=False)
        for column, row in enumerate(chosen):
            loaded: dict[str, dict[str, np.ndarray | float]] = {}
            for branch in ("arith", "typ"):
                profile = _load_spectral_profile(
                    points_root,
                    int(row[f"point_index_{branch}_low"]),
                    branch,
                    temperature,
                    int(row[f"point_index_{branch}_high"]),
                    float(row["interpolation_alpha"]),
                )
                loaded[branch] = profile
                omega = np.asarray(profile["omega"])
                mask = np.abs(omega) <= window
                axes[0, column].plot(
                    omega[mask], np.asarray(profile["rho"])[mask],
                    color=colors[branch], label=branch,
                )
                axes[1, column].plot(
                    omega[mask], np.asarray(profile["probability"])[mask],
                    color=colors[branch],
                )
                axes[2, column].plot(
                    omega[mask], np.asarray(profile["normalized_integrand"])[mask],
                    color=colors[branch],
                )
                axes[3, column].plot(
                    omega[mask], np.asarray(profile["cumulative"])[mask],
                    color=colors[branch],
                )
                diagnostics.append(
                    {
                        **row,
                        "branch": branch,
                        "rho_zero": float(profile["rho_zero"]),
                        "thermal_spectral_weight": float(profile["thermal_spectral_weight"]),
                        "spectral_first_moment": float(profile["spectral_first_moment"]),
                        "spectral_first_moment_negative": float(
                            profile["spectral_first_moment_negative"]
                        ),
                        "spectral_first_moment_positive": float(
                            profile["spectral_first_moment_positive"]
                        ),
                    }
                )
            rho_a = float(loaded["arith"]["rho_zero"])
            rho_t = float(loaded["typ"]["rho_zero"])
            ratio = rho_t / rho_a if rho_a > 0.0 else float("nan")
            for diagnostic in diagnostics[-2:]:
                diagnostic["rho_zero_typ_over_arith"] = ratio
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
        axes[0, 0].set_ylabel(r"spectral DOS $\rho(\omega)$")
        axes[1, 0].set_ylabel(r"$Q_\rho=(-f')\rho/\!\int(-f')\rho$")
        axes[2, 0].set_ylabel(r"$\omega(-f')\rho/\!\int|\omega(-f')\rho|$")
        axes[3, 0].set_ylabel("cumulative spectral moment")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(
            rf"$n_c={filling:g}$, $U/W={interaction:g}$, $T/W={temperature:g}$"
        )
        fig.tight_layout()
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output, diagnostics


def plot_spectral_profiles(
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
    """Plot spectral diagnostics at the transport-profile selection points."""
    rows = _read_rows(summary_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_profile_points(
        rows, interactions, bandwidth, metal_ratio, edge_ratio, sigma_floor
    )
    outputs = [_write_rows(output / "spectral_profile_points.csv", selected)]
    groups: dict[tuple[float, float, float], list[dict]] = {}
    for row in selected:
        key = (
            float(row["target_filling"]),
            float(row["temperature_over_W"]),
            float(row["interaction_over_W"]),
        )
        groups.setdefault(key, []).append(row)
    diagnostics: list[dict[str, float | int | str]] = []
    for (filling, temperature, interaction), group in sorted(groups.items()):
        tag = lambda value: f"{value:.6g}".replace(".", "p")
        figure, group_diagnostics = _plot_group(
            group,
            Path(points_root),
            output / (
                f"spectral_profiles_n_{tag(filling)}_U_{tag(interaction)}_T_{tag(temperature)}.pdf"
            ),
            omega_window,
        )
        outputs.append(figure)
        diagnostics.extend(group_diagnostics)
    outputs.insert(1, _write_rows(output / "spectral_profile_diagnostics.csv", diagnostics))
    return outputs
