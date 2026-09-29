from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .plotting import _draw_map, _grid, _log_norm, _publication_style, _save_publication_figure


def _number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _key(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        _number(row, "target_filling"),
        _number(row, "temperature"),
        _number(row, "interaction"),
        _number(row, "disorder_full_width"),
    )


def _coupled_modes(row: dict[str, str], floor: float = 1.0e-14) -> tuple[float, float]:
    """Eigenvalues of M K^-1 in the (particle, heat) moment basis."""
    transport = np.array(
        [[_number(row, "L11"), _number(row, "L12")],
         [_number(row, "L12"), _number(row, "L22")]],
        dtype=float,
    )
    susceptibility = np.array(
        [[_number(row, "K0_thermo"), _number(row, "K1_thermo")],
         [_number(row, "K1_thermo"), _number(row, "K2_thermo")]],
        dtype=float,
    )
    if not np.all(np.isfinite(transport)) or not np.all(np.isfinite(susceptibility)):
        return float("nan"), float("nan")
    eigenvalues, eigenvectors = np.linalg.eigh(susceptibility)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) <= floor * scale:
        return float("nan"), float("nan")
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    symmetric_diffusion = inverse_sqrt @ transport @ inverse_sqrt
    modes = np.linalg.eigvalsh(0.5 * (symmetric_diffusion + symmetric_diffusion.T))
    tolerance = floor * max(float(np.max(np.abs(modes))), 1.0)
    if float(modes.min()) < -tolerance:
        return float("nan"), float("nan")
    modes = np.maximum(modes, 0.0)
    return float(modes[0]), float(modes[1])


def derive_finite_filling_rows(summary_path: str | Path) -> list[dict[str, str | float]]:
    with Path(summary_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary contains no rows")
    arithmetic = {_key(row): row for row in rows if row.get("branch") == "arith"}
    derived: list[dict[str, str | float]] = []
    for row in rows:
        result: dict[str, str | float] = dict(row)
        sigma = _number(row, "sigma")
        kappa = _number(row, "kappa_e")
        thermopower = _number(row, "thermopower")
        temperature = _number(row, "temperature")
        power_factor = thermopower**2 * sigma
        zt_electronic = (
            power_factor * temperature / kappa
            if np.isfinite(power_factor) and np.isfinite(kappa) and kappa > 0.0
            else float("nan")
        )
        d_minus, d_plus = _coupled_modes(row)
        reference = arithmetic.get(_key(row))
        sigma_arith = _number(reference, "sigma") if reference else float("nan")
        kappa_arith = _number(reference, "kappa_e") if reference else float("nan")
        result.update(
            {
                "power_factor": power_factor,
                "zt_electronic": zt_electronic,
                "coupled_diffusivity_minus": d_minus,
                "coupled_diffusivity_plus": d_plus,
                "relative_sigma": sigma / sigma_arith if sigma_arith > 0.0 else float("nan"),
                "relative_kappa": kappa / kappa_arith if kappa_arith > 0.0 else float("nan"),
            }
        )
        derived.append(result)
    return derived


def _write_rows(path: Path, rows: list[dict[str, str | float]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _normalized(rows: list[dict[str, str | float]], bandwidth: float) -> list[dict[str, str]]:
    normalized = [{key: str(value) for key, value in row.items()} for row in rows]
    for row in normalized:
        row["interaction"] = str(float(row["interaction"]) / bandwidth)
        row["disorder_full_width"] = str(float(row["disorder_full_width"]) / bandwidth)
    return normalized


def _plot_four_maps(
    rows: list[dict[str, str]],
    filling: float,
    temperature: float,
    fields: tuple[tuple[str, str], tuple[str, str]],
    output: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = ("arith", "typ")
    grids = {
        field: [_grid(rows, branch, temperature, filling, field) for branch in branches]
        for field, _ in fields
    }
    norms = {field: _log_norm([grid[2] for grid in values]) for field, values in grids.items()}
    with plt.rc_context(_publication_style()):
        fig = plt.figure(figsize=(7.8, 5.5))
        spec = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.24, hspace=0.28)
        panel = 0
        for row_index, (field, symbol) in enumerate(fields):
            image = None
            for column, branch in enumerate(branches):
                axis = fig.add_subplot(spec[row_index, column])
                x, y, values = grids[field][column]
                image = _draw_map(
                    axis, x, y, values,
                    rf"{branch}, $n_c={filling:g}$, $T/W={temperature:g}$",
                    norms[field],
                )
                if column == 1:
                    axis.set_ylabel("")
                    axis.tick_params(labelleft=False)
                axis.text(0.03, 0.93, f"({chr(97 + panel)})", transform=axis.transAxes,
                          fontweight="bold", va="top")
                panel += 1
            color_axis = fig.add_subplot(spec[row_index, 2])
            if image is None:
                color_axis.set_visible(False)
            else:
                fig.colorbar(image, cax=color_axis, label=symbol)
        fig.subplots_adjust(left=0.09, right=0.94, bottom=0.10, top=0.94)
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def _plot_relative_maps(
    rows: list[dict[str, str]], filling: float, temperature: float, output: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (("relative_sigma", r"$R_\sigma$"), ("relative_kappa", r"$R_\kappa$"))
    grids = [_grid(rows, "typ", temperature, filling, field) for field, _ in fields]
    norm = _log_norm([grid[2] for grid in grids])
    with plt.rc_context(_publication_style()):
        fig = plt.figure(figsize=(7.4, 3.0))
        spec = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.24)
        image = None
        for panel, ((_, symbol), (x, y, values)) in enumerate(zip(fields, grids)):
            axis = fig.add_subplot(spec[0, panel])
            image = _draw_map(
                axis, x, y, values,
                rf"{symbol}, $n_c={filling:g}$, $T/W={temperature:g}$",
                norm,
            )
            if panel == 1:
                axis.set_ylabel("")
                axis.tick_params(labelleft=False)
            axis.text(0.03, 0.93, f"({chr(97 + panel)})", transform=axis.transAxes,
                      fontweight="bold", va="top")
        color_axis = fig.add_subplot(spec[0, 2])
        if image is None:
            color_axis.set_visible(False)
        else:
            fig.colorbar(image, cax=color_axis, label="typ/arith")
        fig.subplots_adjust(left=0.09, right=0.94, bottom=0.16, top=0.91)
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def analyze_finite_filling(
    summary_path: str | Path,
    output_directory: str | Path,
    bandwidth: float = 1.0,
) -> list[Path]:
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    derived = derive_finite_filling_rows(summary_path)
    csv_path = _write_rows(output / "finite_filling_derived.csv", derived)
    rows = _normalized(derived, bandwidth)
    fillings = sorted({_number(row, "target_filling") for row in rows})
    temperatures = sorted({_number(row, "temperature") for row in rows})
    outputs: list[Path] = [csv_path]
    for filling in fillings:
        filling_tag = f"{filling:.6g}".replace(".", "p")
        for temperature_dimensional in temperatures:
            temperature = temperature_dimensional / bandwidth
            temperature_tag = f"{temperature:.6g}".replace(".", "p")
            outputs.append(_plot_relative_maps(
                rows, filling, temperature_dimensional,
                output / f"relative_transport_heatmaps_n_{filling_tag}_T_{temperature_tag}.png",
            ))
            outputs.append(_plot_four_maps(
                rows, filling, temperature_dimensional,
                (("power_factor", r"$S^2\sigma$"), ("zt_electronic", r"$ZT_{\rm el}$")),
                output / f"thermoelectric_performance_heatmaps_n_{filling_tag}_T_{temperature_tag}.png",
            ))
            outputs.append(_plot_four_maps(
                rows, filling, temperature_dimensional,
                (("coupled_diffusivity_minus", r"$D_-$"),
                 ("coupled_diffusivity_plus", r"$D_+$")),
                output / f"coupled_diffusivity_heatmaps_n_{filling_tag}_T_{temperature_tag}.png",
            ))
    return outputs
