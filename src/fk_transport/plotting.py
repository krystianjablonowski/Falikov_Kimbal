from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _read_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary contains no successful points")
    return rows


def _grid(
    rows: list[dict[str, str]], branch: str, temperature: float, filling: float, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["branch"] == branch
        and np.isclose(float(row["temperature"]), temperature, rtol=0.0, atol=1e-12)
        and np.isclose(float(row["target_filling"]), filling, rtol=0.0, atol=1e-12)
    ]
    interactions = np.array(sorted({float(row["interaction"]) for row in rows}))
    disorders = np.array(sorted({float(row["disorder_full_width"]) for row in rows}))
    values = np.full((interactions.size, disorders.size), np.nan)
    u_index = {value: index for index, value in enumerate(interactions)}
    d_index = {value: index for index, value in enumerate(disorders)}
    for row in selected:
        value = float(row[field])
        if np.isfinite(value) and value > 0.0:
            values[u_index[float(row["interaction"])], d_index[float(row["disorder_full_width"])]] = value
    return disorders, interactions, values


def _draw_map(axis, x, y, values, title, label):
    from matplotlib.colors import LogNorm

    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        axis.text(0.5, 0.5, "no positive data", ha="center", va="center", transform=axis.transAxes)
        return None
    image = axis.pcolormesh(
        x,
        y,
        np.ma.masked_invalid(values),
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
    )
    axis.set_title(title)
    axis.set_xlabel(r"disorder $\Delta/W$")
    axis.set_ylabel(r"interaction $U/W$")
    return image, label


def plot_summary(summary_path: str | Path, output_path: str | Path | None = None) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the optional matplotlib dependency") from exc
    summary_path = Path(summary_path)
    rows = _read_rows(summary_path)
    output = Path(output_path) if output_path else summary_path.with_name("transport_summary.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    quantities = [("sigma", r"$\sigma$"), ("kappa_e", r"$\kappa_e$"), ("lorenz_over_L0", "Lorenz / L0"), ("iterations", "iterations")]
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["branch"], row["interaction"], row["temperature"], row["target_filling"])
        groups.setdefault(key, []).append(row)
    for axis, (field, label) in zip(axes.flat, quantities):
        for key, group in groups.items():
            group.sort(key=lambda item: float(item["disorder_full_width"]))
            axis.scatter(
                [float(item["disorder_full_width"]) for item in group],
                [float(item[field]) for item in group],
                marker="o",
                label=f"{key[0]}, U={key[1]}, T={key[2]}, n={key[3]}",
            )
        axis.set_xlabel(r"disorder full width $\Delta$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_transport_heatmaps(
    summary_path: str | Path,
    output_directory: str | Path | None = None,
    bandwidth: float = 1.0,
) -> list[Path]:
    """Plot non-interpolated U-Delta maps for sigma and electronic kappa."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the optional matplotlib dependency") from exc

    summary_path = Path(summary_path)
    rows = _read_rows(summary_path)
    output_directory = Path(output_directory) if output_directory else summary_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    temperatures = sorted({float(row["temperature"]) for row in rows})
    fillings = sorted({float(row["target_filling"]) for row in rows})
    branches = ("arith", "typ")
    outputs: list[Path] = []

    # The stored coordinates are dimensional. Express them in the W=2D convention
    # used in Byczuk et al.; for the standard configuration W=1.
    normalized_rows = [dict(row) for row in rows]
    for row in normalized_rows:
        row["interaction"] = str(float(row["interaction"]) / bandwidth)
        row["disorder_full_width"] = str(float(row["disorder_full_width"]) / bandwidth)

    first_temperature = temperatures[0]
    for filling in fillings:
        filling_tag = f"{filling:.6g}".replace(".", "p")
        if "sigma_T0" in rows[0]:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharex=True, sharey=True)
            for axis, branch in zip(axes, branches):
                x, y, values = _grid(normalized_rows, branch, first_temperature, filling, "sigma_T0")
                drawn = _draw_map(
                    axis,
                    x,
                    y,
                    values,
                    rf"{branch}: $\sigma(T\to0)$, $n_c={filling:g}$",
                    r"$\sigma_0$",
                )
                if drawn:
                    image, label = drawn
                    fig.colorbar(image, ax=axis, label=label)
            output = output_directory / f"transport_heatmap_n_{filling_tag}_sigma_T0.png"
            fig.savefig(output, dpi=200)
            plt.close(fig)
            outputs.append(output)

        for temperature in temperatures:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, sharex=True, sharey=True)
            for column, branch in enumerate(branches):
                for row_index, (field, symbol) in enumerate((("sigma", r"$\sigma$"), ("kappa_e", r"$\kappa_e$"))):
                    axis = axes[row_index, column]
                    x, y, values = _grid(normalized_rows, branch, temperature, filling, field)
                    drawn = _draw_map(
                        axis,
                        x,
                        y,
                        values,
                        f"{branch}, n_c={filling:g}, T/W={temperature / bandwidth:g}",
                        symbol,
                    )
                    if drawn:
                        image, label = drawn
                        fig.colorbar(image, ax=axis, label=label)
            temperature_tag = f"{temperature / bandwidth:.6g}".replace(".", "p")
            output = output_directory / f"transport_heatmaps_n_{filling_tag}_T_{temperature_tag}.png"
            fig.savefig(output, dpi=200)
            plt.close(fig)
            outputs.append(output)
    return outputs
