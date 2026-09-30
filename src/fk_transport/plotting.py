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
    rows: list[dict[str, str]],
    branch: str,
    temperature: float,
    filling: float,
    field: str,
    positive_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["branch"] == branch
        and np.isclose(float(row["temperature"]), temperature, rtol=0.0, atol=1e-12)
        and np.isclose(float(row["target_filling"]), filling, rtol=0.0, atol=1e-12)
    ]
    # Different fillings or calculation stages may use different parameter grids.
    # Building axes from all rows inserts artificial missing cells between adjacent
    # points of the selected dataset and can hide genuine zero crossings.
    interactions = np.array(sorted({float(row["interaction"]) for row in selected}))
    disorders = np.array(sorted({float(row["disorder_full_width"]) for row in selected}))
    values = np.full((interactions.size, disorders.size), np.nan)
    u_index = {value: index for index, value in enumerate(interactions)}
    d_index = {value: index for index, value in enumerate(disorders)}
    for row in selected:
        value = float(row[field])
        if np.isfinite(value) and (value > 0.0 or not positive_only):
            values[u_index[float(row["interaction"])], d_index[float(row["disorder_full_width"])]] = value
    return disorders, interactions, values


def _draw_map(
    axis,
    x,
    y,
    values,
    title,
    norm,
    cmap="inferno",
):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        axis.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=axis.transAxes)
        return None
    image = axis.pcolormesh(
        x,
        y,
        np.ma.masked_invalid(values),
        shading="nearest",
        cmap=cmap,
        norm=norm,
    )
    axis.set_title(title)
    axis.set_xlabel(r"disorder $\Delta/W$")
    axis.set_ylabel(r"interaction $U/W$")
    return image


def _log_norm(arrays):
    from matplotlib.colors import LogNorm

    positive = np.concatenate(
        [array[np.isfinite(array) & (array > 0.0)] for array in arrays]
    )
    if positive.size == 0:
        return None
    lower, upper = float(positive.min()), float(positive.max())
    if lower == upper:
        upper = lower * (1.0 + 1.0e-12)
    return LogNorm(vmin=lower, vmax=upper)


def _signed_norm(arrays):
    from matplotlib.colors import Normalize

    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if finite.size == 0:
        return None
    limit = float(np.max(np.abs(finite)))
    if limit == 0.0:
        limit = 1.0e-12
    return Normalize(vmin=-limit, vmax=limit)


def _signed_colormap():
    """Black-centred map with visually distinct negative and positive arms."""
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list(
        "transport_signed",
        [
            (0.00, "#eff51c"),
            (0.16, "#4dbd83"),
            (0.32, "#355f9f"),
            (0.46, "#24134f"),
            (0.50, "#000000"),
            (0.54, "#3a0707"),
            (0.68, "#9e0b22"),
            (0.84, "#ef2b1d"),
            (1.00, "#ffd84a"),
        ],
    )
    colormap.set_bad("white")
    return colormap


def _publication_style() -> dict:
    return {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
    }


def _finish_axis(axis, panel: str) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.text(0.03, 0.93, panel, transform=axis.transAxes, fontweight="bold", va="top")


def _save_publication_figure(fig, output: Path) -> None:
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))


def plot_summary(
    summary_path: str | Path,
    output_path: str | Path | None = None,
    bandwidth: float = 1.0,
    filling: float | None = None,
    temperature: float | None = None,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the optional matplotlib dependency") from exc
    summary_path = Path(summary_path)
    rows = _read_rows(summary_path)
    output = Path(output_path) if output_path else summary_path.with_name("transport_summary.png")
    filling = (
        sorted({float(row["target_filling"]) for row in rows})[0]
        if filling is None
        else float(filling)
    )
    temperature = (
        sorted({float(row["temperature"]) for row in rows})[0]
        if temperature is None
        else float(temperature)
    )
    available_u = np.array(sorted({float(row["interaction"]) for row in rows}))
    targets = np.linspace(float(available_u.min()), float(available_u.max()), min(5, available_u.size))
    selected_u = sorted({float(available_u[np.argmin(np.abs(available_u - target))]) for target in targets})
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(selected_u)))
    color_by_u = dict(zip(selected_u, colors))
    branch_style = {"arith": ("-", "o"), "typ": ("--", "s")}
    candidates = [
        ("sigma", r"$\sigma$", True),
        ("kappa_e", r"$\kappa_{\mathrm{e}}$", True),
        ("thermopower", r"$S$", False),
        ("c_v_electronic", r"$c_V^{\mathrm{el}}$", True),
        ("lorenz_over_L0", r"$L/L_0$", False),
        ("charge_diffusivity_proxy", r"$D_c$", True),
        ("thermal_diffusivity_proxy", r"$D_E$", True),
    ]
    quantities = [item for item in candidates if item[0] in rows[0]]
    columns = 3 if len(quantities) >= 5 else 2
    rows_count = int(np.ceil(len(quantities) / columns))
    with plt.rc_context(_publication_style()):
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(9.2 if columns == 3 else 7.0, 2.65 * rows_count),
            squeeze=False,
        )
        for panel, (axis, (field, label, logarithmic)) in enumerate(zip(axes.flat, quantities)):
            for interaction in selected_u:
                for branch, (line_style, marker) in branch_style.items():
                    group = [
                        row for row in rows
                        if row["branch"] == branch
                        and np.isclose(float(row["interaction"]), interaction)
                        and np.isclose(float(row["temperature"]), temperature)
                        and np.isclose(float(row["target_filling"]), filling)
                    ]
                    group.sort(key=lambda item: float(item["disorder_full_width"]))
                    x = np.array([float(item["disorder_full_width"]) / bandwidth for item in group])
                    y = np.array([float(item[field]) for item in group])
                    valid = np.isfinite(y) & ((y > 0.0) if logarithmic else True)
                    axis.plot(
                        x[valid], y[valid], linestyle=line_style, marker=marker,
                        color=color_by_u[interaction], markersize=2.7, linewidth=0.9,
                        markerfacecolor="none" if branch == "typ" else color_by_u[interaction],
                        markeredgewidth=0.7,
                    )
            if logarithmic:
                axis.set_yscale("log")
            if field == "lorenz_over_L0":
                axis.axhline(1.0, color="0.45", linewidth=0.7, linestyle=":")
            axis.set_xlabel(r"$\Delta/W$")
            axis.set_ylabel(label)
            _finish_axis(axis, f"({chr(97 + panel)})")
        for axis in axes.flat[len(quantities):]:
            axis.set_visible(False)

        from matplotlib.lines import Line2D
        u_handles = [
            Line2D([0], [0], color=color_by_u[value], linewidth=1.4, label=rf"$U/W={value / bandwidth:g}$")
            for value in selected_u
        ]
        branch_handles = [
            Line2D([0], [0], color="black", linestyle=style[0], marker=style[1],
                   markerfacecolor="none" if branch == "typ" else "black", markersize=4,
                   label=branch)
            for branch, style in branch_style.items()
        ]
        fig.legend(handles=u_handles + branch_handles, loc="upper center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(rf"$n_c={filling:g}$, $T/W={temperature / bandwidth:g}$", y=0.90, fontsize=9)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.78, wspace=0.38, hspace=0.36)
        _save_publication_figure(fig, output)
        plt.close(fig)
    return output


def plot_all_temperature_summaries(
    summary_path: str | Path,
    output_directory: str | Path | None = None,
    bandwidth: float = 1.0,
) -> list[Path]:
    """Create a complete line-plot summary for every filling and temperature."""
    summary_path = Path(summary_path)
    rows = _read_rows(summary_path)
    output_directory = Path(output_directory) if output_directory else summary_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    # Remove the obsolete single-temperature overview so it cannot be mistaken
    # for the complete set or republished alongside the temperature-tagged files.
    for legacy in (
        output_directory / "transport_summary.png",
        output_directory / "transport_summary.pdf",
    ):
        if legacy.is_file():
            legacy.unlink()
    outputs: list[Path] = []
    for filling in sorted({float(row["target_filling"]) for row in rows}):
        filling_tag = f"{filling:.6g}".replace(".", "p")
        for temperature in sorted({float(row["temperature"]) for row in rows}):
            temperature_tag = f"{temperature / bandwidth:.6g}".replace(".", "p")
            output = output_directory / (
                f"transport_summary_n_{filling_tag}_T_{temperature_tag}.png"
            )
            outputs.append(
                plot_summary(
                    summary_path,
                    output,
                    bandwidth,
                    filling=filling,
                    temperature=temperature,
                )
            )
    return outputs


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
            grids = [_grid(normalized_rows, branch, first_temperature, filling, "sigma_T0") for branch in branches]
            norm = _log_norm([grid[2] for grid in grids])
            with plt.rc_context(_publication_style()):
                fig = plt.figure(figsize=(7.4, 3.0))
                grid_spec = fig.add_gridspec(
                    1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.24
                )
                axes = np.asarray(
                    [
                        fig.add_subplot(grid_spec[0, 0]),
                        fig.add_subplot(grid_spec[0, 1]),
                    ]
                )
                axes[1].sharex(axes[0])
                axes[1].sharey(axes[0])
                color_axis = fig.add_subplot(grid_spec[0, 2])
                image = None
                for panel, (axis, branch, (x, y, values)) in enumerate(zip(axes, branches, grids)):
                    image = _draw_map(
                        axis, x, y, values,
                        rf"{branch}: $\sigma(T\to0)$, $n_c={filling:g}$", norm,
                    )
                    if panel == 1:
                        axis.set_ylabel("")
                        axis.tick_params(labelleft=False)
                    _finish_axis(axis, f"({chr(97 + panel)})")
                if image is not None:
                    fig.colorbar(image, cax=color_axis, label=r"$\sigma_0$")
                else:
                    color_axis.set_visible(False)
                fig.subplots_adjust(left=0.09, right=0.94, bottom=0.16, top=0.91)
                output = output_directory / f"transport_heatmap_n_{filling_tag}_sigma_T0.png"
                _save_publication_figure(fig, output)
                plt.close(fig)
                outputs.append(output)

        heatmap_groups = [
            (
                "transport",
                (("sigma", r"$\sigma$"), ("kappa_e", r"$\kappa_{\mathrm{e}}$")),
                "log",
            ),
            (
                "thermoelectric",
                (
                    ("thermopower", r"$S$"),
                    ("dmu_dT_fixed_density", r"$(\partial\mu/\partial T)_n$"),
                ),
                "signed",
            ),
            (
                "thermodynamic",
                (("c_v_electronic", r"$c_V^{\mathrm{el}}$"), ("K0_thermo", r"$\chi_c$")),
                "log",
            ),
            (
                "diffusivity",
                (
                    ("charge_diffusivity_proxy", r"$D_c$"),
                    ("thermal_diffusivity_proxy", r"$D_E$"),
                ),
                "log",
            ),
            (
                "diagnostic",
                (
                    ("lorenz_over_L0", r"$L/L_0$"),
                    ("transport_energy_variance", r"$\mathrm{Var}_{\rm tr}(\omega)$"),
                ),
                "log",
            ),
        ]
        available_groups = [
            (name, fields, scale)
            for name, fields, scale in heatmap_groups
            if all(field in rows[0] for field, _ in fields)
        ]
        for temperature in temperatures:
            temperature_tag = f"{temperature / bandwidth:.6g}".replace(".", "p")
            for group_name, fields, scale in available_groups:
                grids_by_field = {
                    field: [
                        _grid(
                            normalized_rows,
                            branch,
                            temperature,
                            filling,
                            field,
                            positive_only=(scale == "log"),
                        )
                        for branch in branches
                    ]
                    for field, _ in fields
                }
                norms = {
                    field: (
                        _log_norm([grid[2] for grid in grids])
                        if scale == "log"
                        else _signed_norm([grid[2] for grid in grids])
                    )
                    for field, grids in grids_by_field.items()
                }
                with plt.rc_context(_publication_style()):
                    fig = plt.figure(figsize=(7.8, 5.5))
                    grid_spec = fig.add_gridspec(
                        2,
                        3,
                        width_ratios=(1.0, 1.0, 0.045),
                        wspace=0.24,
                        hspace=0.28,
                    )
                    axes = np.empty((2, 2), dtype=object)
                    color_axes = []
                    for row_index in range(2):
                        for column in range(2):
                            axes[row_index, column] = fig.add_subplot(
                                grid_spec[row_index, column]
                            )
                        color_axes.append(fig.add_subplot(grid_spec[row_index, 2]))
                    for axis in axes.flat[1:]:
                        axis.sharex(axes[0, 0])
                        axis.sharey(axes[0, 0])
                    panel = 0
                    for row_index, (field, symbol) in enumerate(fields):
                        row_image = None
                        for column, (branch, (x, y, values)) in enumerate(
                            zip(branches, grids_by_field[field])
                        ):
                            axis = axes[row_index, column]
                            row_image = _draw_map(
                                axis,
                                x,
                                y,
                                values,
                                rf"{branch}, $n_c={filling:g}$, $T/W={temperature / bandwidth:g}$",
                                norms[field],
                                cmap="inferno" if scale == "log" else _signed_colormap(),
                            )
                            if column == 1:
                                axis.set_ylabel("")
                                axis.tick_params(labelleft=False)
                            _finish_axis(axis, f"({chr(97 + panel)})")
                            panel += 1
                        if row_image is not None:
                            fig.colorbar(row_image, cax=color_axes[row_index], label=symbol)
                        else:
                            color_axes[row_index].set_visible(False)
                    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.10, top=0.94)
                    output = output_directory / (
                        f"{group_name}_heatmaps_n_{filling_tag}_T_{temperature_tag}.png"
                    )
                    _save_publication_figure(fig, output)
                    plt.close(fig)
                    outputs.append(output)
    return outputs
