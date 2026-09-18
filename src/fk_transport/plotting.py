from __future__ import annotations

import csv
from pathlib import Path


def plot_summary(summary_path: str | Path, output_path: str | Path | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the optional matplotlib dependency") from exc
    summary_path = Path(summary_path)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary contains no successful points")
    output = Path(output_path) if output_path else summary_path.with_name("transport_summary.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    quantities = [("sigma", r"$\sigma$"), ("kappa_e", r"$\kappa_e$"), ("lorenz_over_L0", r"$\mathcal L/\mathcal L_0$"), ("iterations", "iterations")]
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["branch"], row["interaction"], row["temperature"])
        groups.setdefault(key, []).append(row)
    for axis, (field, label) in zip(axes.flat, quantities):
        for key, group in groups.items():
            group.sort(key=lambda item: float(item["disorder_full_width"]))
            axis.scatter(
                [float(item["disorder_full_width"]) for item in group],
                [float(item[field]) for item in group],
                marker="o",
                label=f"{key[0]}, U={key[1]}, T={key[2]}",
            )
        axis.set_xlabel(r"disorder full width $\Delta$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
