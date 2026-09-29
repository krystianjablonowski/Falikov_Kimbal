from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


KEYS = ("branch", "interaction", "disorder_full_width", "target_filling", "temperature")


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def combine_summaries(coarse, refined, output):
    rows = _read(coarse) + _read(refined)  # refined rows deliberately win
    unique = {}
    for row in rows:
        key = tuple(row[name] if name == "branch" else round(float(row[name]), 12) for name in KEYS)
        unique[key] = row
    rows = list(unique.values())
    fields = sorted({name for row in rows for name in row})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, output


def _crossings(rows, field, threshold, temperature, filling, bandwidth):
    chosen = [r for r in rows if r["branch"] == "typ"
              and np.isclose(float(r["temperature"]), temperature)
              and np.isclose(float(r["target_filling"]), filling)]
    result = []
    for u in sorted({float(r["interaction"]) for r in chosen}):
        group = sorted((r for r in chosen if np.isclose(float(r["interaction"]), u)),
                       key=lambda r: float(r["disorder_full_width"]))
        for left, right in zip(group, group[1:]):
            x0, x1 = float(left["disorder_full_width"]), float(right["disorder_full_width"])
            y0, y1 = float(left[field]), float(right[field])
            if y0 <= 0 or y1 <= 0 or (y0-threshold)*(y1-threshold) > 0 or y0 == y1:
                continue
            fraction = (np.log10(threshold)-np.log10(y0))/(np.log10(y1)-np.log10(y0))
            result.append((u/bandwidth, (x0 + fraction*(x1-x0))/bandwidth))
    return result


def analyze_boundaries(coarse, refined, output_directory, thresholds=(1e-4, 1e-6, 1e-8), bandwidth=1.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise RuntimeError("boundary analysis requires matplotlib") from exc
    out = Path(output_directory)
    rows, combined = combine_summaries(coarse, refined, out / "combined_summary.csv")
    temperatures = sorted({float(r["temperature"]) for r in rows})
    fillings = sorted({float(r["target_filling"]) for r in rows})
    crossing_rows = []
    outputs = [combined]
    labels = {"sigma": r"$\sigma$", "kappa_e": r"$\kappa_e$"}
    for filling in fillings:
        for temperature in temperatures:
            tag = f"n_{filling:.6g}_T_{temperature/bandwidth:.6g}".replace(".", "p")
            fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
            for row_index, field in enumerate(("sigma", "kappa_e")):
                for column, branch in enumerate(("arith", "typ")):
                    selected = [r for r in rows if r["branch"] == branch
                                and np.isclose(float(r["temperature"]), temperature)
                                and np.isclose(float(r["target_filling"]), filling)
                                and float(r[field]) > 0]
                    x = np.array([float(r["disorder_full_width"])/bandwidth for r in selected])
                    y = np.array([float(r["interaction"])/bandwidth for r in selected])
                    z = np.log10([float(r[field]) for r in selected])
                    triangulation = mtri.Triangulation(x, y)
                    image = axes[row_index, column].tricontourf(triangulation, z, levels=24, cmap="inferno")
                    fig.colorbar(image, ax=axes[row_index, column], label=rf"$\log_{{10}}$ {labels[field]}")
                    axes[row_index, column].set_title(f"{branch}, T/W={temperature/bandwidth:g}")
                    axes[row_index, column].set_xlabel(r"$\Delta/W$")
                    axes[row_index, column].set_ylabel(r"$U/W$")
            fig.tight_layout()
            map_path = out / f"adaptive_boundary_maps_{tag}.png"
            fig.savefig(map_path, dpi=300); fig.savefig(map_path.with_suffix(".pdf")); plt.close(fig)
            outputs.append(map_path)

            fig, axis = plt.subplots(figsize=(6.2, 4.2))
            colors = {"sigma": "tab:blue", "kappa_e": "tab:orange"}
            styles = ["-", "--", ":"]
            for field in ("sigma", "kappa_e"):
                for number, threshold in enumerate(thresholds):
                    points = _crossings(rows, field, threshold, temperature, filling, bandwidth)
                    if points:
                        axis.scatter([p[1] for p in points], [p[0] for p in points], s=6,
                                     color=colors[field], marker="o" if field == "sigma" else "s",
                                     label=rf"{labels[field]}, $10^{{{int(np.log10(threshold))}}}$" if number < 3 else None)
                    crossing_rows.extend({"temperature": temperature, "target_filling": filling,
                                          "field": field, "threshold": threshold,
                                          "interaction_over_W": u, "disorder_over_W": d}
                                         for u, d in points)
            axis.set_xlabel(r"crossing $\Delta/W$"); axis.set_ylabel(r"$U/W$")
            axis.legend(frameon=False, ncol=2, fontsize=8); fig.tight_layout()
            path = out / f"boundary_crossings_{tag}.png"
            fig.savefig(path, dpi=300); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
            outputs.append(path)
    crossing_path = out / "boundary_crossings.csv"
    with crossing_path.open("w", newline="", encoding="utf-8") as handle:
        names = ["temperature", "target_filling", "field", "threshold", "interaction_over_W", "disorder_over_W"]
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader(); writer.writerows(crossing_rows)
    outputs.append(crossing_path)
    return outputs
