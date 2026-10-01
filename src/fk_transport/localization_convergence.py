from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import canonical_config, validate_config
from .plotting import _publication_style, _save_publication_figure
from .sweep import build_tasks


SETTINGS = (
    (5.0e-4, 20001, 96, "eta_5e-4_n20001_q96"),
    (2.5e-4, 40001, 96, "eta_2p5e-4_n40001_q96"),
    (1.25e-4, 80001, 96, "eta_1p25e-4_n80001_q96"),
    (5.0e-4, 20001, 192, "eta_5e-4_n20001_q192"),
)


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _close(value: float, target: float) -> bool:
    return math.isclose(value, target, rel_tol=0.0, abs_tol=1.0e-10)


def select_localization_points(
    diagnostics_path: str | Path,
    filling: float,
    temperature: float,
    interactions: list[float],
    ratio_target: float = 1.0e-2,
    neighbors: int = 3,
) -> list[tuple[float, float]]:
    if ratio_target <= 0.0 or neighbors < 0:
        raise ValueError("ratio_target must be positive and neighbors non-negative")
    rows = [
        row for row in _read(diagnostics_path)
        if row.get("branch") == "typ"
        and _close(float(row["target_filling"]), filling)
        and _close(float(row["temperature"]), temperature)
    ]
    selected: set[tuple[float, float]] = set()
    for requested_u in interactions:
        candidates = [row for row in rows if _close(float(row["interaction"]), requested_u)]
        if not candidates:
            available = sorted({float(row["interaction"]) for row in rows})
            raise ValueError(f"interaction {requested_u:g} is unavailable; available values: {available}")
        candidates.sort(key=lambda row: float(row["disorder_full_width"]))
        score = []
        for row in candidates:
            ratio = float(row["rho_typ_over_arith_zero"])
            score.append(
                abs(math.log10(ratio / ratio_target))
                if math.isfinite(ratio) and ratio > 0.0 else float("inf")
            )
        anchor = int(np.argmin(score))
        for index in range(max(0, anchor - neighbors), min(len(candidates), anchor + neighbors + 1)):
            selected.add((requested_u, float(candidates[index]["disorder_full_width"])))
    if not selected:
        raise ValueError("no localization-convergence points were selected")
    return sorted(selected)


def prepare_localization_convergence(
    base_config: dict,
    diagnostics_path: str | Path,
    output_prefix: str | Path,
    filling: float = 0.4,
    temperature: float = 0.02,
    interactions: list[float] | None = None,
    ratio_target: float = 1.0e-2,
    neighbors: int = 3,
) -> dict:
    interactions = interactions or [0.75, 1.5, 2.25, 3.0]
    points = select_localization_points(
        diagnostics_path, filling, temperature, interactions, ratio_target, neighbors
    )
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for eta, n_omega, quadrature, label in SETTINGS:
        cfg = copy.deepcopy(canonical_config(base_config))
        cfg["numerics"]["broadening"] = eta
        cfg["numerics"]["n_disorder_quadrature"] = quadrature
        cfg["grid"]["n_omega"] = n_omega
        cfg["sweep"]["branches"] = ["typ"]
        cfg["sweep"]["temperatures"] = [temperature]
        cfg["sweep"]["target_fillings"] = [filling]
        cfg["sweep"]["parameter_points"] = [
            {"interaction": interaction, "disorder_full_width": disorder}
            for interaction, disorder in points
        ]
        cfg["output"]["directory"] = f"results/stage9_localization_convergence/{label}"
        cfg["localization_convergence"] = {
            "source_diagnostics": str(diagnostics_path),
            "filling": filling,
            "temperature": temperature,
            "interactions": interactions,
            "selection_ratio": ratio_target,
            "neighbors": neighbors,
            "eta": eta,
            "n_omega": n_omega,
            "n_disorder_quadrature": quadrature,
        }
        validate_config(cfg)
        path = prefix.with_name(f"{prefix.name}_{label}.json")
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        reports.append({
            "label": label, "config": str(path), "eta": eta,
            "n_omega": n_omega, "n_disorder_quadrature": quadrature,
            "parameter_points": len(points), "spectral_tasks": len(build_tasks(cfg)),
        })
    return {"selected_parameter_points": len(points), "points": points, "configs": reports}


def analyze_localization_convergence(
    summary_paths: list[str | Path],
    config_paths: list[str | Path],
    output_directory: str | Path,
) -> list[Path]:
    if len(summary_paths) != len(config_paths) or len(summary_paths) < 2:
        raise ValueError("provide equal lists of at least two summaries and configs")
    configs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in config_paths]
    scans = [cfg for cfg in configs if "localization_convergence" in cfg]
    if not scans:
        raise ValueError("at least one config must come from prepare-localization-convergence")
    filling = float(scans[0]["localization_convergence"]["filling"])
    temperature = float(scans[0]["localization_convergence"]["temperature"])
    point_sets = [
        {
            (float(point["interaction"]), float(point["disorder_full_width"]))
            for point in cfg["sweep"]["parameter_points"]
        }
        for cfg in scans
    ]
    selected_points = set.intersection(*point_sets)
    if not selected_points:
        raise ValueError("localization-convergence configs have no common parameter points")
    long_rows: list[dict] = []
    for summary_path, config_path, cfg in zip(summary_paths, config_paths, configs):
        eta = float(cfg["numerics"]["broadening"])
        n_omega = int(cfg["grid"]["n_omega"])
        quadrature = int(cfg["numerics"]["n_disorder_quadrature"])
        label = Path(config_path).stem
        for row in _read(summary_path):
            point = (float(row["interaction"]), float(row["disorder_full_width"]))
            if (
                row["branch"] != "typ"
                or not _close(float(row["target_filling"]), filling)
                or not _close(float(row["temperature"]), temperature)
                or point not in selected_points
            ):
                continue
            rho_a, rho_t = float(row["rho_arith_zero"]), float(row["rho_typ_zero"])
            ratio = rho_t / rho_a if rho_a > 0.0 else float("nan")
            long_rows.append({
                "label": label, "eta": eta, "n_omega": n_omega,
                "n_disorder_quadrature": quadrature,
                "target_filling": float(row["target_filling"]),
                "temperature": float(row["temperature"]),
                "interaction": float(row["interaction"]),
                "disorder_full_width": float(row["disorder_full_width"]),
                "rho_arith_zero": rho_a, "rho_typ_zero": rho_t,
                "rho_typ_over_arith_zero": ratio,
                "sigma": float(row["sigma"]),
                "chemical_potential": float(row["chemical_potential"]),
                "final_residual": float(row["final_residual"]),
                "filling_error": float(row["filling_error"]),
                "iterations": int(float(row["iterations"])),
                "status": row["status"],
            })
    if not long_rows:
        raise ValueError("none of the requested convergence points are present in the summaries")
    fields = list(long_rows[0]) if long_rows else ["label"]
    output = Path(output_directory)
    products = [_write(output / "localization_convergence_points.csv", long_rows, fields)]

    grouped: dict[tuple[float, float, float, float], list[dict]] = defaultdict(list)
    for row in long_rows:
        grouped[(row["target_filling"], row["temperature"], row["interaction"],
                 row["disorder_full_width"])].append(row)
    spread_rows = []
    for key, rows in sorted(grouped.items()):
        ratios = np.asarray([row["rho_typ_over_arith_zero"] for row in rows], dtype=float)
        sigmas = np.asarray([row["sigma"] for row in rows], dtype=float)
        mus = np.asarray([row["chemical_potential"] for row in rows], dtype=float)
        positive_ratio = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        positive_sigma = sigmas[np.isfinite(sigmas) & (sigmas > 0.0)]
        spread_rows.append({
            "target_filling": key[0], "temperature": key[1], "interaction": key[2],
            "disorder_full_width": key[3], "settings_present": len(rows),
            "settings_expected": len(summary_paths),
            "complete_comparison": len(rows) == len(summary_paths),
            "ratio_log10_spread": (
                float(np.ptp(np.log10(positive_ratio))) if positive_ratio.size > 1 else float("nan")
            ),
            "sigma_log10_spread": (
                float(np.ptp(np.log10(positive_sigma))) if positive_sigma.size > 1 else float("nan")
            ),
            "chemical_potential_spread": float(np.ptp(mus)) if mus.size > 1 else float("nan"),
            "maximum_abs_filling_error": max(abs(row["filling_error"]) for row in rows),
            "maximum_final_residual": max(row["final_residual"] for row in rows),
            "all_success": all(row["status"] == "success" for row in rows),
        })
    spread_fields = list(spread_rows[0]) if spread_rows else ["target_filling"]
    products.append(_write(output / "localization_convergence_spread.csv", spread_rows, spread_fields))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interactions = sorted({row["interaction"] for row in long_rows})
    with plt.rc_context(_publication_style()):
        for interaction in interactions:
            selected = [row for row in long_rows if np.isclose(row["interaction"], interaction)]
            fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.2), sharex=True)
            for label in sorted({row["label"] for row in selected}):
                curve = sorted((row for row in selected if row["label"] == label),
                               key=lambda row: row["disorder_full_width"])
                x = [row["disorder_full_width"] for row in curve]
                axes[0].plot(x, [row["rho_typ_over_arith_zero"] for row in curve], "o-",
                             markersize=2.5, linewidth=0.9, label=label.replace("stage9_localization_convergence_", ""))
                axes[1].plot(x, [row["sigma"] for row in curve], "o-", markersize=2.5, linewidth=0.9)
            axes[0].set_yscale("log")
            axes[1].set_yscale("log")
            axes[0].axhline(1.0e-1, color="0.5", linestyle=":", linewidth=0.7)
            axes[0].axhline(1.0e-2, color="0.7", linestyle=":", linewidth=0.7)
            axes[0].set_ylabel(r"$\rho_{typ}(0)/\rho_{arith}(0)$")
            axes[1].set_ylabel(r"$\sigma_{typ}$")
            axes[1].set_xlabel(r"disorder $\Delta/W$")
            axes[0].set_title(rf"$U/W={interaction:g}$")
            fig.suptitle(rf"$n_c={filling:g},\ T/W={temperature:g}$", y=0.995)
            axes[0].legend(frameon=False, fontsize=5.5)
            fig.tight_layout()
            tag = f"U_{interaction:.6g}".replace(".", "p")
            products.append(_save_publication_figure(fig, output / f"localization_convergence_{tag}.pdf"))
            plt.close(fig)
    return products
