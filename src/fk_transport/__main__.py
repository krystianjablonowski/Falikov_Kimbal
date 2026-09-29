from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .boundary_refinement import DEFAULT_FIELDS, generate_boundary_config
from .io import atomic_json
from .plotting import plot_all_temperature_summaries, plot_transport_heatmaps
from .sweep import build_tasks, merge_results, output_root, run_task, scan_status, write_manifest
from .temperature_analysis import (
    fit_activation_groups,
    fit_temperature_scan,
    plot_temperature_scan,
    reweight_point,
)
from .validation import run_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fk_transport")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "manifest", "sweep", "status", "merge", "plot"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "sweep":
            command.add_argument("--index", type=int)
    solve = sub.add_parser("solve")
    solve.add_argument("--config", required=True)
    solve.add_argument("--U", required=True, type=float)
    solve.add_argument("--disorder", required=True, type=float)
    solve.add_argument("--branch", required=True, choices=["arith", "typ"])
    reweight = sub.add_parser("reweight")
    reweight.add_argument("--point", required=True)
    reweight.add_argument("--temperatures", required=True, nargs="+", type=float)
    reweight.add_argument("--output")
    reweight.add_argument("--plot", action="store_true")
    activation = sub.add_parser("activation-fit")
    activation.add_argument("--input", required=True)
    activation.add_argument("--fields", nargs="+", default=["sigma", "kappa_e"])
    activation.add_argument("--temperature-min", type=float)
    activation.add_argument("--temperature-max", type=float)
    activation.add_argument("--output")
    activation_map = sub.add_parser("activation-map")
    activation_map.add_argument("--input", required=True)
    activation_map.add_argument("--fields", nargs="+", default=["sigma", "kappa_e"])
    activation_map.add_argument(
        "--group-by",
        nargs="+",
        default=["branch", "interaction", "disorder_full_width", "target_filling"],
    )
    activation_map.add_argument("--temperature-min", type=float)
    activation_map.add_argument("--temperature-max", type=float)
    activation_map.add_argument("--output")
    refine = sub.add_parser("refine-boundaries")
    refine.add_argument("--config", required=True, help="coarse-stage configuration")
    refine.add_argument("--input", required=True, help="coarse-stage summary.csv")
    refine.add_argument("--output-config", required=True)
    refine.add_argument("--output-directory", default="results/stage2_boundaries")
    refine.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    refine.add_argument("--branch", choices=["arith", "typ"], default="typ")
    refine.add_argument("--temperature", type=float)
    refine.add_argument("--filling", type=float, default=0.5)
    refine.add_argument("--u-step", type=float, default=0.025, help="step in U/W")
    refine.add_argument("--disorder-step", type=float, default=0.025, help="step in Delta/W")
    refine.add_argument("--log-jump", type=float, default=0.75, help="minimum cell range in decades")
    refine.add_argument("--value-floor", type=float, default=1.0e-14)
    refine.add_argument("--padding-cells", type=int, default=1)
    refine.add_argument("--batch-size", type=int, default=8)
    refine.add_argument("--max-jobs", type=int, default=399)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "reweight":
        scan = reweight_point(args.point, args.temperatures, args.output)
        result = {"temperature_scan": str(scan)}
        if args.plot:
            result["plot"] = str(plot_temperature_scan(scan))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "activation-fit":
        output = fit_temperature_scan(
            args.input,
            args.fields,
            args.temperature_min,
            args.temperature_max,
            args.output,
        )
        print(json.dumps({"activation_fits": str(output)}, indent=2))
        return 0
    if args.command == "activation-map":
        output = fit_activation_groups(
            args.input,
            args.fields,
            args.group_by,
            args.temperature_min,
            args.temperature_max,
            args.output,
        )
        print(json.dumps({"activation_summary": str(output)}, indent=2))
        return 0

    if args.command == "refine-boundaries":
        cfg = load_config(args.config)
        report = generate_boundary_config(
            cfg,
            args.input,
            args.output_config,
            fields=args.fields,
            branch=args.branch,
            temperature=args.temperature,
            filling=args.filling,
            u_step_ratio=args.u_step,
            disorder_step_ratio=args.disorder_step,
            log_jump=args.log_jump,
            value_floor=args.value_floor,
            padding_cells=args.padding_cells,
            output_directory=args.output_directory,
            batch_size=args.batch_size,
            max_jobs=args.max_jobs,
        )
        print(json.dumps(report, indent=2))
        return 0

    cfg = load_config(args.config)
    if args.command == "validate":
        report = run_validation(cfg)
        destination = output_root(cfg) / "validation_report.json"
        atomic_json(destination, report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "manifest":
        path = write_manifest(cfg)
        tasks = build_tasks(cfg)
        print(json.dumps({"manifest": str(path), "tasks": len(tasks), "array_max": len(tasks) - 1}))
        return 0
    if args.command == "solve":
        original = cfg["sweep"]
        cfg["sweep"] = {
            **original,
            "interactions": [args.U],
            "disorder_full_widths": [args.disorder],
            "branches": [args.branch],
        }
        print(json.dumps(run_task(cfg, 0), indent=2))
        return 0
    if args.command == "sweep":
        write_manifest(cfg)
        if args.index is None:
            for task in build_tasks(cfg):
                print(json.dumps(run_task(cfg, task["index"])))
        else:
            print(json.dumps(run_task(cfg, args.index), indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(scan_status(cfg), indent=2))
        return 0
    if args.command == "merge":
        print(merge_results(cfg))
        return 0
    if args.command == "plot":
        summary = output_root(cfg) / "summary.csv"
        bandwidth = 2.0 * float(cfg["model"]["half_bandwidth"])
        outputs = plot_all_temperature_summaries(summary, bandwidth=bandwidth)
        outputs.extend(
            plot_transport_heatmaps(
                summary,
                bandwidth=bandwidth,
            )
        )
        print(json.dumps([str(path) for path in outputs], indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
