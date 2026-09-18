from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .io import atomic_json
from .plotting import plot_summary
from .sweep import build_tasks, merge_results, output_root, run_task, scan_status, write_manifest
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
        print(plot_summary(output_root(cfg) / "summary.csv"))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
