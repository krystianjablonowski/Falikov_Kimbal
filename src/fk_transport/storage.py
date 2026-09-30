from __future__ import annotations

import json
from pathlib import Path

from .config import config_hash
from .sweep import build_tasks, output_root


REDUNDANT_POINT_FILES = (
    "spectral.csv.gz",
    "transport.csv.gz",
    "convergence.csv",
    "config.resolved.json",
    "checkpoint.npz",
)


def compact_results(cfg: dict) -> dict[str, int | str]:
    """Remove redundant files only from verified successful point directories.

    ``solution.npz`` is retained for spectral reanalysis. ``observables.json``
    and ``metadata.json`` are retained for merge/status operations.
    """
    root = output_root(cfg)
    expected_hash = config_hash(cfg)
    removed_files = 0
    freed_bytes = 0
    successful_points = 0
    skipped_points = 0
    for task in build_tasks(cfg):
        directory = root / "points" / f"point_{task['index']:06d}"
        metadata_path = directory / "metadata.json"
        solution_path = directory / "solution.npz"
        observables_path = directory / "observables.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped_points += 1
            continue
        if (
            metadata.get("status") != "success"
            or metadata.get("config_hash") != expected_hash
            or not solution_path.is_file()
            or not observables_path.is_file()
        ):
            skipped_points += 1
            continue
        successful_points += 1
        for name in REDUNDANT_POINT_FILES:
            path = directory / name
            if not path.is_file():
                continue
            size = path.stat().st_size
            path.unlink()
            removed_files += 1
            freed_bytes += size
    # Figures are now generated exclusively as PDFs. Remove legacy raster
    # duplicates only from this run's top-level output directory.
    for path in root.glob("*.png"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        path.unlink()
        removed_files += 1
        freed_bytes += size
    return {
        "root": str(root),
        "verified_successful_points": successful_points,
        "skipped_points": skipped_points,
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
    }
