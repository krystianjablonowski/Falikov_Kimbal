from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fk_transport.config import config_hash
from fk_transport.storage import REDUNDANT_POINT_FILES, compact_results


class StorageTests(unittest.TestCase):
    def test_compaction_keeps_reanalysis_and_merge_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = {
                "_config_path": str(root / "configs" / "run.json"),
                "model": {"w1": 0.4},
                "sweep": {
                    "interactions": [1.0],
                    "disorder_full_widths": [1.5],
                    "branches": ["typ"],
                    "temperatures": [0.02],
                    "target_fillings": [0.45],
                },
                "output": {"directory": str(root / "results")},
            }
            point = root / "results" / "points" / "point_000000"
            point.mkdir(parents=True)
            (point / "solution.npz").write_bytes(b"solution")
            (point / "observables.json").write_text("[]", encoding="utf-8")
            (point / "metadata.json").write_text(
                json.dumps({"status": "success", "config_hash": config_hash(cfg)}),
                encoding="utf-8",
            )
            for name in REDUNDANT_POINT_FILES:
                (point / name).write_bytes(b"redundant")
            (root / "results" / "legacy_plot.png").write_bytes(b"raster")

            report = compact_results(cfg)

            self.assertEqual(report["verified_successful_points"], 1)
            self.assertEqual(report["removed_files"], len(REDUNDANT_POINT_FILES) + 1)
            self.assertTrue((point / "solution.npz").is_file())
            self.assertTrue((point / "observables.json").is_file())
            self.assertTrue((point / "metadata.json").is_file())
            self.assertTrue(all(not (point / name).exists() for name in REDUNDANT_POINT_FILES))
            self.assertFalse((root / "results" / "legacy_plot.png").exists())


if __name__ == "__main__":
    unittest.main()
