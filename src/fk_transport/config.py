from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "model": {"half_bandwidth": 0.5, "hopping": 0.25, "w1": 0.5},
    "grid": {"omega_max": 4.0, "n_omega": 2001},
    "numerics": {
        "broadening": 1.0e-3,
        "n_disorder_quadrature": 48,
        "n_band_quadrature": 160,
        "mixing": 0.5,
        "tolerance": 1.0e-8,
        "max_iterations": 500,
        "rho_floor": 1.0e-300,
        "hilbert_padding_factor": 8,
        "checkpoint_every": 25,
        "causality_tolerance": 1.0e-7,
        "sum_rule_tolerance": 2.0e-2,
        "l11_floor": 1.0e-14,
        "moment_tolerance": 1.0e-12,
    },
    "sweep": {
        "interactions": [0.0],
        "disorder_full_widths": [0.0],
        "branches": ["arith", "typ"],
        "temperatures": [0.01, 0.02],
        "target_fillings": [0.5],
    },
    "filling": {
        "mu_min": -3.0,
        "mu_max": 3.0,
        "tolerance": 1.0e-6,
        "max_iterations": 50,
    },
    "output": {"directory": "results"},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configuration requires PyYAML; JSON works without it") from exc
        raw = yaml.safe_load(text)
    else:
        raise ValueError("configuration must be JSON, YAML, or YML")
    cfg = _merge(DEFAULTS, raw or {})
    validate_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    model, grid, num, sweep = cfg["model"], cfg["grid"], cfg["numerics"], cfg["sweep"]
    d, t = float(model["half_bandwidth"]), float(model["hopping"])
    if d <= 0 or t <= 0 or not abs(d - 2.0 * t) <= 1.0e-12 * max(1.0, d):
        raise ValueError("model must satisfy half_bandwidth == 2*hopping > 0")
    if not (0.0 <= float(model["w1"]) <= 1.0):
        raise ValueError("w1 must lie in [0, 1]")
    if int(grid["n_omega"]) < 5 or int(grid["n_omega"]) % 2 != 1:
        raise ValueError("n_omega must be odd and at least 5")
    if float(grid["omega_max"]) <= 0:
        raise ValueError("omega_max must be positive")
    if float(num["broadening"]) <= 0 or not 0 < float(num["mixing"]) <= 1:
        raise ValueError("broadening must be positive and mixing must lie in (0, 1]")
    if int(num["n_disorder_quadrature"]) < 1 or int(num["n_band_quadrature"]) < 2:
        raise ValueError("quadrature orders are too small")
    if any(branch not in {"arith", "typ"} for branch in sweep["branches"]):
        raise ValueError("branches may contain only 'arith' and 'typ'")
    if any(float(x) < 0 for x in sweep["disorder_full_widths"]):
        raise ValueError("disorder_full_widths must be non-negative")
    if any(float(x) <= 0 for x in sweep["temperatures"]):
        raise ValueError("temperatures must be positive")


def canonical_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
