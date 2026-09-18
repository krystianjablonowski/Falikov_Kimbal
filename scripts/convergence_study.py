from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from fk_transport.config import load_config
from fk_transport.observables import transport_observables
from fk_transport.solver import solve_medium
from fk_transport.transport import transport_function


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--U", type=float, required=True)
    parser.add_argument("--disorder", type=float, required=True)
    parser.add_argument("--branch", choices=["arith", "typ"], required=True)
    parser.add_argument("--section", choices=["grid", "numerics"], required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--output", default="convergence.csv")
    args = parser.parse_args()
    base = load_config(args.config)
    rows = []
    for raw_value in args.values:
        cfg = copy.deepcopy(base)
        old = cfg[args.section].get(args.parameter)
        value = int(raw_value) if isinstance(old, int) and not isinstance(old, bool) else float(raw_value)
        cfg[args.section][args.parameter] = value
        result = solve_medium(cfg, args.U, args.disorder, args.branch, args.U / 2.0)
        tau = transport_function(
            result.omega,
            result.self_energy,
            result.chemical_potential,
            cfg["model"]["half_bandwidth"],
            cfg["numerics"]["broadening"],
            cfg["numerics"]["n_band_quadrature"],
        )
        for temperature in cfg["sweep"]["temperatures"]:
            obs = transport_observables(result.omega, tau, temperature)
            rows.append(
                {
                    "section": args.section,
                    "parameter": args.parameter,
                    "value": value,
                    "temperature": temperature,
                    "solver_status": result.status,
                    "rho_zero": result.metrics["rho_zero"],
                    "sigma": obs["sigma"],
                    "kappa_e": obs["kappa_e"],
                    "lorenz": obs["lorenz"],
                    "max_imag_sigma": result.metrics["max_imag_sigma_relevant"],
                    "iterations": result.iterations,
                }
            )
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
