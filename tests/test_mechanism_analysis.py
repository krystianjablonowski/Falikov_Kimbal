import csv
from pathlib import Path

import numpy as np

from fk_transport.mechanism_analysis import (
    _reliability_summary,
    analyze_transport_mechanism,
    calculate_mechanism_diagnostics,
)


def test_mechanism_moments_and_branch_ratios(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    points = tmp_path / "points"
    fields = [
        "index", "branch", "target_filling", "temperature", "interaction",
        "disorder_full_width", "sigma", "thermopower", "L12",
    ]
    rows = []
    omega = np.linspace(-1.0, 1.0, 1001)
    for index, branch in enumerate(("arith", "typ")):
        rows.append(
            {
                "index": index, "branch": branch, "target_filling": 0.4,
                "temperature": 0.05, "interaction": 1.0,
                "disorder_full_width": 1.0,
                "sigma": 2.0 if branch == "arith" else 0.5,
                "thermopower": -1.0, "L12": 0.025,
            }
        )
        point = points / f"point_{index:06d}"
        point.mkdir(parents=True)
        center = 0.05
        rho = np.exp(-((omega - center) / 0.2) ** 2)
        np.savez_compressed(
            point / "solution.npz", omega=omega, tau=rho,
            rho_arith=rho, rho_typ=0.5 * rho,
        )
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics, _ = calculate_mechanism_diagnostics([summary], [points])
    assert len(diagnostics) == 2
    assert all(float(row["spectral_centroid"]) > 0.0 for row in diagnostics)
    assert all(float(row["transport_centroid"]) > 0.0 for row in diagnostics)
    assert all(np.isclose(float(row["sigma_typ_over_arith"]), 0.25) for row in diagnostics)
    assert all(np.isclose(float(row["rho_zero_typ_over_arith"]), 0.5) for row in diagnostics)


def test_reliability_threshold_changes_maximum():
    rows = [
        {
            "branch": "typ", "target_filling": 0.3, "temperature": 0.05,
            "interaction": 1.0, "disorder_full_width": 1.0,
            "sigma": 1.0e-5, "thermopower": 2.0, "power_factor": 4.0e-5,
        },
        {
            "branch": "typ", "target_filling": 0.3, "temperature": 0.05,
            "interaction": 1.0, "disorder_full_width": 2.0,
            "sigma": 1.0e-9, "thermopower": 10.0, "power_factor": 1.0e-7,
        },
    ]
    summary = _reliability_summary(rows, [1.0e-6, 1.0e-10])
    by_floor = {float(row["sigma_floor"]): row for row in summary}
    assert by_floor[1.0e-6]["max_abs_S_typ"] == 2.0
    assert by_floor[1.0e-10]["max_abs_S_typ"] == 10.0
    assert by_floor[1.0e-10]["max_power_factor_typ"] == 4.0e-5


def test_temperature_filter_avoids_unrequested_rows(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    points = tmp_path / "points"
    fields = [
        "index", "branch", "target_filling", "temperature", "interaction",
        "disorder_full_width", "sigma", "thermopower", "L12",
    ]
    rows = []
    omega = np.linspace(-0.5, 0.5, 101)
    for index, branch in enumerate(("arith", "typ")):
        point = points / f"point_{index:06d}"
        point.mkdir(parents=True)
        curve = np.exp(-omega**2 / 0.02)
        np.savez_compressed(
            point / "solution.npz", omega=omega, tau=curve,
            rho_arith=curve, rho_typ=curve,
        )
        for temperature in (0.02, 0.05):
            rows.append(
                {
                    "index": index, "branch": branch, "target_filling": 0.4,
                    "temperature": temperature, "interaction": 1.0,
                    "disorder_full_width": 1.0, "sigma": 0.01,
                    "thermopower": 0.0, "L12": 0.0,
                }
            )
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics, filtered = calculate_mechanism_diagnostics(
        [summary], [points], [0.05]
    )
    assert len(diagnostics) == 2
    assert len(filtered) == 2
    assert all(np.isclose(float(row["temperature"]), 0.05) for row in diagnostics)


def test_full_mechanism_analysis_writes_tables_and_figures(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    points = tmp_path / "points"
    fields = [
        "index", "branch", "target_filling", "temperature", "interaction",
        "disorder_full_width", "sigma", "thermopower", "L12",
    ]
    rows = []
    omega = np.linspace(-0.5, 0.5, 201)
    index = 0
    for branch in ("arith", "typ"):
        for disorder, l12 in ((0.5, -0.01), (1.0, 0.01)):
            sigma = 0.02 if branch == "arith" else 0.01
            rows.append(
                {
                    "index": index, "branch": branch, "target_filling": 0.4,
                    "temperature": 0.05, "interaction": 1.0,
                    "disorder_full_width": disorder, "sigma": sigma,
                    "thermopower": -l12 / (0.05 * sigma), "L12": l12,
                }
            )
            point = points / f"point_{index:06d}"
            point.mkdir(parents=True)
            curve = np.exp(-((omega - 0.03) / 0.15) ** 2)
            np.savez_compressed(
                point / "solution.npz", omega=omega, tau=curve,
                rho_arith=curve, rho_typ=0.5 * curve,
            )
            index += 1
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "analysis"
    products = analyze_transport_mechanism(
        [summary], [points], output, [1.0], [1.0e-8]
    )
    assert (output / "mechanism_diagnostics.csv").is_file()
    assert (output / "reliability_summary.csv").is_file()
    assert any(path.name.startswith("centroid_correlation") for path in products)
    assert any(path.name.startswith("mechanism_linecut") for path in products)
    assert all(path.is_file() for path in products)
