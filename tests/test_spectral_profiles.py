import csv
from pathlib import Path

import numpy as np

from fk_transport.spectral_profiles import _load_spectral_profile, plot_spectral_profiles


def test_load_spectral_profile_uses_branch_consistent_density(tmp_path: Path):
    point = tmp_path / "point_000003"
    point.mkdir()
    omega = np.linspace(-0.5, 0.5, 101)
    np.savez_compressed(
        point / "solution.npz",
        omega=omega,
        rho_arith=np.ones_like(omega),
        rho_typ=2.0 * np.ones_like(omega),
    )
    arith = _load_spectral_profile(tmp_path, 3, "arith", 0.05)
    typ = _load_spectral_profile(tmp_path, 3, "typ", 0.05)
    assert arith["rho_zero"] == 1.0
    assert typ["rho_zero"] == 2.0
    assert abs(float(arith["spectral_first_moment"])) < 1.0e-14


def test_plot_spectral_profiles_writes_figure_and_diagnostics(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    points = tmp_path / "points"
    fields = [
        "index", "branch", "target_filling", "temperature", "interaction",
        "disorder_full_width", "sigma", "thermopower", "L12",
    ]
    rows = []
    index = 0
    omega = np.linspace(-0.5, 0.5, 101)
    for branch in ("arith", "typ"):
        for disorder, thermopower, l12, sigma in (
            (0.5, 0.2, -0.01, 1.0e-2),
            (1.0, -0.8, 0.02, 1.0e-3),
            (1.5, -0.1, 0.003, 1.0e-6),
        ):
            rows.append(
                {
                    "index": index, "branch": branch, "target_filling": 0.4,
                    "temperature": 0.02, "interaction": 1.0,
                    "disorder_full_width": disorder,
                    "sigma": sigma if branch == "arith" else sigma * 0.5,
                    "thermopower": thermopower, "L12": l12,
                }
            )
            point = points / f"point_{index:06d}"
            point.mkdir(parents=True)
            np.savez_compressed(
                point / "solution.npz", omega=omega,
                rho_arith=np.exp(-omega**2 / 0.02),
                rho_typ=0.5 * np.exp(-(omega - 0.01)**2 / 0.02),
            )
            index += 1
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "analysis"
    products = plot_spectral_profiles(summary, points, output, [1.0])
    assert (output / "spectral_profile_points.csv").is_file()
    assert (output / "spectral_profile_diagnostics.csv").is_file()
    assert any(path.name.startswith("spectral_profiles_n_0p4") for path in products)
