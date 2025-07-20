#!/usr/bin/env python3
"""Save benchmark metrics to CSV (and optionally LaTeX)."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from difusion2d_13pt import solve_13point_2d
from difusion2d_9pt import solve_9point_2d
from difusion2d_ftcs import solve_ftcs_2d
from utils import l2_error

Solver = Callable[[int, int, float, float, float, int, float], np.ndarray]


def run_solver(solver: Solver, nx: int, ny: int, dx: float, dy: float, dt: float,
               nt: int, alpha: float) -> Tuple[np.ndarray, float]:
    """Run *solver* and return its output and runtime."""
    start = time.perf_counter()
    u = solver(nx, ny, dx, dy, dt, nt, alpha)
    runtime = time.perf_counter() - start
    return u, runtime


def benchmark(nx: int = 101, ny: int = 101, dt: float = 1e-6, nt: int = 1000,
              alpha: float = 1.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run all solvers and return their times and errors."""
    dx = dy = 1.0 / (nx - 1)

    ref = solve_13point_2d(nx, ny, dx, dy, dt / 2, nt * 2, alpha)

    solvers: Dict[str, Solver] = {
        "FTCS": solve_ftcs_2d,
        "9 puntos": solve_9point_2d,
        "13 puntos": solve_13point_2d,
    }

    times: Dict[str, float] = {}
    errors: Dict[str, float] = {}
    for name, solver in solvers.items():
        u, runtime = run_solver(solver, nx, ny, dx, dy, dt, nt, alpha)
        times[name] = runtime
        errors[name] = l2_error(u, ref)
    return times, errors


def save_csv(filepath: Path, times: Dict[str, float], errors: Dict[str, float]) -> None:
    """Save benchmark results to *filepath* in CSV format."""
    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Método", "Tiempo", "Error"])
        for name in times:
            writer.writerow([name, f"{times[name]:.6f}", f"{errors[name]:.6e}"])


def save_latex(filepath: Path, times: Dict[str, float], errors: Dict[str, float]) -> None:
    """Save benchmark results as a LaTeX tabular."""
    lines = [
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Método & Tiempo (s) & Error \\",
        "\\hline",
    ]
    for name in times:
        lines.append(f"{name} & {times[name]:.3f} & {errors[name]:.3e}\\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    filepath.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Save benchmark metrics")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "resultados" / "benchmark_results.csv", help="CSV output path")
    parser.add_argument("--tex", type=Path, default=Path(__file__).resolve().parent.parent / "resultados" / "benchmark_results.tex", help="LaTeX output path")
    args = parser.parse_args()

    times, errors = benchmark()
    save_csv(args.output, times, errors)
    if args.latex:
        save_latex(args.tex, times, errors)


if __name__ == "__main__":
    main()
