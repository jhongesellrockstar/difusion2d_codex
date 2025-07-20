"""Benchmark script comparing 2D diffusion solvers."""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from difusion2d_13pt import solve_13point_2d
from difusion2d_9pt import solve_9point_2d
from difusion2d_ftcs import solve_ftcs_2d
from utils import l2_error
from visualization import plot_contour

Solver = Callable[[int, int, float, float, float, int, float], np.ndarray]


def run_solver(solver: Solver, nx: int, ny: int, dx: float, dy: float, dt: float,
               nt: int, alpha: float) -> Tuple[np.ndarray, float]:
    """Run *solver* and return the result and its runtime."""
    start = time.perf_counter()
    u = solver(nx, ny, dx, dy, dt, nt, alpha)
    runtime = time.perf_counter() - start
    return u, runtime


def benchmark(nx: int = 101, ny: int = 101, dt: float = 1e-6, nt: int = 1000,
              alpha: float = 1.0, visualize: bool = False) -> None:
    """Compare the available solvers on a moderate grid."""
    dx = dy = 1.0 / (nx - 1)

    ref = solve_13point_2d(nx, ny, dx, dy, dt / 2, nt * 2, alpha)

    solvers: Dict[str, Solver] = {
        "FTCS": solve_ftcs_2d,
        "9-point": solve_9point_2d,
        "13-point": solve_13point_2d,
    }

    times = []
    errors = []
    states = {}
    for name, solver in solvers.items():
        u, runtime = run_solver(solver, nx, ny, dx, dy, dt, nt, alpha)
        err = l2_error(u, ref)
        times.append(runtime)
        errors.append(err)
        states[name] = u

    print("\nBenchmark results:")
    print(f"{'Scheme':>10} {'Time (s)':>12} {'L2 error':>12}")
    for name, t, e in zip(solvers.keys(), times, errors):
        print(f"{name:>10} {t:12.6f} {e:12.6e}")

    fig, ax1 = plt.subplots()
    ind = np.arange(len(solvers))
    width = 0.35
    ax1.bar(ind, times, width, label="Time (s)")
    ax1.set_ylabel("Time (s)")
    ax1.set_xticks(ind)
    ax1.set_xticklabels(list(solvers.keys()))
    ax1.set_title("Runtime and accuracy")

    ax2 = ax1.twinx()
    ax2.plot(ind, errors, "o-r", label="L2 error")
    ax2.set_ylabel("L2 error")

    fig.tight_layout()
    plt.show()

    if visualize:
        plot_contour(states["13-point"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark diffusion solvers")
    parser.add_argument("--visualize", action="store_true",
                        help="Show contour of the 13-point result")
    args = parser.parse_args()
    benchmark(visualize=args.visualize)
