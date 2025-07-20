"""9-point explicit solver for the 2D diffusion equation."""

from __future__ import annotations

import numpy as np


def solve_9point_2d(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    alpha: float,
    s: float = 0.25,
) -> np.ndarray:
    """Solve the 2D diffusion equation using a 9-point explicit scheme.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the x and y directions.
    dx, dy : float
        Grid spacing in the x and y directions (should be equal).
    dt : float
        Time step size.
    nt : int
        Number of time steps to perform.
    alpha : float
        Diffusion coefficient.
    s : float, optional
        Stability parameter. ``alpha * dt / dx**2`` must be less than or
        equal to ``s``.

    Returns
    -------
    np.ndarray
        Temperature field after ``nt`` time steps with zero Dirichlet
        boundaries.
    """
    if abs(dx - dy) > 1e-14:
        raise ValueError("The 9-point scheme assumes a uniform grid: dx must equal dy")

    r = alpha * dt / dx**2
    if r > s:
        raise ValueError(
            f"Stability condition violated: alpha*dt/dx**2 = {r:.3e} > s = {s}"
        )

    x = np.linspace(0.0, (nx - 1) * dx, nx)
    y = np.linspace(0.0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y)

    x0 = 0.5 * (nx - 1) * dx
    y0 = 0.5 * (ny - 1) * dy
    sigma = 0.1 * min(nx * dx, ny * dy)
    u = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

    # Apply zero Dirichlet boundary conditions
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    u_new = np.empty_like(u)
    for _ in range(nt):
        u_new[1:-1, 1:-1] = (
            (1 - 5 * r) * u[1:-1, 1:-1]
            + r * (
                u[2:, 1:-1]
                + u[:-2, 1:-1]
                + u[1:-1, 2:]
                + u[1:-1, :-2]
            )
            + 0.25
            * r
            * (
                u[2:, 2:]
                + u[2:, :-2]
                + u[:-2, 2:]
                + u[:-2, :-2]
            )
        )

        # Reapply Dirichlet conditions
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0

        u, u_new = u_new, u

    return u


if __name__ == "__main__":
    res = solve_9point_2d(nx=51, ny=51, dx=0.02, dy=0.02, dt=1e-4, nt=100, alpha=1.0)
    print(res)
