"""13-point explicit solver for the 2D diffusion equation."""

from __future__ import annotations

import numpy as np


def _pad_dirichlet(u: np.ndarray) -> np.ndarray:
    """Return ``u`` extended with two ghost cells using reflection and sign flip.

    This helper is used to implement linear extrapolation for zero Dirichlet
    boundaries, allowing the 13-point stencil to be evaluated near the
    domain edges without accessing values outside the array.
    """
    ext = np.pad(u, 2, mode="reflect")
    ext[:2, :] *= -1
    ext[-2:, :] *= -1
    ext[:, :2] *= -1
    ext[:, -2:] *= -1
    return ext


def solve_13point_2d(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    alpha: float,
    s: float = 0.5,
) -> np.ndarray:
    """Solve the 2D diffusion equation using Dehghan's (1,13) explicit scheme.

    The method employs a 13-point stencil that includes the four immediate
    neighbours, the four diagonal points and the four points two cells away in
    the coordinate directions.  A simple extrapolation is used at the domain
    borders so that the stencil can be applied uniformly on the whole grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the ``x`` and ``y`` directions.
    dx, dy : float
        Grid spacing in ``x`` and ``y`` (assumed equal).
    dt : float
        Time step.
    nt : int
        Number of time iterations.
    alpha : float
        Diffusion coefficient.
    s : float, optional
        Stability parameter. The quantity ``alpha*dt/dx**2`` must not exceed
        ``s``.

    Returns
    -------
    numpy.ndarray
        Temperature field after ``nt`` time steps with zero Dirichlet
        boundaries.
    """
    if abs(dx - dy) > 1e-14:
        raise ValueError("The (1,13) scheme assumes a uniform grid: dx must equal dy")

    r = alpha * dt / dx ** 2
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

    # Zero Dirichlet boundary conditions
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    u_new = np.empty_like(u)
    for _ in range(nt):
        ext = _pad_dirichlet(u)
        u_new[:, :] = (
            (1 - 8 * r) * u
            + r
            * (
                ext[3:-1, 2:-2]
                + ext[1:-3, 2:-2]
                + ext[2:-2, 3:-1]
                + ext[2:-2, 1:-3]
            )
            + 0.5
            * r
            * (
                ext[3:-1, 3:-1]
                + ext[3:-1, 1:-3]
                + ext[1:-3, 3:-1]
                + ext[1:-3, 1:-3]
            )
            + 0.5
            * r
            * (
                ext[4:, 2:-2]
                + ext[:-4, 2:-2]
                + ext[2:-2, 4:]
                + ext[2:-2, :-4]
            )
        )

        # Reapply Dirichlet boundaries
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0

        u, u_new = u_new, u

    return u


if __name__ == "__main__":
    res = solve_13point_2d(nx=51, ny=51, dx=0.02, dy=0.02, dt=1e-4, nt=10, alpha=1.0)
    print(res)
