"""2D diffusion solver using FTCS scheme"""

import numpy as np


def solve_ftcs_2d(nx: int, ny: int, dx: float, dy: float, dt: float,
                   nt: int, alpha: float) -> np.ndarray:
    """Solve the 2D diffusion equation using the FTCS scheme.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the x and y directions.
    dx, dy : float
        Grid spacing in the x and y directions.
    dt : float
        Time step size.
    nt : int
        Number of time steps to perform.
    alpha : float
        Diffusion coefficient.

    Returns
    -------
    np.ndarray
        Temperature field after ``nt`` time steps with Dirichlet
        boundary conditions set to zero.
    """
    # Create coordinate arrays
    x = np.linspace(0.0, (nx - 1) * dx, nx)
    y = np.linspace(0.0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y)

    # Initial condition: Gaussian pulse in the center
    x0 = 0.5 * (nx - 1) * dx
    y0 = 0.5 * (ny - 1) * dy
    sigma = 0.1 * min(nx * dx, ny * dy)
    u = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

    # Apply Dirichlet boundary conditions (zero temperature) explicitly
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    r_x = alpha * dt / dx ** 2
    r_y = alpha * dt / dy ** 2

    u_new = np.empty_like(u)
    for _ in range(nt):
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + r_x * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
            + r_y * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        )

        # Enforce Dirichlet boundaries
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0

        u, u_new = u_new, u

    return u


if __name__ == "__main__":
    # Simple example when executed directly
    res = solve_ftcs_2d(nx=51, ny=51, dx=0.02, dy=0.02, dt=1e-4, nt=100, alpha=1.0)
    print(res)
