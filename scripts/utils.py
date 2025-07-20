import numpy as np


def gaussian_pulse(nx: int, ny: int, dx: float, dy: float, *, x0: float | None = None, y0: float | None = None, sigma: float | None = None) -> np.ndarray:
    """Return a Gaussian pulse centered in the domain.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the x and y directions.
    dx, dy : float
        Grid spacing in x and y.
    x0, y0 : float, optional
        Center of the pulse. Defaults to the domain center.
    sigma : float, optional
        Width of the pulse. Defaults to 10% of the smallest domain length.
    """
    x = np.linspace(0.0, (nx - 1) * dx, nx)
    y = np.linspace(0.0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y)

    if x0 is None:
        x0 = 0.5 * (nx - 1) * dx
    if y0 is None:
        y0 = 0.5 * (ny - 1) * dy
    if sigma is None:
        sigma = 0.1 * min(nx * dx, ny * dy)

    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))


def sine_distribution(nx: int, ny: int, dx: float, dy: float, *, mode_x: int = 1, mode_y: int = 1) -> np.ndarray:
    """Return a sine wave distribution.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in the x and y directions.
    dx, dy : float
        Grid spacing in x and y.
    mode_x, mode_y : int, optional
        Number of sine wave periods in x and y.
    """
    x = np.linspace(0.0, (nx - 1) * dx, nx)
    y = np.linspace(0.0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y)

    return np.sin(mode_x * np.pi * X / ((nx - 1) * dx)) * np.sin(mode_y * np.pi * Y / ((ny - 1) * dy))


def l2_error(u_numeric: np.ndarray, u_exact: np.ndarray) -> float:
    """Return the L2 error between numerical and exact solutions."""
    if u_numeric.shape != u_exact.shape:
        raise ValueError("Arrays must have the same shape")
    return float(np.sqrt(np.mean((u_numeric - u_exact) ** 2)))
