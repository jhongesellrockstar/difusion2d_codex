"""Visualization utilities for 2D diffusion results."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def plot_contour(u: np.ndarray) -> None:
    """Display a contour plot of the temperature field ``u``.

    Parameters
    ----------
    u : numpy.ndarray
        Two-dimensional array representing the temperature field.
    """
    fig, ax = plt.subplots()
    cont = ax.contourf(u, cmap="viridis", levels=50)
    fig.colorbar(cont, ax=ax, label="Temperature")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Temperature field")
    plt.show()


def animate_diffusion(history: Sequence[np.ndarray], *, interval: int = 50) -> animation.FuncAnimation:
    """Animate the diffusion process given a sequence of states.

    Parameters
    ----------
    history : Sequence[numpy.ndarray]
        Sequence of 2D arrays with the temperature field at each time step.
    interval : int, optional
        Delay between frames in milliseconds. Default is ``50``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object which can be further saved or displayed.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(history[0], cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax, label="Temperature")

    def update(frame: np.ndarray) -> list[animation.Artist]:
        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=history,
        interval=interval,
        blit=True,
        repeat=False,
    )
    plt.show()
    return anim
