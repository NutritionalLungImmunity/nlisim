import numpy as np

from simulation.state import State


def gradient(state: State, var: np.ndarray, axis: int) -> np.ndarray:
    """Return the gradient of `var` w.r.t. the given axis.

    This implements a basic first-order one-sided finite differences.
    """
    if axis not in (0, 1):
        raise Exception('Invalid axis provided')

    if axis == 1:
        # TODO: assumes bc's are the same at every boundary
        return gradient(state.replace(dx=state.dy, dy=state.dx), var.T, 0).T

    dy = state.dy
    bc = state.bc

    grad = np.zeros(var.shape, dtype=var.dtype)
    grad[:-1, :] = (var[1:, :] - var[:-1, :]) / dy
    bc.gradient(state, var, grad, axis)
    return grad


def laplacian(state: State, var: np.ndarray) -> np.ndarray:
    """Return the lapacian of `var`.

    This implements second order central differences.
    """
    dx = state.dx
    dy = state.dy
    bc = state.bc

    lapl = np.zeros(var.shape, dtype=var.dtype)

    lapl[1:-1, :] = (var[:-2, :] - 2 * var[1:-1, :] + var[2:, :]) / (dy * dy)
    lapl[:, 1:-1] += (var[:, :-2] - 2 * var[:, 1:-1] + var[:, 2:]) / (dx * dx)
    bc.laplacian(state, var, lapl)
    return lapl
