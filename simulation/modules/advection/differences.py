import numpy as np

from simulation.state import State

# TODO: handle boundary conditions correctly


def gradient(state: State, var: np.ndarray, axis: int) -> np.ndarray:
    """Return the gradient of `var` w.r.t. the given axis.

    This implements a basic first-order one-sided finite differences.
    """
    if axis == 0:
        coordinates = state.grid.z
    elif axis == 1:
        coordinates = state.grid.y
    elif axis == 2:
        coordinates = state.grid.x
    else:
        raise ValueError('Invalid axis')

    return np.gradient(var, coordinates, axis=axis)


def laplacian(state: State, var: np.ndarray) -> np.ndarray:
    """Return the lapacian of `var`.

    This implements second order central differences.
    """
    dz = state.grid.delta(0)
    dy = state.grid.delta(1)
    dx = state.grid.delta(2)

    lapl = np.zeros(state.grid.shape, dtype=var.dtype)
    lapl[1:-1, :, :] += (var[2:, :, :] - 2 * var[1:-1, :, :] + var[:-2, :, :]) / dz[1:-1, :, :]
    lapl[:, 1:-1, :] += (var[:, 2:, :] - 2 * var[:, 1:-1, :] + var[:, :-2, :]) / dy[:, 1:-1, :]
    lapl[:, :, 1:-1] += (var[:, :, 2:] - 2 * var[:, :, 1:-1] + var[:, :, :-2]) / dx[:, :, 1:-1]
    return lapl
