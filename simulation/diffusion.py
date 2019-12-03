import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from simulation.state import RectangularGrid


# TODO: Boundary conditions?
def discrete_laplacian(grid: RectangularGrid, dtype: np.dtype, dt: float) -> LinearOperator:
    """Return a discrete laplacian operator for the given grid."""
    dz = grid.delta(0)
    dy = grid.delta(1)
    dx = grid.delta(2)

    hz2 = dt / np.square(dz[1:-1, :, :])
    hy2 = dt / np.square(dy[:, 1:-1, :])
    hx2 = dt / np.square(dx[:, :, 1:-1])

    def matvec(var: np.ndarray) -> np.ndarray:
        var = var.reshape(grid.shape)

        # Discrete laplacian
        lapl = np.zeros(grid.shape, dtype=var.dtype)
        lapl[1:-1, :, :] -= (var[2:, :, :] - 2 * var[1:-1, :, :] + var[:-2, :, :]) * hz2
        lapl[:, 1:-1, :] -= (var[:, 2:, :] - 2 * var[:, 1:-1, :] + var[:, :-2, :]) * hy2
        lapl[:, :, 1:-1] -= (var[:, :, 2:] - 2 * var[:, :, 1:-1] + var[:, :, :-2]) * hx2

        # RHS (TODO: implement 2nd order time stepping?)
        lapl += var
        return lapl

    n = len(grid)
    return LinearOperator(matvec=matvec, dtype=dtype, shape=(n, n))


def diffusion_step(grid: RectangularGrid, var: np.ndarray, diffusivity: float, dt: float) -> None:
    """Apply diffusion to a variable.

    Solves laplaces equation in 3D using implicit time steps.  The variable is
    advanced in time by `dt` time units using gmres.
    """
    lapl = discrete_laplacian(grid, var.dtype, dt)
    var_next, info = cg(lapl, var.ravel())
    if info != 0:
        raise Exception(f'GMRES failed ({info})')
    var[:] = var_next.reshape(grid.shape)
