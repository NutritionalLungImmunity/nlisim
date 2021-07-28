import numpy as np
from scipy.sparse import csr_matrix, dok_matrix, eye
from scipy.sparse.linalg import cg

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid

_dtype_float64 = np.dtype('float64')


def discrete_laplacian(
    grid: RectangularGrid, mask: np.ndarray, dtype: np.dtype = _dtype_float64
) -> csr_matrix:
    """Return a discrete laplacian operator for the given restricted grid.

    This computes a standard laplacian operator as a scipy linear operator, except it is
    restricted to a grid mask.  The use case for this is to compute surface diffusion
    on a gridded variable.  The mask is generated from a category on the lung_tissue
    variable.
    """
    graph_shape = len(grid), len(grid)
    laplacian = dok_matrix(graph_shape)

    delta_z = grid.delta(0)
    delta_y = grid.delta(1)
    delta_x = grid.delta(2)

    for k, j, i in zip(*(mask).nonzero()):
        voxel = Voxel(x=i, y=j, z=k)
        voxel_index = grid.get_flattened_index(voxel)
        normalization = 0

        for neighbor in grid.get_adjacent_voxels(voxel, corners=False):
            ni = neighbor.x
            nj = neighbor.y
            nk = neighbor.z

            if not mask[nk, nj, ni]:
                continue

            neighbor_index = grid.get_flattened_index(neighbor)

            dx = delta_x[k, j, i] * (i - ni)
            dy = delta_y[k, j, i] * (j - nj)
            dz = delta_z[k, j, i] * (k - nk)
            distance2 = 1 / (dx * dx + dy * dy + dz * dz)

            normalization -= distance2
            laplacian[voxel_index, neighbor_index] = distance2

        laplacian[voxel_index, voxel_index] = normalization

    return laplacian.tocsr()


def apply_diffusion(
    variable: np.ndarray, laplacian: csr_matrix, diffusivity: float, dt: float
) -> np.ndarray:
    """Apply diffusion to a variable.

    Solves Laplace's equation in 3D using implicit time steps.  The variable is
    advanced in time by `dt` time units using GMRES.

    Notes that the output of this function might contain negative values caused by
    rounding error. You can truncate the result by var_next[var_next < 0] = 0.

    The intended use case for this method is to perform "surface diffusion" generated
    by a mask from the `lung_tissue` variable, e.g.

        surface_mask = lung_tissue == TissueTypes.EPITHELIUM
        laplacian = discrete_laplacian(grid, mask)
        iron_concentration[:] = apply_diffusion(iron_concentration, laplacian, diffusivity, dt)
    """
    operator = eye(*laplacian.shape) - (diffusivity * dt) * laplacian
    var_next, info = cg(operator, variable.ravel())
    if info != 0:
        raise Exception(f'GMRES failed ({info})')
    return var_next.reshape(variable.shape)
