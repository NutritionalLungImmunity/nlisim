from typing import Any, Tuple

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
from scipy.sparse import csr_matrix, dok_matrix, eye, identity

# noinspection PyPackageRequirements
from scipy.sparse.linalg import cg
from skfem import Basis, ElementTetP1, MeshTet1, condense
from skfem.models import laplace
from skfem.utils import solve

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid, TetrahedralMesh
from nlisim.state import State

_dtype_float64 = np.dtype('float64')


def discrete_laplacian(grid: RectangularGrid, mask: np.ndarray, dtype=_dtype_float64) -> csr_matrix:
    """Return a discrete laplacian operator for the given restricted mesh.

    This computes a standard laplacian operator as a scipy linear operator, except it is
    restricted to a mesh mask.  The use case for this is to compute surface diffusion
    on a gridded variable.  The mask is generated from a category on the lung_tissue
    variable.
    """
    graph_shape = len(grid), len(grid)
    laplacian = dok_matrix(graph_shape, dtype=dtype)

    delta_z = grid.delta(0)
    delta_y = grid.delta(1)
    delta_x = grid.delta(2)

    for k, j, i in zip(*(mask).nonzero()):
        voxel = Voxel(x=i, y=j, z=k)
        voxel_index = grid.get_flattened_index(voxel)

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
            inverse_distance2 = 1 / (dx * dx + dy * dy + dz * dz)  # units: 1/(µm^2)

            laplacian[voxel_index, voxel_index] -= inverse_distance2
            laplacian[voxel_index, neighbor_index] += inverse_distance2

    return laplacian.tocsr()


def periodic_discrete_laplacian(
    grid: RectangularGrid, mask: np.ndarray, dtype: np.dtype = _dtype_float64
) -> csr_matrix:
    """Return a laplacian operator with periodic boundary conditions.

    This computes a standard laplacian operator as a scipy linear operator, except it is
    restricted to a mesh mask.  The use case for this is to compute surface diffusion
    on a gridded variable.  The mask is generated from a category on the lung_tissue
    variable.
    """
    graph_shape = len(grid), len(grid)
    z_extent, y_extent, x_extent = grid.shape
    laplacian = dok_matrix(graph_shape, dtype=dtype)

    delta_z = grid.delta(0)
    delta_y = grid.delta(1)
    delta_x = grid.delta(2)

    for k, j, i in zip(*(mask).nonzero()):
        voxel = Voxel(x=i, y=j, z=k)
        voxel_index = grid.get_flattened_index(voxel)

        for offset in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            # voxel coordinate displacements
            dk, dj, di = offset

            # find the neighbor for periodic boundary conditions
            neighbor: Voxel = Voxel(
                x=(i + di) % x_extent, y=(j + dj) % y_extent, z=(k + dk) % z_extent
            )

            # but maybe it isn't in the mask (i.e. air)
            if not mask[neighbor.z, neighbor.y, neighbor.x]:
                continue

            neighbor_index = grid.get_flattened_index(neighbor)

            # continuous space displacements
            dx = delta_x[k, j, i] * di
            dy = delta_y[k, j, i] * dj
            dz = delta_z[k, j, i] * dk
            inverse_distance2 = 1 / (dx * dx + dy * dy + dz * dz)  # units: 1/(µm^2)

            laplacian[voxel_index, voxel_index] -= inverse_distance2
            laplacian[voxel_index, neighbor_index] += inverse_distance2

    return laplacian.tocsr()


def apply_grid_diffusion(
    variable: np.ndarray,
    laplacian: csr_matrix,
    diffusivity: float,
    dt: float,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Apply diffusion to a variable.

    Solves Laplace's equation in 3D using Crank-Nicholson.  The variable is
    advanced in time by `dt` time units using the conjugate gradient method.

    Note that, due to numerical error, we cannot guarantee that the quantity
    of the molecule will remain constant.

    The intended use case for this method is to perform "surface diffusion" generated
    by a mask from the `lung_tissue` variable, e.g.

        surface_mask = lung_tissue == TissueTypes.EPITHELIUM
        laplacian = discrete_laplacian(mesh, mask)
        iron_concentration[:] = apply_diffusion(iron_concentration, laplacian, diffusivity, dt)
    """
    a = eye(*laplacian.shape) - (diffusivity * dt / 2.0) * laplacian
    b = eye(*laplacian.shape) + (diffusivity * dt / 2.0) * laplacian
    var_next, info = cg(a, b @ variable.ravel(), tol=tolerance)
    if info > 0:
        raise Exception(f'CG failed (after {info} iterations)')
    elif info < 0:
        raise Exception(f'CG failed ({info})')

    return np.maximum(0.0, var_next.reshape(variable.shape))


def assemble_mesh_laplacian_crank_nicholson(
    *,
    state: State,
    diffusivity: float,
    dt: float,
) -> Tuple[csr_matrix, csr_matrix, Any]:
    mesh: TetrahedralMesh = state.mesh

    # load a tetrahedral mesh, will be a MeshTet1
    skmesh: MeshTet1 = MeshTet1(doflocs=mesh.points.T, t=mesh.element_point_indices.T)
    basis = Basis(skmesh, ElementTetP1())

    # create matrices for Crank-Nicholson method
    a = identity(basis.dofs.N) + 0.5 * dt * diffusivity * laplace.assemble(basis)
    b = identity(basis.dofs.N) - 0.5 * dt * diffusivity * laplace.assemble(basis)

    # Empty call to get_dofs() matches all boundary DOFs.
    dofs = basis.get_dofs()

    return a, b, dofs


def apply_mesh_diffusion_crank_nicholson(
    *,
    variable: np.ndarray,
    cn_a: csr_matrix,
    cn_b: csr_matrix,
    dofs: Any,
    tolerance: float = 1e-10,
):
    return solve(*condense(cn_a, cn_b @ variable, D=dofs), tolerance=tolerance)
