import numpy as np
import pytest

from nlisim.diffusion import discrete_laplacian
from nlisim.grid import TetrahedralMesh


@pytest.fixture
def mesh():
    yield TetrahedralMesh.construct_uniform(shape=(3, 3, 3), spacing=(1, 1, 1))


@pytest.fixture
def mask(mesh):
    yield np.zeros(mesh.shape, dtype=np.dtype('bool'))


def test_dense_laplacian(mesh, mask):
    mask[:] = True
    laplacian = (np.asarray(discrete_laplacian(mesh, mask).todense())).reshape(
        mesh.shape + mesh.shape
    )

    assert laplacian.sum() == 0
    assert laplacian[1, 1, 1, 1, 1, 1] == -6
    assert laplacian[0, 0, 0, 0, 0, 0] == -3
    assert laplacian[1, 0, 0, 0, 0, 0] == 1


def test_single_element_laplacian(mesh, mask):
    mask[1, 1, 1] = True
    laplacian = (np.asarray(discrete_laplacian(mesh, mask).todense())).reshape(
        mesh.shape + mesh.shape
    )

    assert laplacian.sum() == 0


def test_surface_laplacian(mesh, mask):
    mask[:, :, 1] = True
    laplacian = (np.asarray(discrete_laplacian(mesh, mask).todense())).reshape(
        mesh.shape + mesh.shape
    )

    assert laplacian.sum() == 0
    assert laplacian[1, 1, 1, 1, 1, 1] == -4
    assert (laplacian[:, :, 0, :, :, :] == 0).all()
    assert laplacian[0, 1, 1, 1, 1, 1] == 1
