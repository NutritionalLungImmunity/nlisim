import numpy as np
import pytest

from nlisim.diffusion import discrete_laplacian
from nlisim.grid import RectangularGrid


@pytest.fixture
def grid():
    yield RectangularGrid.construct_uniform(shape=(3, 3, 3), spacing=(1, 1, 1))


@pytest.fixture
def mask(grid):
    yield np.zeros(grid.shape, dtype=np.dtype('bool'))


def test_dense_laplacian(grid, mask):
    mask[:] = True
    laplacian = (np.asarray(discrete_laplacian(grid, mask).todense())).reshape(
        grid.shape + grid.shape
    )

    assert laplacian.sum() == 0
    assert laplacian[1, 1, 1, 1, 1, 1] == -6
    assert laplacian[0, 0, 0, 0, 0, 0] == -3
    assert laplacian[1, 0, 0, 0, 0, 0] == 1


def test_single_element_laplacian(grid, mask):
    mask[1, 1, 1] = True
    laplacian = (np.asarray(discrete_laplacian(grid, mask).todense())).reshape(
        grid.shape + grid.shape
    )

    assert laplacian.sum() == 0


def test_surface_laplacian(grid, mask):
    mask[:, :, 1] = True
    laplacian = (np.asarray(discrete_laplacian(grid, mask).todense())).reshape(
        grid.shape + grid.shape
    )

    assert laplacian.sum() == 0
    assert laplacian[1, 1, 1, 1, 1, 1] == -4
    assert (laplacian[:, :, 0, :, :, :] == 0).all()
    assert laplacian[0, 1, 1, 1, 1, 1] == 1
