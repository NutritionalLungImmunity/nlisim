import numpy as np
from pytest import fixture

# from nlisim.grid import RectangularGrid
from nlisim.oldmodules.molecules import Molecules


@fixture
def iron():
    # a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(10)
    yield i


@fixture
def cyto():
    # a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(0)
    yield i


@fixture
def tissue():
    # a 10 x 10 x 10 grid of blood
    t = np.empty((10, 10, 10))
    t.fill(1)
    t[6:] = 3
    yield t


@fixture
def air_tissue():
    # a 10 x 10 x 10 grid of blood
    t = np.empty((10, 10, 10))
    t.fill(0)
    t[5, 5, 5] = 3
    yield t


# tests


# def test_diffuse(tissue, grid: RectangularGrid, cyto):
#     cyto[1, 2, 3] = 26

#     Molecules.diffuse(cyto, grid, tissue)

#     assert cyto[1, 2, 3] == 1
#     assert cyto[0, 2, 3] == 1
#     assert cyto[1, 1, 3] == 1
#     assert cyto[1, 2, 2] == 1
#     assert cyto[1, 2, 4] == 1
#     assert cyto[1, 3, 3] == 1
#     assert cyto[2, 2, 3] == 1

#     assert cyto[3, 2, 1] == 0


def test_convolution_diffusion(tissue, cyto):
    cyto[1, 2, 3] = 27

    Molecules.convolution_diffusion(cyto, tissue)

    assert cyto[1, 2, 3] == 1
    assert cyto[0, 2, 3] == 1
    assert cyto[1, 1, 3] == 1
    assert cyto[1, 2, 2] == 1
    assert cyto[1, 2, 4] == 1
    assert cyto[1, 3, 3] == 1
    assert cyto[2, 2, 3] == 1

    assert cyto[3, 2, 1] == 0

    assert cyto.sum() == 27


def test_convolution_diffusion_air(air_tissue, cyto):
    cyto[5, 5, 5] = 27

    Molecules.convolution_diffusion(cyto, air_tissue)

    assert cyto[5, 5, 5] == 1

    assert cyto.sum() == 1


def test_degrade(cyto):
    cyto[1, 2, 3] = 10

    Molecules.degrade(cyto, 0.1)

    assert cyto[1, 2, 3] == 9


# def test_diffuse_iron(iron, grid, tissue):
#     iron[:] = 0
#     iron[1, 2, 3] = 260
#     tissue[1, 2, 3] = 1

#     Molecules.diffuse_iron(iron, grid, tissue, 26)

#     assert iron[1, 2, 3] == 1
