import numpy as np
from pytest import fixture

from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.modules.molecules import Molecules


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


# tests


def diffuse(tissue, grid: RectangularGrid, cyto):
    cyto[1,2,3] = 26

    Molecules.diffuse(cyto, grid, tissue)

    assert cyto[1,2,3] == 26
    
    

