from typing import cast, Set

from h5py import Group
import numpy as np
from pytest import fixture

from simulation.config import SimulationConfig
from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.modules.fungus import (
    FungusCellData,
    FungusCellList,
    FungusState,
)
from simulation.modules.macrophage import (
    MacrophageCellData,
    MacrophageCellList,
    MacrophageState,
)
from simulation.state import State

@fixture
def iron():
	# a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(10)
    yield i

@fixture
def macrophage_list(grid: RectangularGrid):
    macrophage = MacrophageCellList(grid=grid)
    yield macrophage

@fixture
def populated_macrophage(macrophage_list: MacrophageCellList):
	points = []
	for i in range(10, 60, 10):
		points.append(Point(x=i, y=i, z=i))
	
	for point in points:
		macrophage_list.append(
	        MacrophageCellData.create_cell(
	            point=point,
	        )
	    )
	yield macrophage_list