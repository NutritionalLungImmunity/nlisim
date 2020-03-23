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
from simulation.state import State

@fixture
def iron():
	# a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(10)
    yield i

@fixture
def fungus_list(grid: RectangularGrid):
    fungus = FungusCellList(grid=grid)
    yield fungus

@fixture
def populated_fungus(fungus_list: FungusCellList):
	points = []
	for i in range(10, 60, 10):
		points.append(Point(x=i, y=i, z=i))
	
	for point in points:
		fungus_list.append(
	        FungusCellData.create_cell(
	            point=point,
	            status=FungusCellData.Status.GROWABLE,
	            form=FungusCellData.Form.HYPHAE,
	            iron=0,
	            mobile=False,
	        )
	    )
	yield fungus_list

def test_iron_uptake(populated_fungus: FungusCellList, iron):
	iron_min = 5
	iron_max = 100
	iron_absorb = 0.5
	assert iron[5, 5, 5] == 10

	populated_fungus.iron_uptake(iron, iron_max, iron_min, iron_absorb)

	for cell in populated_fungus.cell_data:
		assert cell['iron'] == 5

def test_fungus_spawn(populated_fungus, point):
	populated_fungus.spawn_hypahael_cell(point, 10, 1)

	loc = populated_fungus.cell_data[-1]['point']
	iron = populated_fungus.cell_data[-1]['iron']

	assert abs(loc[0] - point[0]) < 1 and abs(loc[1] - point[1]) and abs(loc[2] - point[2])
	assert len(populated_fungus) == 6
	assert iron == 10

def test_age(populated_fungus):
	cells = populated_fungus.cell_data
	cells['dead'][0] = True
	for _ in range(10):
		populated_fungus.age()

	# print(cells['iteration'])

	assert cells['iteration'][0] == 0
	assert (cells['iteration'][1:] == 10).all()

def test_kill(populated_fungus):
	cells = populated_fungus.cell_data
	cells['health'][0] = 0
	cells['point'][1][0] = -1
	populated_fungus.kill()
	# print(cells)
	assert (cells['dead'][0:1] == True).all() and (cells['dead'][2:]== False).all()
	assert (cells['status'][0:1] == FungusCellData.Status.DEAD).all()
	assert (cells['status'][2:] != FungusCellData.Status.DEAD).all()

def test_change_state(populated_fungus):
	iter_to_change_status = 10
	cells = populated_fungus.cell_data
	for _ in range(10):
		populated_fungus.age()
	cells['form'][0:2] = FungusCellData.Form.CONIDIA
	cells['status'][0] = FungusCellData.Status.RESTING
	cells['status'][1] = FungusCellData.Status.SWOLLEN
	populated_fungus.change_status(iter_to_change_status)
	assert (cells['iteration'][0:2] == 0).all()
	assert cells['status'][0] == FungusCellData.Status.SWOLLEN
	assert cells['status'][1] == FungusCellData.Status.GERMINATED

