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

@fixture
def fungus_list(grid: RectangularGrid):
    fungus = FungusCellList(grid=grid)
    yield fungus

@fixture
def populated_fungus(fungus_list: FungusCellList, grid: RectangularGrid):
	points = []
	for i in range(int(grid.xv[1]), int(grid.xv[6]), 10):
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


def test_recruit_new(macrophage_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    p_rec_r = 1.0
    rec_rate_ph = 2

    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5,5,5] = 2

    # test correct location recruitment
    macrophage_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            p_rec_r, 
            tissue, 
            grid, 
            cyto)
    vox = grid.get_voxel(macrophage_list[-1]['point'])
    
    assert len(macrophage_list) == 2
    assert vox.x == 5 and vox.y == 5 and vox.z == 5

    # test recruit none due to below threshold
    rec_r = 20
    p_rec_r = 1.0
    rec_rate_ph = 2
    macrophage_list.recruit_new(
        rec_rate_ph, 
        rec_r, 
        p_rec_r, 
        tissue, 
        grid, 
        cyto)
    
    assert len(macrophage_list) == 2

def test_recruit_new_multiple_locations(
    macrophage_list: MacrophageCellList, 
    tissue, 
    grid: RectangularGrid, cyto):

    rec_r = 2
    p_rec_r = 1.0
    rec_rate_ph = 50

    cyto[5,5,5] = 2
    cyto[4,5,5] = 2

    macrophage_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            p_rec_r, 
            tissue, 
            grid, 
            cyto)
    
    assert len(macrophage_list) == 50

    for cell in macrophage_list.cell_data:
        vox = grid.get_voxel(cell['point'])
        assert vox.z == 5 and vox.y == 5 and vox.x in [4, 5]

def test_absorb_cytokines(macrophage_list: MacrophageCellList, cyto, grid: RectangularGrid):
    rec_r = 2
    p_rec_r = 1.0
    rec_rate_ph = 1

    cyto[1,2,3] = 64

    macrophage_list.append(
	        MacrophageCellData.create_cell(
	            point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
	        )
	    )
    
    assert len(macrophage_list) == 1
    assert cyto[1,2,3] == 64

    m_abs = 0.5
    macrophage_list.absorb_cytokines(m_abs, cyto, grid)
    assert cyto[1,2,3] == 32

    macrophage_list.absorb_cytokines(m_abs, cyto, grid)
    assert cyto[1,2,3] == 16

    macrophage_list.append(
	        MacrophageCellData.create_cell(
	            point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
	        )
	    )
    
    macrophage_list.absorb_cytokines(m_abs, cyto, grid)
    assert cyto[1,2,3] == 4



def test_produce_cytokines_1(macrophage_list: MacrophageCellList, grid: RectangularGrid, populated_fungus: FungusCellList, cyto):
    m_det = 1
    Mn = 10

    assert cyto[3,3,3] == 0

    macrophage_list.append(
	        MacrophageCellData.create_cell(
	            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
	        )
	    )

    vox = grid.get_voxel(macrophage_list[0]['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    macrophage_list.produce_cytokines(m_det, Mn, grid, populated_fungus, cyto)

    assert cyto[3,3,3] == 10

def test_produce_cytokines_N(macrophage_list: MacrophageCellList, grid: RectangularGrid, populated_fungus: FungusCellList, cyto):
    m_det = 3
    Mn = 10

    assert cyto[3,3,3] == 0
    macrophage_list.append(
	        MacrophageCellData.create_cell(
	            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
	        )
	    )
    vox = grid.get_voxel(macrophage_list[0]['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    # 3
    macrophage_list.produce_cytokines(m_det, Mn, grid, populated_fungus, cyto)
    assert cyto[3,3,3] == 30

    # 6
    m_det = 6
    cyto[3,3,3] = 0

    macrophage_list.produce_cytokines(m_det, Mn, grid, populated_fungus, cyto)
    assert cyto[3,3,3] == 50      

#move(state)
#internalize_conidia(state)
#damage_conidia(state, previous_time)