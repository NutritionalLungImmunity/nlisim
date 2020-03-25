import numpy as np
from pytest import fixture

from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.modules.fungus import (
    FungusCellData,
    FungusCellList,
)
from simulation.modules.neutrophil import (
    NeutrophilCellData,
    NeutrophilCellList,
)


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
def neutrophil_list(grid: RectangularGrid):
    neutrophil = NeutrophilCellList(grid=grid)
    yield neutrophil


@fixture
def populated_neutrophil(neutrophil_list: NeutrophilCellList, grid: RectangularGrid):
    neutrophil_list.append(
        NeutrophilCellData.create_cell(point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),)
    )

    yield neutrophil_list


@fixture
def fungus_list(grid: RectangularGrid):
    fungus = FungusCellList(grid=grid)
    yield fungus


@fixture
def populated_fungus(fungus_list: FungusCellList, grid: RectangularGrid):
    points = []
    for i in range(int(grid.x[1]), int(grid.x[6]), 10):
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


# tests


def test_recruit_new(neutrophil_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    rec_rate_ph = 2
    granule_count = 5
    neutropenic = False
    previous_time = 1


    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5, 5, 5] = 2

    # test correct location recruitment
    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    vox = grid.get_voxel(neutrophil_list[-1]['point'])

    assert len(neutrophil_list) == 2
    assert vox.x == 5 and vox.y == 5 and vox.z == 5

    # test recruit none due to below threshold
    rec_r = 20
    p_rec_r = 1.0
    rec_rate_ph = 2

    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    assert len(neutrophil_list) == 2

def test_recruit_new_not_blood(neutrophil_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    rec_rate_ph = 2
    granule_count = 5
    neutropenic = False
    previous_time = 1


    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5, 5, 5] = 2
    tissue[5, 5, 5] = 99

    # test correct location recruitment
    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    assert len(neutrophil_list) == 0

def test_recruit_new_neutopenic_day_2(neutrophil_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    rec_rate_ph = 6
    granule_count = 5
    neutropenic = True
    previous_time = 50 # between days 2 -4


    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5, 5, 5] = 2

    # test recruit less due to neutropenic
    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    assert len(neutrophil_list) == 0

    # test correct location recruitment
    neutropenic = False

    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    vox = grid.get_voxel(neutrophil_list[-1]['point'])
    assert vox.x == 5 and vox.y == 5 and vox.z == 5
    assert len(neutrophil_list) == 6

def test_recruit_new_neutopenic_day_3(neutrophil_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    rec_rate_ph = 6
    granule_count = 5
    neutropenic = True
    previous_time = 72 # between days 2 -4


    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5, 5, 5] = 2

    # test recruit less due to neutropenic
    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    

    assert len(neutrophil_list) == 3

    # test correct location recruitment
    neutropenic = False

    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    vox = grid.get_voxel(neutrophil_list[-1]['point'])
    assert vox.x == 5 and vox.y == 5 and vox.z == 5
    assert len(neutrophil_list) == 9# 6 + 3

def test_recruit_new_multiple_locations(
    neutrophil_list: NeutrophilCellList, tissue, grid: RectangularGrid, cyto
):
    rec_r = 2
    rec_rate_ph = 50
    granule_count = 5
    neutropenic = False
    previous_time = 1

    cyto[5, 5, 5] = 2
    cyto[4, 5, 5] = 2

    neutrophil_list.recruit_new(
            rec_rate_ph, 
            rec_r, 
            granule_count, 
            neutropenic, 
            previous_time, 
            grid, 
            tissue, 
            cyto)

    assert len(neutrophil_list) == 50

    for cell in neutrophil_list.cell_data:
         vox = grid.get_voxel(cell['point'])
         assert vox.z == 5 and vox.y == 5 and vox.x in [4, 5]

def test_absorb_cytokines(neutrophil_list:NeutrophilCellList, cyto, grid: RectangularGrid):
    cyto[1, 2, 3] = 64

    neutrophil_list.append(
       NeutrophilCellData.create_cell(
           point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
           status=NeutrophilCellData.Status.RESTING,
           granule_count=5)
    )

    assert len(neutrophil_list) == 1
    assert cyto[1, 2, 3] == 64

    n_abs = 0.5
    neutrophil_list.absorb_cytokines(n_abs, cyto, grid)
    assert cyto[1, 2, 3] == 32

    neutrophil_list.absorb_cytokines(n_abs, cyto, grid)
    assert cyto[1, 2, 3] == 16

    neutrophil_list.append(
       NeutrophilCellData.create_cell(
           point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
           status=NeutrophilCellData.Status.RESTING,
           granule_count=5)
    )

    neutrophil_list.absorb_cytokines(n_abs, cyto, grid)
    assert cyto[1, 2, 3] == 4

def test_produce_cytokines_0(
    neutrophil_list: NeutrophilCellList,
    grid: RectangularGrid,
    populated_fungus: FungusCellList,
    cyto,
):
    n_det = 0
    n_n = 10

    assert cyto[3, 3, 3] == 0

    neutrophil_list.append(
       NeutrophilCellData.create_cell(
           point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
           status=NeutrophilCellData.Status.RESTING,
           granule_count=5)
    )

    vox = grid.get_voxel(neutrophil_list[0]['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    neutrophil_list.produce_cytokines(n_det, n_n, grid, populated_fungus, cyto)

    assert cyto[3, 3, 3] == 10


def test_produce_cytokines_n(
    neutrophil_list: NeutrophilCellList,
    grid: RectangularGrid,
    populated_fungus: FungusCellList,
    cyto,
):
    n_n = 10
    neutrophil_list.append(
       NeutrophilCellData.create_cell(
           point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
           status=NeutrophilCellData.Status.RESTING,
           granule_count=5)
    )
    
    vox = grid.get_voxel(neutrophil_list[0]['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    for cell in populated_fungus.cell_data:
        vox = grid.get_voxel(cell['point'])
    assert vox.z in [1, 2, 3, 4, 5] and vox.y in [1, 2, 3, 4, 5] and vox.x in [1, 2, 3, 4, 5]

    # 1
    n_det = 1
    assert cyto[3, 3, 3] == 0

    neutrophil_list.produce_cytokines(n_det, n_n, grid, populated_fungus, cyto)
    assert cyto[3, 3, 3] == 30

    # 2
    n_det = 2
    cyto[3, 3, 3] = 0

    neutrophil_list.produce_cytokines(n_det, n_n, grid, populated_fungus, cyto)
    assert cyto[3, 3, 3] == 50