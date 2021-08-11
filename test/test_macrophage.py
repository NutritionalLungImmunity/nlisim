import numpy as np
from pytest import fixture

from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.oldmodules.fungus import FungusCellData, FungusCellList
from nlisim.oldmodules.macrophage import MacrophageCellData, MacrophageCellList


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
def populated_macrophage(macrophage_list: MacrophageCellList, grid: RectangularGrid):
    macrophage_list.append(
        MacrophageCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
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


def test_recruit_new(macrophage_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    p_rec_r = 1.0
    rec_rate_ph = 2

    # no cytokines
    assert cyto[1, 2, 3] == 0

    cyto[1, 2, 3] = 2

    # test correct location recruitment
    macrophage_list.recruit_new(rec_rate_ph, rec_r, p_rec_r, tissue, grid, cyto)
    vox = grid.get_voxel(macrophage_list[-1]['point'])

    assert len(macrophage_list) == 2
    assert vox.x == 3 and vox.y == 2 and vox.z == 1

    # test recruit none due to below threshold
    rec_r = 20
    p_rec_r = 1.0
    rec_rate_ph = 2
    macrophage_list.recruit_new(rec_rate_ph, rec_r, p_rec_r, tissue, grid, cyto)

    assert len(macrophage_list) == 2


def test_recruit_new_multiple_locations(
    macrophage_list: MacrophageCellList, tissue, grid: RectangularGrid, cyto
):

    rec_r = 2
    p_rec_r = 1.0
    rec_rate_ph = 50

    cyto[1, 2, 3] = 2
    cyto[4, 5, 6] = 2

    macrophage_list.recruit_new(rec_rate_ph, rec_r, p_rec_r, tissue, grid, cyto)

    assert len(macrophage_list) == 50

    for cell in macrophage_list.cell_data:
        vox = grid.get_voxel(cell['point'])
        assert vox.x in [3, 6] and vox.y in [2, 5] and vox.z in [1, 4]


def test_absorb_cytokines(macrophage_list: MacrophageCellList, cyto, grid: RectangularGrid):
    cyto[1, 2, 3] = 64

    macrophage_list.append(
        MacrophageCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
        )
    )

    assert len(macrophage_list) == 1
    assert cyto[1, 2, 3] == 64

    m_abs = 0.5
    macrophage_list.absorb_cytokines(m_abs, cyto, grid)
    assert cyto[1, 2, 3] == 32

    macrophage_list.absorb_cytokines(m_abs, cyto, grid)
    assert cyto[1, 2, 3] == 16

    macrophage_list.append(
        MacrophageCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
        )
    )

    macrophage_list.absorb_cytokines(m_abs, cyto, grid)
    assert cyto[1, 2, 3] == 4


def test_produce_cytokines_0(
    macrophage_list: MacrophageCellList,
    grid: RectangularGrid,
    populated_fungus: FungusCellList,
    cyto,
):
    m_det = 0
    m_n = 10

    assert cyto[3, 3, 3] == 0

    macrophage_list.append(
        MacrophageCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
        )
    )

    vox = grid.get_voxel(macrophage_list[0]['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    macrophage_list.produce_cytokines(m_det, m_n, grid, populated_fungus, cyto)

    assert cyto[3, 3, 3] == 10


def test_produce_cytokines_n(
    macrophage_list: MacrophageCellList,
    grid: RectangularGrid,
    populated_fungus: FungusCellList,
    cyto,
):
    m_n = 10
    macrophage_list.append(
        MacrophageCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
        )
    )
    vox = grid.get_voxel(macrophage_list[0]['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    for cell in populated_fungus.cell_data:
        vox = grid.get_voxel(cell['point'])
    assert vox.z in [1, 2, 3, 4, 5] and vox.y in [1, 2, 3, 4, 5] and vox.x in [1, 2, 3, 4, 5]

    # 1
    m_det = 1
    assert cyto[3, 3, 3] == 0

    macrophage_list.produce_cytokines(m_det, m_n, grid, populated_fungus, cyto)
    assert cyto[3, 3, 3] == 30

    # 2
    m_det = 2
    cyto[3, 3, 3] = 0

    macrophage_list.produce_cytokines(m_det, m_n, grid, populated_fungus, cyto)
    assert cyto[3, 3, 3] == 50


def test_move_1(
    populated_macrophage: MacrophageCellList, grid: RectangularGrid, cyto, tissue, fungus_list
):
    rec_r = 10

    cell = populated_macrophage[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10

    populated_macrophage.move(rec_r, grid, cyto, tissue, fungus_list)

    vox = grid.get_voxel(cell['point'])
    assert vox.z == 4 and vox.y == 3 and vox.x == 3


def test_move_n(
    populated_macrophage: MacrophageCellList, grid: RectangularGrid, cyto, tissue, fungus_list
):
    rec_r = 10

    cell = populated_macrophage[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10
    cyto[3, 4, 3] = 10
    cyto[4, 4, 3] = 10

    populated_macrophage.move(rec_r, grid, cyto, tissue, fungus_list)

    vox = grid.get_voxel(cell['point'])
    assert vox.z in [3, 4] and vox.y in [3, 4] and vox.x == 3


def test_move_air(
    populated_macrophage: MacrophageCellList, grid: RectangularGrid, cyto, tissue, fungus_list
):
    rec_r = 10

    cell = populated_macrophage[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10
    cyto[3, 4, 3] = 15
    tissue[3, 4, 3] = 0  # air

    populated_macrophage.move(rec_r, grid, cyto, tissue, fungus_list)

    vox = grid.get_voxel(cell['point'])
    assert vox.z == 4 and vox.y == 3 and vox.x == 3


# internalize_conidia
def test_internalize_conidia_none(
    populated_macrophage: MacrophageCellList,
    grid: RectangularGrid,
    populated_fungus: FungusCellList,
):
    m_det = 0

    cell = populated_macrophage[0]
    vox = grid.get_voxel(cell['point'])
    assert len(populated_fungus.get_cells_in_voxel(vox)) == 1

    populated_macrophage.internalize_conidia(m_det, 50, 1, grid, populated_fungus)

    assert populated_macrophage.len_phagosome(0) == 0
    for v in cell['phagosome']:
        assert v == -1


def test_internalize_conidia_0(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    m_det = 0

    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    vox = grid.get_voxel(macrophage_list[0]['point'])

    assert len(fungus_list.get_cells_in_voxel(vox)) == 1

    f_index = int(fungus_list.get_cells_in_voxel(vox))  # 0
    assert f_index == 0

    fungus_list[f_index]['form'] = FungusCellData.Form.CONIDIA
    fungus_list[f_index]['status'] = FungusCellData.Status.RESTING

    macrophage_list.internalize_conidia(m_det, 50, 1, grid, fungus_list)

    assert grid.get_voxel(fungus_list[f_index]['point']) == vox
    assert fungus_list.cell_data['internalized'][f_index]
    assert macrophage_list.len_phagosome(0) == 1


def test_internalize_conidia_n(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):

    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))

    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    point = Point(x=45, y=35, z=35)
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    point = Point(x=55, y=35, z=35)
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    # internalize some not all
    macrophage_list.internalize_conidia(1, 50, 1, grid, fungus_list)

    assert fungus_list.cell_data['internalized'][0]
    assert fungus_list.cell_data['internalized'][2]
    assert macrophage_list.len_phagosome(0) == 4

    # internalize all
    macrophage_list.internalize_conidia(2, 50, 1, grid, fungus_list)

    assert fungus_list.cell_data['internalized'][5]
    assert macrophage_list.len_phagosome(0) == 6


def test_internalize_and_move(
    macrophage_list: MacrophageCellList,
    grid: RectangularGrid,
    fungus_list: FungusCellList,
    cyto,
    tissue,
):
    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))

    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    macrophage_list.internalize_conidia(1, 50, 1, grid, fungus_list)

    assert fungus_list.cell_data['internalized'][0]
    assert fungus_list.cell_data['internalized'][1]
    assert macrophage_list.len_phagosome(0) == 2

    rec_r = 10

    cell = macrophage_list[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10

    macrophage_list.move(rec_r, grid, cyto, tissue, fungus_list)

    vox = grid.get_voxel(cell['point'])
    assert vox.z == 4 and vox.y == 3 and vox.x == 3

    for f in fungus_list:
        vox = grid.get_voxel(f['point'])
        assert vox.z == 4 and vox.y == 3 and vox.x == 3


# damage_conidia(state, previous_time)
def test_damage_conidia(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    macrophage_list.internalize_conidia(1, 50, 1, grid, fungus_list)

    macrophage_list.damage_conidia(kill, t, health, fungus_list)

    assert fungus_list.cell_data['health'][0] == 50

    macrophage_list.damage_conidia(kill, t, health, fungus_list)

    assert fungus_list.cell_data['health'][0] == 0


def test_kill_macrophage(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    # should release all conidia

    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    macrophage_list.internalize_conidia(1, 50, 1, grid, fungus_list)
    assert fungus_list.cell_data['internalized'][0]

    # simulate death
    macrophage_list.clear_all_phagosome(0, fungus_list)

    assert not fungus_list.cell_data['internalized'][0]
    assert macrophage_list.len_phagosome(0) == 0


def test_sporeless1(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    # should release all conidia

    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    if len(fungus_list.alive(fungus_list.cell_data['form'] == FungusCellData.Form.CONIDIA)) == 0:
        macrophage_list.remove_if_sporeless(0.1)
    assert not macrophage_list.cell_data[0]['dead']

    macrophage_list.remove_if_sporeless(0.1)
    assert macrophage_list.cell_data[0]['dead']


def test_sporeless0(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    # should release all conidia

    point = Point(x=35, y=35, z=35)
    macrophage_list.append(MacrophageCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    if len(fungus_list.alive(fungus_list.cell_data['form'] == FungusCellData.Form.CONIDIA)) == 0:
        macrophage_list.remove_if_sporeless(0.1)
    assert not macrophage_list.cell_data[0]['dead']

    macrophage_list.remove_if_sporeless(0.1)
    assert macrophage_list.cell_data[0]['dead']

    macrophage_list.remove_if_sporeless(0.1)
    assert macrophage_list.cell_data[0]['dead']


def test_sporeless30(
    macrophage_list: MacrophageCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    # should release all conidia

    point = Point(x=35, y=35, z=35)
    for _ in range(0, 30):
        macrophage_list.append(MacrophageCellData.create_cell(point=point))

    fungus_list.append(
        FungusCellData.create_cell(
            point=point, status=FungusCellData.Status.RESTING, form=FungusCellData.Form.HYPHAE
        )
    )

    macrophage_list.remove_if_sporeless(0.3)
    assert len(macrophage_list.alive()) < 30
