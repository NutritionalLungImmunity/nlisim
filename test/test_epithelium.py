import numpy as np
from pytest import fixture

from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.oldmodules.epithelium import EpitheliumCellData, EpitheliumCellList
from nlisim.oldmodules.fungus import FungusCellData, FungusCellList


@fixture
def iron():
    # a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(10)
    yield i


@fixture
def n_cyto():
    # a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(0)
    yield i


@fixture
def m_cyto():
    # a 10 x 10 x 10 grid with 10 iron
    i = np.empty((10, 10, 10))
    i.fill(0)
    yield i


@fixture
def tissue():
    # a 10 x 10 x 10 grid of blood
    t = np.empty((10, 10, 10))
    t.fill(3)
    yield t


@fixture
def epithelium_list(grid: RectangularGrid):
    epithelium = EpitheliumCellList(grid=grid)
    yield epithelium


@fixture
def populated_epithelium(epithelium_list: EpitheliumCellList, grid: RectangularGrid):
    epithelium_list.append(
        EpitheliumCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
        )
    )

    yield epithelium_list


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

# internalize_conidia
def test_internalize_conidia_none(
    populated_epithelium: EpitheliumCellList,
    grid: RectangularGrid,
    fungus_list: FungusCellList,
):
    cell = populated_epithelium[0]
    vox = grid.get_voxel(cell['point'])
    assert len(fungus_list.get_cells_in_voxel(vox)) == 0

    populated_epithelium.internalize_conidia(0, 10, 1, grid, fungus_list)

    assert populated_epithelium.len_phagosome(0) == 0
    for v in cell['phagosome']:
        assert v == -1


def test_internalize_conidia_1(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    vox = grid.get_voxel(epithelium_list[0]['point'])

    epithelium_list.internalize_conidia(0, 10, 1, grid, fungus_list)

    assert grid.get_voxel(fungus_list[0]['point']) == vox
    assert epithelium_list.len_phagosome(0) == 1
    assert 0 in epithelium_list[0]['phagosome']


def test_internalize_conidia_2(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    vox = grid.get_voxel(epithelium_list[0]['point'])

    epithelium_list.internalize_conidia(0, 10, 1, grid, fungus_list)

    assert grid.get_voxel(fungus_list[0]['point']) == vox
    assert epithelium_list.len_phagosome(0) == 2
    assert 0 in epithelium_list[0]['phagosome']
    assert 1 in epithelium_list[0]['phagosome']


def test_internalize_conidia_2b(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    vox = grid.get_voxel(epithelium_list[0]['point'])
    fungus_list[0]['internalized'] = True  # say by macrophage

    epithelium_list.internalize_conidia(0, 10, 1, grid, fungus_list)

    assert grid.get_voxel(fungus_list[0]['point']) == vox
    assert epithelium_list.len_phagosome(0) == 1
    assert 0 not in epithelium_list[0]['phagosome']
    assert 1 in epithelium_list[0]['phagosome']


def test_internalize_conidia_max(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    max_spores = 10
    epithelium_list[0]['phagosome'][:max_spores] = 99  # artificially fill

    epithelium_list.internalize_conidia(0, max_spores, 1, grid, fungus_list)

    assert epithelium_list.len_phagosome(0) == max_spores
    assert 0 not in epithelium_list[0]['phagosome']
    assert not fungus_list[0]['internalized']


def test_dead_conidia_1(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    epithelium_list.internalize_conidia(0, 10, 1, grid, fungus_list)
    fungus_list[0]['dead'] = True  # simulate killing

    epithelium_list.remove_dead_fungus(fungus_list)

    assert epithelium_list.len_phagosome(0) == 0
    assert 0 not in epithelium_list[0]['phagosome']


def test_produce_cytokines_0(
    epithelium_list: EpitheliumCellList,
    grid: RectangularGrid,
    fungus_list: FungusCellList,
    m_cyto,
    n_cyto,
):
    s_det = 0
    h_det = 0
    cyto_rate = 10

    assert m_cyto[3, 3, 3] == 0
    assert n_cyto[3, 3, 3] == 0

    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))

    fungus_list.append(
        FungusCellData.create_cell(
            point=point,
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.CONIDIA,
            iron=0,
            mobile=False,
        )
    )

    # fungus is not swollen or germinated
    epithelium_list.cytokine_update(s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus_list, grid)

    assert m_cyto[3, 3, 3] == 0
    assert n_cyto[3, 3, 3] == 0

    fungus_list[0]['status'] = FungusCellData.Status.SWOLLEN

    epithelium_list.cytokine_update(s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus_list, grid)

    assert m_cyto[3, 3, 3] == 10
    assert n_cyto[3, 3, 3] == 10


def test_produce_cytokines_0b(
    epithelium_list: EpitheliumCellList,
    grid: RectangularGrid,
    fungus_list: FungusCellList,
    m_cyto,
    n_cyto,
):
    s_det = 0
    h_det = 0
    cyto_rate = 10

    assert m_cyto[3, 3, 3] == 0
    assert n_cyto[3, 3, 3] == 0

    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))

    fungus_list.append(
        FungusCellData.create_cell(
            point=point,
            status=FungusCellData.Status.SWOLLEN,
            form=FungusCellData.Form.CONIDIA,
            iron=0,
            mobile=False,
        )
    )

    epithelium_list.cytokine_update(s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus_list, grid)

    assert m_cyto[3, 3, 3] == 10
    assert n_cyto[3, 3, 3] == 10

    fungus_list.append(
        FungusCellData.create_cell(
            point=point,
            status=FungusCellData.Status.GROWABLE,
            form=FungusCellData.Form.HYPHAE,
            iron=0,
            mobile=False,
        )
    )

    epithelium_list.cytokine_update(s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus_list, grid)

    assert m_cyto[3, 3, 3] == 20
    assert n_cyto[3, 3, 3] == 30


def test_produce_cytokines_2(
    epithelium_list: EpitheliumCellList,
    grid: RectangularGrid,
    fungus_list: FungusCellList,
    m_cyto,
    n_cyto,
):
    s_det = 1
    h_det = 2
    cyto_rate = 10

    assert m_cyto[3, 3, 3] == 0
    assert n_cyto[3, 3, 3] == 0

    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))

    spoint = Point(x=15, y=35, z=35)
    fungus_list.append(
        FungusCellData.create_cell(
            point=spoint,
            status=FungusCellData.Status.SWOLLEN,
            form=FungusCellData.Form.CONIDIA,
            iron=0,
            mobile=False,
        )
    )

    epithelium_list.cytokine_update(s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus_list, grid)

    assert m_cyto[3, 3, 3] == 0
    assert n_cyto[3, 3, 3] == 0

    fungus_list.append(
        FungusCellData.create_cell(
            point=spoint,
            status=FungusCellData.Status.GROWABLE,
            form=FungusCellData.Form.HYPHAE,
            iron=0,
            mobile=False,
        )
    )

    epithelium_list.cytokine_update(s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus_list, grid)

    assert m_cyto[3, 3, 3] == 0
    assert n_cyto[3, 3, 3] == 10


def test_damage_conidia(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    epithelium_list.internalize_conidia(0, 10, 1, grid, fungus_list)

    epithelium_list.damage(kill, t, health, fungus_list)

    assert fungus_list.cell_data['health'][0] == 50

    epithelium_list.damage(kill, t, health, fungus_list)

    assert fungus_list.cell_data['health'][0] == 0


def test_kill_epithelium(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    # should release all conidia

    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    fungus_list.cell_data['internalized'][0] = True
    epithelium_list[0]['phagosome'][0] = 0  # internalized

    epithelium_list.die_by_germination(fungus_list)
    assert fungus_list.cell_data['internalized'][0]
    assert epithelium_list.len_phagosome(0) == 1

    fungus_list.cell_data['status'][0] = FungusCellData.Status.GERMINATED

    epithelium_list.die_by_germination(fungus_list)

    assert not fungus_list.cell_data['internalized'][0]
    assert epithelium_list.len_phagosome(0) == 0


def test_kill_epithelium_n(
    epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
):
    # should release all conidia

    point = Point(x=35, y=35, z=35)
    epithelium_list.append(EpitheliumCellData.create_cell(point=point))
    for _ in range(0, 10):
        fungus_list.append(
            FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
        )

    epithelium_list.internalize_conidia(0, 10, 1, grid, fungus_list)

    epithelium_list.die_by_germination(fungus_list)
    for i in range(0, 10):
        assert fungus_list.cell_data['internalized'][i]
    assert epithelium_list.len_phagosome(0) == 10

    fungus_list.cell_data['status'][6] = FungusCellData.Status.GERMINATED

    epithelium_list.die_by_germination(fungus_list)

    for i in range(0, 10):
        assert not fungus_list.cell_data['internalized'][i]
    assert epithelium_list.len_phagosome(0) == 0
