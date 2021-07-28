import numpy as np
from pytest import fixture

from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.oldmodules.fungus import FungusCellData, FungusCellList


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


def test_initialize_spores_epi(fungus_list, epi_geometry):
    fungus_list.initialize_spores(epi_geometry, 5)
    cells = fungus_list.cell_data
    assert len(cells) == 5
    assert (cells['status'] == FungusCellData.Status.RESTING).all()
    assert (cells['form'] == FungusCellData.Form.CONIDIA).all()
    assert (cells['iron'] == 0).all()


def test_initialize_spores_air(fungus_list, air_geometry):
    fungus_list.initialize_spores(air_geometry, 5)
    cells = fungus_list.cell_data
    assert len(cells) == 0
    assert (cells['status'] == FungusCellData.Status.RESTING).all()
    assert (cells['form'] == FungusCellData.Form.CONIDIA).all()
    assert (cells['iron'] == 0).all()


def test_iron_uptake(populated_fungus: FungusCellList, iron):
    iron_min = 5
    iron_max = 100
    iron_absorb = 0.5
    assert iron[5, 5, 5] == 10

    populated_fungus.iron_uptake(iron, iron_max, iron_min, iron_absorb)

    for cell in populated_fungus.cell_data:
        assert cell['iron'] == 5


def test_age(populated_fungus):
    cells = populated_fungus.cell_data
    cells['dead'][0] = True
    for _ in range(10):
        populated_fungus.age()

    assert cells['iteration'][0] == 0
    assert (cells['iteration'][1:] == 10).all()


def test_kill(populated_fungus):
    cells = populated_fungus.cell_data
    cells['health'][0] = 0
    cells['point'][1][0] = -1
    populated_fungus.kill()

    assert (cells['dead'][0:2]).all()
    assert not (cells['dead'][2:]).any()
    assert (cells['status'][0:2] == FungusCellData.Status.DEAD).all()
    assert (cells['status'][2:] != FungusCellData.Status.DEAD).all()


def test_change_state(populated_fungus):
    p_internal_swell = 0
    rest_time = 10
    swell_time = 10
    cells = populated_fungus.cell_data
    for _ in range(10):
        populated_fungus.age()
    cells['form'][0:2] = FungusCellData.Form.CONIDIA
    cells['status'][0] = FungusCellData.Status.RESTING
    cells['status'][1] = FungusCellData.Status.SWOLLEN
    cells['internalized'] = False
    populated_fungus.change_status(p_internal_swell, rest_time, swell_time)
    assert (cells['iteration'][0:2] == 0).all()
    assert cells['status'][0] == FungusCellData.Status.SWOLLEN
    assert cells['status'][1] == FungusCellData.Status.GERMINATED
    assert (cells['status'][2:] == FungusCellData.Status.GROWABLE).all()


def test_internal_change_state(populated_fungus):
    rest_time = 10
    swell_time = 11
    cells = populated_fungus.cell_data
    for _ in range(10):
        populated_fungus.age()
    cells['form'][0:2] = FungusCellData.Form.CONIDIA
    cells['status'][0] = FungusCellData.Status.RESTING
    cells['status'][1] = FungusCellData.Status.SWOLLEN
    cells['internalized'] = True
    populated_fungus.change_status(1, rest_time, swell_time)
    assert cells['iteration'][0] == 0
    assert cells['iteration'][1] == 10
    assert cells['status'][0] == FungusCellData.Status.SWOLLEN
    assert cells['status'][1] == FungusCellData.Status.SWOLLEN
    populated_fungus.age()
    populated_fungus.change_status(1, rest_time, swell_time)
    assert cells['status'][1] == FungusCellData.Status.GERMINATED


def test_internal_change_state_zero_swallow(populated_fungus):
    rest_time = 10
    swell_time = 10
    cells = populated_fungus.cell_data
    for _ in range(10):
        populated_fungus.age()
    cells['form'][0:2] = FungusCellData.Form.CONIDIA
    cells['status'][0] = FungusCellData.Status.RESTING
    cells['status'][1] = FungusCellData.Status.SWOLLEN
    cells['internalized'] = True
    populated_fungus.change_status(0, rest_time, swell_time)
    assert cells['iteration'][0] != 0
    assert cells['iteration'][1] == 0
    assert cells['status'][0] == FungusCellData.Status.RESTING
    assert cells['status'][1] == FungusCellData.Status.GERMINATED


def test_grow_conidia(populated_fungus):
    cells = populated_fungus.cell_data
    cells['form'] = FungusCellData.Form.CONIDIA
    cells['status'] = FungusCellData.Status.GERMINATED
    cells['iteration'] = 10
    populated_fungus.grow_hyphae(iron_min_grow=1, grow_time=5, p_branch=1, spacing=1)
    cells = populated_fungus.cell_data
    assert len(cells) == 10
    assert (cells['status'][:5] == FungusCellData.Status.GROWN).all()
    assert (cells['form'][:5] == FungusCellData.Form.HYPHAE).all()
    assert (cells['status'][5:] == FungusCellData.Status.GROWABLE).all()


def test_grow_hyphae_not_branch(populated_fungus):
    cells = populated_fungus.cell_data
    cells['form'] = FungusCellData.Form.HYPHAE
    cells['status'] = FungusCellData.Status.GROWABLE
    cells['iteration'] = 10
    cells['iron'] = 30
    populated_fungus.grow_hyphae(iron_min_grow=1, grow_time=5, p_branch=0, spacing=1)
    cells = populated_fungus.cell_data

    assert len(cells) == 10
    assert (cells['iron'] == 15).all()
    assert (cells['status'][:5] == FungusCellData.Status.GROWN).all()
    assert (cells['status'][5:] == FungusCellData.Status.GROWABLE).all()


def test_grow_hyphae_branch(populated_fungus):
    cells = populated_fungus.cell_data
    cells['form'] = FungusCellData.Form.HYPHAE
    cells['status'] = FungusCellData.Status.GROWABLE
    cells['iteration'] = 10
    cells['iron'] = 30
    populated_fungus.grow_hyphae(iron_min_grow=1, grow_time=5, p_branch=1, spacing=1)
    cells = populated_fungus.cell_data

    assert len(cells) == 15
    assert (cells['iron'] == 10).all()
    assert (cells['status'][:5] == FungusCellData.Status.GROWN).all()
    assert (cells['status'][5:] == FungusCellData.Status.GROWABLE).all()


def test_grow_hyphae_internal(populated_fungus):
    cells = populated_fungus.cell_data
    cells['form'] = FungusCellData.Form.HYPHAE
    cells['status'] = FungusCellData.Status.GROWABLE
    cells['iteration'] = 10
    cells['iron'] = 30
    cells['internalized'] = True
    populated_fungus.grow_hyphae(iron_min_grow=1, grow_time=5, p_branch=0, spacing=1)
    cells = populated_fungus.cell_data

    assert len(cells) == 5
    assert (cells['iron'] == 30).all()


def test_spawn_spores(populated_fungus):
    points = np.zeros((2, 3))
    points[0] = Point(x=5, y=5, z=5)
    points[1] = Point(x=3, y=3, z=3)
    populated_fungus.spawn_spores(points)
    cells = populated_fungus.cell_data

    assert len(cells) == 7
    assert (cells['status'][5:] == FungusCellData.Status.RESTING).all()
    assert (cells['form'][5:] == FungusCellData.Form.CONIDIA).all()
