import numpy as np
from pytest import fixture

from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.oldmodules.fungus import FungusCellData, FungusCellList
from nlisim.oldmodules.neutrophil import NeutrophilCellData, NeutrophilCellList


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
        NeutrophilCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),
            status=NeutrophilCellData.Status.NONGRANULATING,
            granule_count=5,
        )
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
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    vox = grid.get_voxel(neutrophil_list[-1]['point'])

    assert len(neutrophil_list) == 2
    # TODO: why are x=y=z? what is the point of this?
    assert vox.x == 5 and vox.y == 5 and vox.z == 5

    # test recruit none due to below threshold
    rec_r = 20
    rec_rate_ph = 2

    neutrophil_list.recruit_new(
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

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
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    assert len(neutrophil_list) == 0


def test_recruit_new_neutopenic_day_2(neutrophil_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    rec_rate_ph = 6
    granule_count = 5
    neutropenic = True
    previous_time = 15  # between days 1 - 3

    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5, 5, 5] = 2

    # test recruit less due to neutropenic
    neutrophil_list.recruit_new(
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    assert len(neutrophil_list) == 0

    # test correct location recruitment
    neutropenic = False

    neutrophil_list.recruit_new(
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    vox = grid.get_voxel(neutrophil_list[-1]['point'])
    assert vox.x == 5 and vox.y == 5 and vox.z == 5
    assert len(neutrophil_list) == 6


def test_recruit_new_neutopenic_day_3(neutrophil_list, tissue, grid: RectangularGrid, cyto):
    rec_r = 2
    rec_rate_ph = 6
    granule_count = 5
    neutropenic = True
    previous_time = 64  # between days 2 -4

    # no cytokines
    assert cyto[5, 5, 5] == 0

    cyto[5, 5, 5] = 2

    # test recruit less due to neutropenic
    neutrophil_list.recruit_new(
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    assert len(neutrophil_list) == 6

    # test correct location recruitment
    neutropenic = False

    neutrophil_list.recruit_new(
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    vox = grid.get_voxel(neutrophil_list[-1]['point'])
    assert vox.x == 5 and vox.y == 5 and vox.z == 5
    assert len(neutrophil_list) == 12


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
        rec_rate_ph, rec_r, granule_count, neutropenic, previous_time, grid, tissue, cyto
    )

    assert len(neutrophil_list) == 50

    for cell in neutrophil_list.cell_data:
        vox = grid.get_voxel(cell['point'])
        assert vox.x == 5 and vox.y == 5 and vox.z in [4, 5]


def test_absorb_cytokines(neutrophil_list: NeutrophilCellList, cyto, grid: RectangularGrid):
    cyto[1, 2, 3] = 64

    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=Point(x=grid.x[3], y=grid.y[2], z=grid.z[1]),
            status=NeutrophilCellData.Status.NONGRANULATING,
            granule_count=5,
        )
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
            status=NeutrophilCellData.Status.NONGRANULATING,
            granule_count=5,
        )
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
            status=NeutrophilCellData.Status.NONGRANULATING,
            granule_count=5,
        )
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
            status=NeutrophilCellData.Status.NONGRANULATING,
            granule_count=5,
        )
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


def test_move_1(populated_neutrophil: NeutrophilCellList, grid: RectangularGrid, cyto, tissue):
    rec_r = 10

    cell = populated_neutrophil[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10

    populated_neutrophil.move(rec_r, grid, cyto, tissue)

    cell = populated_neutrophil[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 4 and vox.y == 3 and vox.x == 3


def test_move_n(populated_neutrophil: NeutrophilCellList, grid: RectangularGrid, cyto, tissue):
    rec_r = 10

    cell = populated_neutrophil[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10
    cyto[3, 4, 3] = 10
    cyto[4, 4, 3] = 10

    populated_neutrophil.move(rec_r, grid, cyto, tissue)

    cell = populated_neutrophil[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z in [3, 4] and vox.y in [3, 4] and vox.x == 3


def test_move_air(populated_neutrophil: NeutrophilCellList, grid: RectangularGrid, cyto, tissue):
    rec_r = 10

    cell = populated_neutrophil[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 3 and vox.y == 3 and vox.x == 3

    assert cyto.all() == 0
    cyto[4, 3, 3] = 10
    cyto[3, 4, 3] = 15
    tissue[3, 4, 3] = 0  # air

    populated_neutrophil.move(rec_r, grid, cyto, tissue)

    cell = populated_neutrophil[0]
    vox = grid.get_voxel(cell['point'])
    assert vox.z == 4 and vox.y == 3 and vox.x == 3


def test_damage_hyphae_conidia(
    neutrophil_list: NeutrophilCellList, grid: RectangularGrid, fungus_list: FungusCellList, iron
):
    n_det = 1
    n_kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=5
        )
    )

    # conidia
    fungus_list.append(
        FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
    )

    neutrophil_list.damage_hyphae(n_det, n_kill, t, health, grid, fungus_list, iron)

    assert fungus_list[0]['health'] == 100
    assert neutrophil_list[0]['granule_count'] == 5
    assert neutrophil_list[0]['status'] == NeutrophilCellData.Status.NONGRANULATING


def test_damage_hyphae_0(
    neutrophil_list: NeutrophilCellList, grid: RectangularGrid, fungus_list: FungusCellList, iron
):
    n_det = 0
    n_kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=5
        )
    )

    # hyphae
    fungus_list.append(
        FungusCellData.create_cell(
            point=point, status=FungusCellData.Status.RESTING, form=FungusCellData.Form.HYPHAE
        )
    )

    neutrophil_list.damage_hyphae(n_det, n_kill, t, health, grid, fungus_list, iron)

    assert fungus_list[0]['health'] == 50
    assert neutrophil_list[0]['granule_count'] == 4
    assert neutrophil_list[0]['status'] == NeutrophilCellData.Status.GRANULATING

    vox = grid.get_voxel(neutrophil_list[0]['point'])
    assert iron[vox.z, vox.y, vox.x] == 0


def test_damage_hyphae_n_det_1(
    neutrophil_list: NeutrophilCellList, grid: RectangularGrid, fungus_list: FungusCellList, iron
):
    n_det = 1
    n_kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=5
        )
    )

    fungus_list.append(
        FungusCellData.create_cell(
            point=point, status=FungusCellData.Status.RESTING, form=FungusCellData.Form.HYPHAE
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=45, y=35, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=25, y=35, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=45, y=45, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=25, y=25, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )

    neutrophil_list.damage_hyphae(n_det, n_kill, t, health, grid, fungus_list, iron)

    assert fungus_list[0]['health'] == 50
    assert fungus_list[1]['health'] == 50
    assert fungus_list[2]['health'] == 50
    assert neutrophil_list[0]['granule_count'] == 0
    assert neutrophil_list[0]['status'] == NeutrophilCellData.Status.GRANULATING

    vox = grid.get_voxel(neutrophil_list[0]['point'])
    assert iron[vox.z + 0, vox.y + 0, vox.x + 0] == 0
    assert iron[vox.z + 0, vox.y + 0, vox.x + 1] == 0
    assert iron[vox.z + 0, vox.y + 0, vox.x - 1] == 0

    assert iron[vox.z + 0, vox.y + 1, vox.x + 1] == 0
    assert iron[vox.z + 0, vox.y - 1, vox.x - 1] == 0
    assert iron[vox.z + 1, vox.y + 0, vox.x + 1] == 10
    assert iron[vox.z + 1, vox.y + 1, vox.x + 1] == 10


def test_damage_hyphae_granuleless(
    neutrophil_list: NeutrophilCellList, grid: RectangularGrid, fungus_list: FungusCellList, iron
):
    n_det = 1
    n_kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=2
        )
    )

    fungus_list.append(
        FungusCellData.create_cell(
            point=point, status=FungusCellData.Status.RESTING, form=FungusCellData.Form.HYPHAE
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=45, y=35, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=25, y=35, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )

    neutrophil_list.damage_hyphae(n_det, n_kill, t, health, grid, fungus_list, iron)

    assert fungus_list[0]['health'] == 50
    # one should be 50, the other 100. It doesn't matter which is which
    assert fungus_list[1]['health'] == 100 or fungus_list[2]['health'] == 100
    assert fungus_list[1]['health'] == 50 or fungus_list[2]['health'] == 50
    assert neutrophil_list[0]['granule_count'] == 0
    assert neutrophil_list[0]['status'] == NeutrophilCellData.Status.NONGRANULATING


def test_update(
    neutrophil_list: NeutrophilCellList, grid: RectangularGrid, fungus_list: FungusCellList, iron
):
    n_det = 1
    n_kill = 2
    t = 1
    health = 100

    point = Point(x=35, y=35, z=35)
    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=2
        )
    )

    fungus_list.append(
        FungusCellData.create_cell(
            point=point, status=FungusCellData.Status.RESTING, form=FungusCellData.Form.HYPHAE
        )
    )
    fungus_list.append(
        FungusCellData.create_cell(
            point=Point(x=45, y=35, z=35),
            status=FungusCellData.Status.RESTING,
            form=FungusCellData.Form.HYPHAE,
        )
    )

    neutrophil_list.damage_hyphae(n_det, n_kill, t, health, grid, fungus_list, iron)

    assert fungus_list[0]['health'] == 50
    assert fungus_list[1]['health'] == 50
    assert neutrophil_list[0]['granule_count'] == 0
    assert neutrophil_list[0]['status'] == NeutrophilCellData.Status.GRANULATING

    neutrophil_list.update()

    assert neutrophil_list[0]['status'] == NeutrophilCellData.Status.NONGRANULATING


def test_age(
    neutrophil_list: NeutrophilCellList,
):
    age_limit = 2

    point = Point(x=35, y=35, z=35)
    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=2
        )
    )

    # age = 0
    neutrophil_list.age()
    neutrophil_list.kill_by_age(age_limit)

    assert len(neutrophil_list.alive()) == 1
    assert neutrophil_list[0]['iteration'] == 1

    neutrophil_list.append(
        NeutrophilCellData.create_cell(
            point=point, status=NeutrophilCellData.Status.NONGRANULATING, granule_count=2
        )
    )

    # age = 1, 0
    neutrophil_list.age()
    neutrophil_list.kill_by_age(age_limit)

    assert len(neutrophil_list.alive()) == 2
    assert neutrophil_list[0]['iteration'] == 2

    # age = 2, 1
    neutrophil_list.age()
    neutrophil_list.kill_by_age(age_limit)

    assert len(neutrophil_list.alive()) == 1
    assert neutrophil_list[0]['dead']
    assert neutrophil_list[0]['iteration'] == 3
    assert neutrophil_list[1]['iteration'] == 2

    # age = 2
    neutrophil_list.age()
    neutrophil_list.kill_by_age(age_limit)

    assert len(neutrophil_list.alive()) == 0
