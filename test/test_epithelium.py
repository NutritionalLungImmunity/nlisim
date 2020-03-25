import numpy as np
from pytest import fixture

from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.modules.fungus import (
    FungusCellData,
    FungusCellList,
)
from simulation.modules.epithelium import (
    EpitheliumCellData,
    EpitheliumCellList,
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
    t.fill(3)
    yield t


@fixture
def epithelium_list(grid: RectangularGrid):
    epithelium = EpitheliumCellList(grid=grid)
    yield epithelium


@fixture
def populated_epithelium(epithelium_list: EpitheliumCellList, grid: RectangularGrid):
    epithelium_list.append(
        EpitheliumCellData.create_cell(point=Point(x=grid.x[3], y=grid.y[3], z=grid.z[3]),)
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
    populated_fungus: FungusCellList,
):
    cell = populated_epithelium[0]
    vox = grid.get_voxel(cell['point'])
    assert len(populated_fungus.get_cells_in_voxel(vox)) == 1

    populated_epithelium.internalize(10, populated_fungus, grid)

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
    fungus_list[0]['internalized'] = True

    epithelium_list.internalize(10, fungus_list, grid)

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
    fungus_list.cell_data['internalized'] = True

    epithelium_list.internalize(10, fungus_list, grid)

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
    fungus_list[1]['internalized'] = True

    epithelium_list.internalize(10, fungus_list, grid)

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
    vox = grid.get_voxel(epithelium_list[0]['point'])
    fungus_list[0]['internalized'] = True
    epithelium_list[0]['phagosome'][:max_spores] = 99 # artificially fill

    epithelium_list.internalize(10, fungus_list, grid)

    assert epithelium_list.len_phagosome(0) == max_spores
    assert 0 not in epithelium_list[0]['phagosome']
    assert not fungus_list[0]['internalized']

# def test_internalize_conidia_n(
#     epithelium_list: EpitheliumCellList, grid: RectangularGrid, fungus_list: FungusCellList
# ):
# 
#     point = Point(x=35, y=35, z=35)
#     epithelium_list.append(EpitheliumCellData.create_cell(point=point))
# 
#     fungus_list.append(
#         FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
#     )
#     fungus_list.append(
#         FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
#     )
# 
#     point = Point(x=45, y=35, z=35)
#     fungus_list.append(
#         FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
#     )
#     fungus_list.append(
#         FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
#     )
# 
#     point = Point(x=55, y=35, z=35)
#     fungus_list.append(
#         FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
#     )
#     fungus_list.append(
#         FungusCellData.create_cell(point=point, status=FungusCellData.Status.RESTING)
#     )
# 
#     # internalize some not all
#     epithelium_list.internalize_conidia(1, grid, fungus_list)
# 
#     assert fungus_list.cell_data['internalized'][0]
#     assert fungus_list.cell_data['internalized'][2]
#     assert epithelium_list.len_phagosome(0) == 4
# 
#     # internalize all
#     epithelium_list.internalize_conidia(2, grid, fungus_list)
# 
#     assert fungus_list.cell_data['internalized'][5]
#     assert epithelium_list.len_phagosome(0) == 6