from h5py import Group
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture, raises

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid


@fixture
def cell(grid: RectangularGrid, point: Point):
    # a single cell in the middle of the domain
    cell = CellData.create_cell(point=point)
    cells = CellData([cell])
    yield cells


@fixture
def cell_list(grid: RectangularGrid, point: Point):
    cells = CellList.create_from_seed(grid=grid, point=point)
    yield cells


def test_append_cell_list(cell, cell_list):
    original_length = len(cell_list)
    cell_list.append(cell)
    assert len(cell_list) == original_length + 1
    assert cell == cell_list[-1]


def test_extend_cell_list(cell, cell_list):
    original_length = len(cell_list)
    cell_list.extend([cell, cell])
    assert len(cell_list) == original_length + 2
    assert cell == cell_list[-1]
    assert cell == cell_list[-2]


def test_serialize(cell_list: CellList, hdf5_group: Group):
    cell_list_group = cell_list.save(hdf5_group, 'test', {})
    assert cell_list_group['cell_data'].shape == (len(cell_list),)
    assert cell_list_group['cell_data'].dtype == cell_list.cell_data.dtype


def test_get_cell(cell_list: CellList):
    assert cell_list[0] == cell_list.cell_data[0]


def test_getitem_error(cell_list: CellList):
    with raises(TypeError):
        _ = cell_list['a']  # type: ignore


def test_out_of_memory_error(grid: RectangularGrid, cell: CellData):
    cell_list = CellList(grid=grid, max_cells=1)
    cell_list.append(cell)

    with raises(Exception):
        cell_list.append(cell)


def test_filter_out_dead(grid: RectangularGrid):
    cells = CellList(grid=grid)
    cells.extend([CellData.create_cell(dead=bool(i % 2)) for i in range(10)])

    assert_array_equal(cells.alive(), [0, 2, 4, 6, 8])
    assert_array_equal(cells.alive([1, 2, 3, 6]), [2, 6])

    mask = np.arange(10) < 5
    assert_array_equal(cells.alive(mask), [0, 2, 4])


def test_get_neighboring_cells(grid: RectangularGrid):
    point = Point(x=4.5, y=4.5, z=4.5)

    raw_cells = [CellData.create_cell(point=point) for _ in range(5)]
    raw_cells[1]['point'] = Point(x=-1, y=4.5, z=4.5)
    raw_cells[4]['point'] = Point(x=4.5, y=4.5, z=-1)

    cells = CellList(grid=grid)
    cells.extend(raw_cells)

    assert_array_equal(cells.get_neighboring_cells(cells[0]), [0, 2, 3])
    assert_array_equal(cells.get_cells_in_voxel(Voxel(x=0, y=0, z=0)), [0, 2, 3])


def test_move_cell(grid: RectangularGrid):
    point = Point(x=4.5, y=4.5, z=4.5)

    raw_cells = [CellData.create_cell(point=point) for _ in range(5)]
    raw_cells[1]['point'] = Point(x=-1, y=4.5, z=4.5)
    raw_cells[4]['point'] = Point(x=4.5, y=4.5, z=-1)

    cells = CellList(grid=grid)
    cells.extend(raw_cells)

    cells[0]['point'] = Point(x=50, y=50, z=50)

    # updating an incorrect index will not update the cell at index 0
    cells.update_voxel_index([1, 3])
    assert_array_equal(cells.get_neighboring_cells(cells[2]), [0, 2, 3])
    assert cells._reverse_voxel_index[0] == grid.get_voxel(point)

    # this should correctly update the voxel index
    cells.update_voxel_index([0])
    assert_array_equal(cells.get_neighboring_cells(cells[0]), [0])
    assert cells._reverse_voxel_index[0] == grid.get_voxel(cells[0]['point'])
