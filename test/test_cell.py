from h5py import Group
import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture, raises

from simulation.cell import CellData, CellList
from simulation.coordinates import Point
from simulation.state import RectangularGrid


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
