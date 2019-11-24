from pytest import fixture

from simulation.cell import CellData, CellList
from simulation.coordinates import Point
from simulation.state import RectangularGrid


@fixture
def cells(grid: RectangularGrid, point: Point):
    # a single cell in the middle of the domain
    cell = CellData.create_cell(point=point)
    cells = CellData([cell])
    yield cells


@fixture
def cell_list(grid: RectangularGrid, point: Point):
    cells = CellList.create_from_seed(grid=grid, point=point)
    yield cells
