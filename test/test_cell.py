from pytest import fixture

from simulation.cell import CellTree
from simulation.coordinates import Point
from simulation.state import RectangularGrid


@fixture
def grid():
    # a 100 x 100 x 100 unit grid
    yield RectangularGrid.construct_uniform((10, 10, 10), (10, 10, 10))


@fixture
def cell_tree(grid: RectangularGrid):
    # a single cell in the middle of the domain
    tree = CellTree(1, point=Point(50, 50, 50))
    tree['status'] = tree.Status.HYPHAE
    yield tree


def test_initial_attributes(cell_tree: CellTree):
    cell = cell_tree[0]
    assert cell['growable']
    assert not cell['branchable']


def test_branched_attributes(grid: RectangularGrid, cell_tree: CellTree):
    cell_tree['branchable'] = True
    cell_tree = cell_tree.branch(1, grid)

    assert len(cell_tree) == 2

    assert not cell_tree[0]['branchable']

    assert cell_tree[1]['growable']
    assert not cell_tree[1]['branchable']


def test_elongated_attributes(grid: RectangularGrid, cell_tree: CellTree):
    cell_tree = cell_tree.elongate(grid)

    assert len(cell_tree) == 2

    assert not cell_tree[0]['growable']
    assert cell_tree[0]['branchable']

    assert cell_tree[1]['growable']
    assert not cell_tree[1]['branchable']


def test_split_iron_pool(grid: RectangularGrid, cell_tree: CellTree):
    cell_tree['iron_pool'] = 1

    cell_tree = cell_tree.elongate(grid)
    cell_tree = cell_tree.branch(1, grid)

    assert len(cell_tree) == 3
    assert (cell_tree['iron_pool'] == [0.25, 0.5, 0.25]).all()
