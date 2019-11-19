from typing import Set

from pytest import fixture, mark

from simulation.cell import CellArray, CellTree
from simulation.coordinates import Point
from simulation.state import RectangularGrid


@fixture
def grid():
    # a 100 x 100 x 100 unit grid
    yield RectangularGrid.construct_uniform((10, 10, 10), (10, 10, 10))


@fixture
def point():
    yield Point(50, 50, 50)


@fixture
def cells(grid: RectangularGrid, point: Point):
    # a single cell in the middle of the domain
    cell = CellArray.create_cell(point=point)
    cells = CellArray([cell])
    yield cells


@fixture
def cell_tree(grid: RectangularGrid, point: Point):
    tree = CellTree.create_from_seed(grid=grid, point=point)
    yield tree


@fixture
def populated_tree(cell_tree: CellTree, point: Point):
    """Return a non-trivial cell tree."""
    # TODO: use valid locations
    cell_tree[0]['growable'] = False
    cell_tree[0]['branchable'] = False
    for _ in range(5):
        cell = CellArray.create_cell(point=point)
        cell['growable'] = True
        cell['branchable'] = True
        cell_tree.append(cell, parent=0)

    for i in [2, 3]:
        cell_tree[i]['growable'] = False
        cell_tree[i]['branchable'] = False
        for _ in range(2 * i):
            cell = CellArray.create_cell(point=point)
            cell['growable'] = True
            cell['branchable'] = True
            cell_tree.append(cell, parent=i)

    for i in [4, 10]:
        cell_tree[i]['growable'] = False
        cell_tree[i]['branchable'] = False

        cell = CellArray.create_cell(point=point)
        cell['growable'] = True
        cell['branchable'] = True
        cell_tree.append(cell, parent=i)

    yield cell_tree


def test_initial_attributes(cells: CellArray):
    cell = cells[0]
    assert cell['growable']
    assert not cell['branchable']


def test_branched_attributes(cell_tree: CellTree):
    cell_tree.cells['branchable'] = True
    cell_tree.branch(1)

    assert len(cell_tree) == 2

    assert not cell_tree[0]['branchable']

    assert cell_tree[1]['growable']
    assert not cell_tree[1]['branchable']


def test_elongated_attributes(cell_tree: CellArray):
    cell_tree.elongate()

    assert len(cell_tree) == 2

    assert not cell_tree[0]['growable']
    assert cell_tree[0]['branchable']

    assert cell_tree[1]['growable']
    assert not cell_tree[1]['branchable']


@mark.skip('Move to module')
def test_split_iron_pool(cell_tree: CellArray):
    cell_tree.cells['iron_pool'] = 1

    cell_tree.elongate()
    cell_tree.branch(1)

    assert len(cell_tree) == 3
    assert (cell_tree.cells['iron_pool'] == [0.25, 0.5, 0.25]).all()


def test_traverse_tree(populated_tree: CellTree):
    visited: Set[int] = set()

    def visit(root):
        assert root.index not in visited
        visited.add(root.index)
        for child in root.children:
            visit(child)

    visit(populated_tree.root)
    assert visited == set(range(len(populated_tree)))


def test_mutate_cell(populated_tree: CellTree):
    cell = populated_tree[4]
    growable = not cell['growable']
    cell['growable'] = growable

    assert populated_tree.cells['growable'][4] == growable
