from typing import Set

from pytest import fixture

from simulation.coordinates import Point
from simulation.modules.afumigatus import AfumigatusCellData, AfumigatusCellTree
from simulation.state import RectangularGrid

Status = AfumigatusCellData.Status


@fixture
def cell_tree(grid: RectangularGrid, point: Point):
    tree = AfumigatusCellTree.create_from_seed(grid=grid, point=point, status=Status.HYPHAE)
    yield tree


@fixture
def populated_tree(cell_tree: AfumigatusCellTree, point: Point):
    """Return a non-trivial cell tree."""
    # TODO: use valid locations
    cell_tree[0]['growable'] = False
    cell_tree[0]['branchable'] = False
    for _ in range(5):
        cell = AfumigatusCellData.create_cell(point=point)
        cell['growable'] = True
        cell['branchable'] = True
        cell_tree.append(cell, parent=0)

    for i in [2, 3]:
        cell_tree[i]['growable'] = False
        cell_tree[i]['branchable'] = False
        for _ in range(2 * i):
            cell = AfumigatusCellData.create_cell(point=point)
            cell['growable'] = True
            cell['branchable'] = True
            cell_tree.append(cell, parent=i)

    for i in [4, 10]:
        cell_tree[i]['growable'] = False
        cell_tree[i]['branchable'] = False

        cell = AfumigatusCellData.create_cell(point=point)
        cell['growable'] = True
        cell['branchable'] = True
        cell_tree.append(cell, parent=i)

    yield cell_tree


def test_split_iron_pool(cell_tree: AfumigatusCellTree):
    cell_tree.cell_data['iron_pool'] = 1

    cell_tree.elongate()
    assert len(cell_tree) == 2

    cell_tree.cell_data['status'] = Status.HYPHAE

    cell_tree.branch(1)
    assert len(cell_tree) == 3
    assert (cell_tree.cell_data['iron_pool'] == [0.25, 0.5, 0.25]).all()


def test_elongate(cell_tree: AfumigatusCellTree):
    cell_tree.cell_data['iron_pool'] = 1
    for _ in range(100):
        cell_tree.cell_data['status'] = Status.HYPHAE
        cell_tree.elongate()

    assert len(cell_tree) == 101


def test_branch(cell_tree: AfumigatusCellTree):
    cell_tree.cell_data['iron_pool'] = 1
    cell_tree.cell_data['status'] = Status.HYPHAE
    cell_tree.elongate()

    cell_tree.cell_data['status'] = Status.HYPHAE
    cell_tree.branch(1)

    cell_tree.cell_data['status'] = Status.HYPHAE
    cell_tree.elongate()

    cell_tree.cell_data['status'] = Status.HYPHAE
    cell_tree.elongate()

    cell_tree.cell_data['status'] = Status.HYPHAE
    cell_tree.branch(1)

    assert len(cell_tree) == 11


def test_initial_attributes(point: Point):
    cells = AfumigatusCellData([AfumigatusCellData.create_cell(point=point)])
    cell = cells[0]
    assert cell['growable']
    assert not cell['branchable']


def test_branched_attributes(cell_tree: AfumigatusCellTree):
    cell_tree.cell_data['branchable'] = True
    cell_tree.branch(1)

    assert len(cell_tree) == 2

    assert not cell_tree[0]['branchable']

    assert cell_tree[1]['growable']
    assert not cell_tree[1]['branchable']


def test_elongated_attributes(cell_tree: AfumigatusCellData):
    cell_tree.elongate()

    assert len(cell_tree) == 2

    assert not cell_tree[0]['growable']
    assert cell_tree[0]['branchable']

    assert cell_tree[1]['growable']
    assert not cell_tree[1]['branchable']


def test_traverse_tree(populated_tree: AfumigatusCellTree):
    visited: Set[int] = set()

    def visit(root):
        assert root.index not in visited
        visited.add(root.index)
        for child in root.children:
            visit(child)

    visit(populated_tree.roots[0])
    assert visited == set(range(len(populated_tree)))


def test_mutate_cell(populated_tree: AfumigatusCellTree):
    cell = populated_tree[4]
    growable = not cell['growable']
    cell['growable'] = growable

    assert populated_tree.cell_data['growable'][4] == growable
