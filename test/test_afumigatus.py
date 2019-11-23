from pytest import fixture

from simulation.coordinates import Point
from simulation.modules.afumigatus import AfumigatusCellTree, Status
from simulation.state import RectangularGrid


@fixture
def cell_tree(grid: RectangularGrid, point: Point):
    tree = AfumigatusCellTree.create_from_seed(grid=grid, point=point, status=Status.HYPHAE)
    yield tree


def test_split_iron_pool(cell_tree: AfumigatusCellTree):
    cell_tree.cells['iron_pool'] = 1

    cell_tree.elongate()
    assert len(cell_tree) == 2

    cell_tree.cells['status'] = Status.HYPHAE

    cell_tree.branch(1)
    assert len(cell_tree) == 3
    assert (cell_tree.cells['iron_pool'] == [0.25, 0.5, 0.25]).all()


def test_elongate(cell_tree: AfumigatusCellTree):
    cell_tree.cells['iron_pool'] = 1
    for _ in range(100):
        cell_tree.cells['status'] = Status.HYPHAE
        cell_tree.elongate()

    assert len(cell_tree) == 101


def test_branch(cell_tree: AfumigatusCellTree):
    cell_tree.cells['iron_pool'] = 1
    cell_tree.cells['status'] = Status.HYPHAE
    cell_tree.elongate()

    cell_tree.cells['status'] = Status.HYPHAE
    cell_tree.branch(1)

    cell_tree.cells['status'] = Status.HYPHAE
    cell_tree.elongate()

    cell_tree.cells['status'] = Status.HYPHAE
    cell_tree.elongate()

    cell_tree.cells['status'] = Status.HYPHAE
    cell_tree.branch(1)

    assert len(cell_tree) == 11
