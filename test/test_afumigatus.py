from typing import Set, cast

from h5py import Group
from pytest import fixture

from nlisim.config import SimulationConfig
from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.modules.afumigatus import AfumigatusCellData, AfumigatusCellTreeList, AfumigatusState
from nlisim.state import State

Status = AfumigatusCellData.Status


@fixture
def cell_tree(grid: RectangularGrid, point: Point):
    tree = AfumigatusCellTreeList.create_from_seed(grid=grid, point=point, status=Status.HYPHAE)
    yield tree


@fixture
def populated_tree(cell_tree: AfumigatusCellTreeList, point: Point):
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


def test_split_iron_pool(cell_tree: AfumigatusCellTreeList):
    cell_tree.cell_data['iron_pool'] = 1

    cell_tree.elongate()
    assert len(cell_tree) == 2

    cell_tree.cell_data['status'] = Status.HYPHAE

    cell_tree.branch(1)
    assert len(cell_tree) == 3
    assert (cell_tree.cell_data['iron_pool'] == [0.25, 0.5, 0.25]).all()


def test_elongate(cell_tree: AfumigatusCellTreeList):
    cell_tree.cell_data['iron_pool'] = 1
    for _ in range(100):
        cell_tree.cell_data['status'] = Status.HYPHAE
        cell_tree.elongate()

    assert len(cell_tree) == 101


def test_branch(cell_tree: AfumigatusCellTreeList):
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


def test_branched_attributes(cell_tree: AfumigatusCellTreeList):
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


def test_traverse_tree(populated_tree: AfumigatusCellTreeList):
    visited: Set[int] = set()

    def visit(root):
        assert root.index not in visited
        visited.add(root.index)
        for child in root.children:
            visit(child)

    visit(populated_tree.roots[0])
    assert visited == set(range(len(populated_tree)))


def test_mutate_cell(populated_tree: AfumigatusCellTreeList):
    cell = populated_tree[4]
    growable = not cell['growable']
    cell['growable'] = growable

    assert populated_tree.cell_data['growable'][4] == growable


def test_serialize(hdf5_group: Group, cell_tree: AfumigatusCellTreeList):
    config = SimulationConfig(
        {
            'simulation': {
                'modules': 'nlisim.modules.afumigatus.Afumigatus',
                'nx': 20,
                'ny': 40,
                'nz': 20,
                'dx': 20,
                'dy': 40,
                'dz': 20,
                'validate': True,
            }
        }
    )
    state = State.create(config)

    state.afumigatus.tree.append(cell_tree.cell_data[0])
    state.afumigatus.save_state(hdf5_group)

    assert 'tree' in hdf5_group
    assert hdf5_group['tree']['row'][:] == [0]
    assert hdf5_group['tree']['col'][:] == [0]
    assert hdf5_group['tree']['value'][:] == [1]
    assert hdf5_group['tree']['cells']['cell_data'].shape == (1,)
    assert hdf5_group['tree']['cells']['cell_data'].dtype == cell_tree.cell_data.dtype


def test_deserialize(hdf5_group: Group, cell_tree: AfumigatusCellTreeList):
    config = SimulationConfig(
        {
            'simulation': {
                'modules': 'nlisim.modules.afumigatus.Afumigatus',
                'nx': 20,
                'ny': 40,
                'nz': 20,
                'dx': 20,
                'dy': 40,
                'dz': 20,
                'validate': True,
            }
        }
    )
    state = State.create(config)

    state.afumigatus.tree.append(cell_tree.cell_data[0])
    state.afumigatus.save_state(hdf5_group)

    copy = cast(AfumigatusState, AfumigatusState.load_state(state, hdf5_group))
    assert state.afumigatus.tree.cell_data == copy.tree.cell_data


def test_kill_cell(populated_tree: AfumigatusCellTreeList):
    old_parent = populated_tree[4].parent
    assert old_parent is not None

    new_roots = populated_tree[4].kill()
    assert len(new_roots) == len(populated_tree.roots) - 1
    assert old_parent['growable']
    assert populated_tree[4]['dead']
