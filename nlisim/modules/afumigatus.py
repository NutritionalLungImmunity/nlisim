from collections.abc import MutableMapping
from enum import IntEnum
from typing import Iterable, Iterator, List, Optional, Union

import attr
from h5py import Group
import numpy as np
from scipy.sparse import coo_matrix, dok_matrix as sparse_matrix
from scipy.spatial.transform import Rotation

from nlisim.cell import CellData, CellList, CellType
from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.random import rg
from nlisim.state import State, get_class_path


class AfumigatusCellData(CellData):
    GROWTH_SCALE_FACTOR = 0.02  # from original code
    BOOLEAN_NETWORK_LENGTH = 23

    class Status(IntEnum):
        RESTING_CONIDIA = 0
        SWELLING_CONIDIA = 1
        GERMTUBE = 5
        HYPHAE = 2
        DYING = 3
        DEAD = 4
        INTERNALIZED = 6

    class State(IntEnum):
        FREE = 0
        INTERNALIZING = 1
        RELEASING = 2

    AFUMIGATUS_FIELDS = [
        ('boolean_network', 'b1', BOOLEAN_NETWORK_LENGTH),
        ('growth', Point.dtype),
        ('growable', 'b1'),
        ('switched', 'b1'),
        ('branchable', 'b1'),
        ('state', 'u1'),
        ('status', 'u1'),
        ('iron_pool', 'f8'),
        ('iron', 'b1'),
        ('iteration', 'i4'),
        ('health', 'f8'),
        ('mobile', 'b1'),
    ]

    FIELDS = CellData.FIELDS + AFUMIGATUS_FIELDS
    dtype = np.dtype(FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        iron_pool: float = 0,
        status: Status = Status.RESTING_CONIDIA,
        state: State = State.FREE,
        **kwargs,
    ) -> np.record:

        growth = cls.GROWTH_SCALE_FACTOR * Point.from_array(2 * rg.random(3) - 1)
        network = cls.initial_boolean_network()
        growable = True
        switched = False
        branchable = False
        iteration = 0
        iron = False
        health = 100
        mobile = True

        return CellData.create_cell_tuple(**kwargs) + (
            network,
            growth,
            growable,
            switched,
            branchable,
            state,
            status,
            iron_pool,
            iron,
            iteration,
            health,
            mobile,
        )

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        return np.asarray(
            [
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )


@attr.s(auto_attribs=True, kw_only=True, frozen=True)
class AfumigatusCell(MutableMapping):
    tree: 'AfumigatusCellTreeList'
    index: int = attr.ib()

    @index.validator
    def _validate_index(self, attribute, value):
        if value >= len(self.tree):
            raise ValueError('Invalid cell index')

    def __attrs_post_init__(self):
        if self.index < 0:
            object.__setattr__(self, 'index', self.index + len(self.tree))

    @property
    def record(self) -> CellType:
        return self.tree.cells[self.index]

    @property
    def parent(self) -> Optional['AfumigatusCell']:
        indices = self.tree.adjacency[:, self.index].nonzero()[0]
        if len(indices) == 1:
            return None
        if len(indices) == 2 and indices[0] == self.index:
            index = indices[0]
            if index == self.index:
                index = indices[1]
            return self.__class__(tree=self.tree, index=index)

        raise Exception(f'Invalid adjacency matrix at index {self.index}')

    @property
    def children(self) -> Iterator['AfumigatusCell']:
        indices = self.tree.adjacency[self.index].nonzero()[1]
        for index in indices:
            if index != self.index:
                yield self.__class__(tree=self.tree, index=index)

    def add_child(self, cell: Union[CellType, 'AfumigatusCell']):
        if isinstance(cell, AfumigatusCell):
            cell = cell.record
        self.tree.append(cell, parent=self.index)

    def kill(self) -> List['AfumigatusCell']:
        """Mark the current cell as dead and remove it from the tree."""
        return self.tree.kill(self.index)

    def __getitem__(self, key):
        return self.record.__getitem__(key)

    def __setitem__(self, key, value):
        self.tree.cell_data[key][self.index] = value

    def __delitem__(self, key):
        return self.record.__delitem__(key)

    def __iter__(self):
        for key in self.record.dtype.fields.keys():
            yield key

    def __contains__(self, key):
        return key in self.record.dtype.fields

    def __len__(self):
        return len(self.record)


@attr.s(kw_only=True, frozen=True, repr=False)
class AfumigatusCellList(CellList):
    CellDataClass = AfumigatusCellData


@attr.s(kw_only=True, frozen=True, repr=False)
class AfumigatusCellTreeList(object):
    cells: CellList = attr.ib()
    _adjacency: sparse_matrix = attr.ib()

    @_adjacency.default
    def __set_default_adjacency(self):
        return sparse_matrix((0, 0))

    def __attrs_post_init__(self):
        cells = self.cells
        adjacency = self._adjacency

        object.__setattr__(self, '_adjacency', sparse_matrix((cells.max_cells, cells.max_cells)))

        if len(cells) > 0:
            for key, value in adjacency.items():
                self._adjacency[key] = value

    @property
    def adjacency(self) -> sparse_matrix:
        return self._adjacency[: len(self), : len(self)]

    @property
    def roots(self) -> List[CellType]:
        # This is linear in the number of nodes... might need to be optimized.
        roots = []
        for i in range(len(self)):
            cell = self[i]
            if not cell['dead'] and cell.parent is None:
                roots.append(cell)
        return roots

    @property
    def cell_data(self) -> AfumigatusCellData:
        return self.cells.cell_data

    @property
    def max_cells(self) -> int:
        return self.cells.max_cells

    @property
    def grid(self) -> RectangularGrid:
        return self.cells.grid

    @classmethod
    def random_branch_direction(cls, growth: np.ndarray) -> np.ndarray:
        """Rotate by a random 45 degree angle from the axis of growth."""
        growth = Point.from_array(growth)

        # get a random vector orthogonal to the growth vector
        axis = Point.from_array(np.cross(growth, rg.standard_normal(3)))
        axis = (np.pi / 4) * axis / axis.norm()

        # rotate the growth vector 45 degrees along the random axis
        rotation = Rotation.from_rotvec(axis)
        return rotation.apply(growth)

    @classmethod
    def create_from_seed(cls, grid, **kwargs) -> 'AfumigatusCellTreeList':
        cells = AfumigatusCellList.create_from_seed(grid, **kwargs)
        adjacency = sparse_matrix((1, 1))
        adjacency[0, 0] = 1

        return cls(cells=cells, adjacency=adjacency)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index: int) -> AfumigatusCell:
        if isinstance(index, str):
            raise TypeError('Expected an integer index, did you mean `tree.cell_data[key]`?')
        return AfumigatusCell(tree=self, index=index)

    def append(self, cell, parent=None):
        index = len(self)
        self.cells.append(cell)
        self._adjacency[index, index] = 1

        if parent is not None:
            self._adjacency[parent, index] = 1
            iron_pool = self.cells[parent]['iron_pool'] / 2
            self.cells[parent]['iron_pool'] = self.cells[-1]['iron_pool'] = iron_pool

    def extend(self, cells: CellData, parents: Iterable[Union[int, None]] = None) -> None:
        if parents is None:
            parents = [None] * len(cells)

        for cell, parent in zip(cells, parents):
            self.append(cell, parent)

    def kill(self, index: int) -> List[AfumigatusCell]:
        """Kill the cell at the given index and remove it from the tree.

        This method will return a list of cells which were children of the
        killed cell.  These children will be the roots of newly formed trees.
        """
        self.cell_data[index]['dead'] = True
        roots = []
        for i in range(len(self)):
            if i == index:
                continue
            if self._adjacency.pop((i, index), None):
                self[i]['growable'] = True
            if self._adjacency.pop((index, i), None):
                roots.append(self[i])
        return roots

    def is_growable(self) -> np.ndarray:
        cells = self.cells.cell_data
        return self.cells.alive(
            cells['growable']
            & cells.point_mask(cells['point'] + cells['growth'], self.grid)
            & (cells['status'] == AfumigatusCellData.Status.HYPHAE)
        )

    def is_branchable(self, branch_probability: float) -> np.ndarray:
        cells = self.cells.cell_data
        return self.cells.alive(
            cells['branchable']
            & (rg.random(cells.shape) < branch_probability)
            & (cells['status'] == AfumigatusCellData.Status.HYPHAE)
        )

    def absorb_iron(self):
        """Absorb iron from the environment.

        This is a dummy method for the visualization testing purpose.
        """
        #  TODO: implement a real iron uptake model
        cells = self.cells.cell_data
        iron = rg.random(len(self.cells))
        cells['iron_pool'] = np.add(cells['iron_pool'], iron)

    def age(self):
        """Add 1 to iteration each time step."""
        cells = self.cells.cell_data
        cells['iteration'] += 1

    def alive(self):
        return self.cells.alive()

    def elongate(self):
        cells = self.cell_data
        mask = self.is_growable()
        if len(mask) == 0:
            return

        cells['growable'][mask] = False
        cells['branchable'][mask] = True

        children = AfumigatusCellData(len(mask), initialize=True)
        children['point'] = cells['point'][mask] + cells['growth'][mask]
        children['growth'] = cells['growth'][mask]

        self.extend(children, parents=mask)

    def branch(self, branch_probability: float):
        cells = self.cell_data
        indices = self.is_branchable(branch_probability)
        if len(indices) == 0:
            return

        children = AfumigatusCellData(len(indices), initialize=True)
        children['growth'] = cells[indices]['growth']
        children['growth'] = np.apply_along_axis(
            self.random_branch_direction, 1, children['growth']
        )

        children['point'] = cells['point'][indices] + children['growth']

        # filter out children lying outside of the domain
        indices = indices[cells.point_mask(children['point'], self.grid)]
        if len(indices) == 0:
            return

        # set properties
        cells['branchable'][indices] = False
        children['growable'] = True
        children['branchable'] = False

        self.extend(children, parents=indices)

    def save(self, group: Group, name: str, metadata: dict) -> Group:
        composite_group = group.create_group(name)
        composite_group.attrs['class'] = get_class_path(self)
        composite_group.attrs['type'] = 'CellTree'
        self.cells.save(composite_group, 'cells', metadata)

        sp = self.adjacency.tocoo()
        composite_group.create_dataset(name='row', data=sp.row)
        composite_group.create_dataset(name='col', data=sp.col)
        composite_group.create_dataset(name='value', data=sp.data)
        return composite_group

    @classmethod
    def load(
        cls, global_state: State, group: Group, name: str, metadata: dict
    ) -> 'AfumigatusCellTreeList':
        composite_dataset = group[name]
        cells = AfumigatusCellList.load(global_state, composite_dataset, 'cells', metadata)

        adjacency = coo_matrix(
            (
                composite_dataset['value'],
                (composite_dataset['row'], composite_dataset['col']),
            ),
            shape=(cells.max_cells, cells.max_cells),
        ).todok()

        return cls(cells=cells, adjacency=adjacency)


def cell_list_factory(self: 'AfumigatusState'):
    cells = AfumigatusCellList(grid=self.global_state.grid)
    return AfumigatusCellTreeList(cells=cells)


@attr.s(kw_only=True)
class AfumigatusState(ModuleState):
    tree: AfumigatusCellTreeList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    p_lodge: float = 0.5
    ITER_TO_CHANGE_STATUS: int = 2


class Afumigatus(ModuleModel):
    name = 'afumigatus'
    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus

        afumigatus.p_lodge = self.config.getfloat('p_lodge')
        afumigatus.ITER_TO_CHANGE_STATUS = self.config.getint('ITER_TO_CHANGE_STATUS')

        point = Point(x=4, y=4, z=4)
        afumigatus.tree = AfumigatusCellTreeList.create_from_seed(
            grid=state.grid, point=point, status=AfumigatusCellData.Status.HYPHAE
        )

        return state

    def advance(self, state: State, previous_time: float) -> State:
        # afumigatus: AfumigatusState = state.afumigatus
        # grid = state.grid
        # tissue = state.geometry.lung_tissue
        #
        # for index in afumigatus.tree.cells.alive():
        #    cell = afumigatus.tree.cells[index]
        #    vox = grid.get_voxel(cell['point'])
        #
        #    if cell['mobile']:
        #        if vox.x > Voxel(x=grid.xv[-1], y=grid.yv[0], z=grid.zv[0]):
        #            cell['status'] = AfumigatusCellData.Status.DEAD
        #            cell['dead'] = True
        #        if tissue[vox.z, vox.y, vox.x] == TissueTypes.EPITHELIUM:
        #            if afumigatus.p_lodge > rg.random():
        #                cell['mobile'] = False
        #    if cell['iteration'] >= afumigatus.ITER_TO_CHANGE_STATUS:
        #        if cell['status'] == AfumigatusCellData.Status.RESTING_CONIDIA:
        #            cell['status'] = AfumigatusCellData.Status.SWELLING_CONIDIA
        #        elif cell['status'] == AfumigatusCellData.Status.SWELLING_CONIDIA:
        #            cell['status'] = AfumigatusCellData.Status.GERMTUBE
        #        elif cell['status'] == AfumigatusCellData.Status.INTERNALIZED:
        #            if afumigatus.p_internal_swell > rg.random():
        #                cell['status'] = AfumigatusCellData.Status.SWELLING_CONIDIA
        #
        #        if cell['status'] == AfumigatusCellData.Status.SWELLING_CONIDIA:
        #            cell['iteration'] = 0

        return state
