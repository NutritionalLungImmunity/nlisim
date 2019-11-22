from collections.abc import MutableMapping
from typing import Any, Iterable, Optional, Type, Union

import attr
import numpy as np
from scipy.sparse import dok_matrix as sparse_matrix
from scipy.spatial.transform import Rotation

from simulation.coordinates import Point
from simulation.state import RectangularGrid

MAX_CELL_TREE_SIZE = 10000

# the way numpy types single records is strange...
CellType = Any


class CellArray(np.ndarray):
    GROWTH_SCALE_FACTOR = 0.02  # from original code

    BASE_FIELDS = [
        ('point', Point.dtype),
        ('growth', Point.dtype),
        ('growable', 'b1'),
        ('switched', 'b1'),
        ('branchable', 'b1')
    ]

    # typing for dtype doesn't work correctly with this argument
    dtype = np.dtype(BASE_FIELDS, align=True)  # type: ignore

    def __new__(cls, arg: Union[int, Iterable[np.record]], **kwargs):
        if isinstance(arg, int):
            return np.ndarray(shape=(arg,), dtype=cls.dtype).view(cls)

        return np.asarray(arg, dtype=cls.dtype).view(cls)

    @classmethod
    def create_cell(cls, point: Point = None, **kwargs) -> np.record:
        if point is None:
            point = Point()
        growth = cls.GROWTH_SCALE_FACTOR * Point.from_array(2 * np.random.rand(3) - 1)
        growable = True
        switched = False
        branchable = False

        return np.rec.array([
            (point, growth, growable, switched, branchable)
        ], dtype=cls.dtype)[0]

    @classmethod
    def random_branch_direction(cls, growth: np.ndarray) -> np.ndarray:
        """Rotate by a random 45 degree angle from the axis of growth."""
        growth = Point.from_array(growth)

        # get a random vector orthogonal to the growth vector
        axis = Point.from_array(np.cross(growth, np.random.randn(3)))
        axis = (np.pi / 4) * axis / axis.norm()

        # rotate the growth vector 45 degrees along the random axis
        rotation = Rotation.from_rotvec(axis)
        return rotation.apply(growth)

    @classmethod
    def point_mask(cls, points: np.ndarray, grid: RectangularGrid):
        """Generate a mask array from a set of points.

        The output is a boolean array indicating if the point at that index
        is a valid location for a cell.
        """
        assert points.shape[1] == 3, 'Invalid point array shape'
        point = points.T.view(Point)

        # TODO: add geometry restriction
        return (
            (grid.xv[0] <= point.x) & (point.x <= grid.xv[-1]) &
            (grid.yv[0] <= point.y) & (point.y <= grid.yv[-1]) &
            (grid.zv[0] <= point.z) & (point.z <= grid.zv[-1])
        )


@attr.s(auto_attribs=True, kw_only=True, frozen=True)
class Cell(MutableMapping):
    tree: 'CellTree'
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
    def parent(self) -> Optional['Cell']:
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
    def children(self) -> Iterable['Cell']:
        indices = self.tree.adjacency[self.index].nonzero()[1]
        for index in indices:
            if index != self.index:
                yield self.__class__(tree=self.tree, index=index)

    def add_child(self, cell: Union[CellType, 'Cell']):
        if isinstance(cell, Cell):
            cell = cell.record
        self.tree.append(cell, parent=self.index)

    def __getitem__(self, key):
        return self.record.__getitem__(key)

    def __setitem__(self, key, value):
        self.tree.cells[key][self.index] = value

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
class CellTree(object):
    CellArrayClass: Type[CellArray] = CellArray

    max_cells: int = attr.ib(default=MAX_CELL_TREE_SIZE)
    grid: RectangularGrid = attr.ib()
    _cells: CellArray = attr.ib()
    _adjacency: sparse_matrix = attr.ib()
    _ncells: int = attr.ib(init=False)

    @_cells.default
    def __set_default_cells(self) -> CellArray:
        return self.CellArrayClass(0)

    @_adjacency.default
    def __set_default_adjacency(self):
        return sparse_matrix((0, 0))

    def __attrs_post_init__(self):
        cells = self._cells
        adjacency = self._adjacency

        object.__setattr__(self, '_ncells', len(cells))
        object.__setattr__(self, '_cells', self.CellArrayClass(self.max_cells))
        object.__setattr__(self, '_adjacency', sparse_matrix((self.max_cells, self.max_cells)))

        if len(cells) > 0:
            self._cells[:len(cells)] = cells
            for key, value in adjacency.items():
                self._adjacency[key] = value

    def __len__(self) -> int:
        return self._ncells

    def __repr__(self) -> str:
        return f'CellTree[{self._ncells}]'

    def __getitem__(self, index: int) -> Cell:
        if isinstance(index, str):
            raise TypeError('Expected an integer index, did you mean `tree.cells[key]`?')
        return Cell(tree=self, index=index)

    @property
    def root(self) -> Optional[CellType]:
        if not len(self):
            return None
        return self[0]

    @property
    def cells(self) -> CellArray:
        return self._cells[:self._ncells]

    @property
    def adjacency(self) -> sparse_matrix:
        return self._adjacency[:self._ncells, :self._ncells]

    @classmethod
    def create_from_seed(cls, grid, **kwargs) -> 'CellTree':
        cell = cls.CellArrayClass.create_cell(**kwargs)
        cells = cls.CellArrayClass([cell])
        adjacency = sparse_matrix((1, 1))
        adjacency[0, 0] = 1

        return cls(grid=grid, cells=cells, adjacency=adjacency)

    def append(self, cell: CellType, parent: int = None) -> None:
        if len(self) >= self.max_cells - 1:
            raise Exception('Not enough free space in cell tree')

        index = self._ncells
        object.__setattr__(self, '_ncells', self._ncells + 1)
        self._cells[index] = cell
        self._adjacency[index, index] = 1

        if parent is not None:
            self._adjacency[parent, index] = 1

    def extend(self, cells: CellArray, parents: Iterable[Union[int, None]] = None) -> None:
        if parents is None:
            parents = [None] * len(cells)

        for cell, parent in zip(cells, parents):
            self.append(cell, parent)

    def is_growable(self) -> np.ndarray:
        cells = self.cells
        return (
            cells['growable'] &
            cells.point_mask(cells['point'] + cells['growth'], self.grid)
        )

    def is_branchable(self, branch_probability: float) -> np.ndarray:
        cells = self.cells
        return (
            cells['branchable'] &
            (np.random.rand(*cells.shape) < branch_probability)
        )

    def elongate(self):
        cells = self.cells
        mask = self.is_growable().nonzero()[0]
        if len(mask) == 0:
            return

        cells['growable'][mask] = False
        cells['branchable'][mask] = True

        children = self.CellArrayClass(len(mask))
        children['point'] = cells['point'][mask] + cells['growth'][mask]
        children['growth'] = cells['growth'][mask]

        self.extend(children, parents=mask)

    def branch(self, branch_probability: float):
        cells = self.cells
        indices = self.is_branchable(branch_probability).nonzero()[0]
        if len(indices) == 0:
            return

        children = self.CellArrayClass(len(indices))
        children['growth'] = np.apply_along_axis(
            cells.random_branch_direction, 1, children['growth'])

        children['point'] = cells['point'][indices] + children['growth'][indices]

        # filter out children lying outside of the domain
        indices = indices[cells.point_mask(children['point'], self.grid)]
        if len(indices) == 0:
            return

        # set properties
        cells['branchable'][indices] = False
        children['growable'] = True
        children['branchable'] = False

        self.extend(children, parents=indices)
