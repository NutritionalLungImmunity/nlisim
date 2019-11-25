from typing import Any, Iterable, Type, Union

import attr
from h5py import Group
import numpy as np

from simulation.coordinates import Point
from simulation.state import get_class_path, RectangularGrid, State

MAX_CELL_LIST_SIZE = 10000

# the way numpy types single records is strange...
CellType = Any


class CellData(np.ndarray):
    GROWTH_SCALE_FACTOR = 0.02  # from original code

    BASE_FIELDS = [
        ('point', Point.dtype),
    ]

    # typing for dtype doesn't work correctly with this argument
    dtype = np.dtype(BASE_FIELDS, align=True)  # type: ignore

    def __new__(cls, arg: Union[int, Iterable[np.record]], initialize: bool = False, **kwargs):
        if isinstance(arg, (int, np.int64, np.int32)):
            array = np.ndarray(shape=(arg,), dtype=cls.dtype).view(cls)
            if initialize:
                for index in range(arg):
                    array[index] = cls.create_cell(**kwargs)
            return array

        return np.asarray(arg, dtype=cls.dtype).view(cls)

    @classmethod
    def create_cell(cls, point: Point = None, **kwargs) -> np.record:
        if point is None:
            point = Point()

        return np.rec.array([(point,)], dtype=cls.dtype)[0]

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
            (grid.xv[0] <= point.x)
            & (point.x <= grid.xv[-1])
            & (grid.yv[0] <= point.y)
            & (point.y <= grid.yv[-1])
            & (grid.zv[0] <= point.z)
            & (point.z <= grid.zv[-1])
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class CellList(object):
    CellDataClass: Type[CellData] = CellData

    max_cells: int = attr.ib(default=MAX_CELL_LIST_SIZE)
    grid: RectangularGrid = attr.ib()
    _cell_data: CellData = attr.ib()
    _ncells: int = attr.ib(init=False)

    @_cell_data.default
    def __set_default_cells(self) -> CellData:
        return self.CellDataClass(0)

    def __attrs_post_init__(self):
        cells = self._cell_data

        object.__setattr__(self, '_ncells', len(cells))
        object.__setattr__(self, '_cell_data', self.CellDataClass(self.max_cells))

        if len(cells) > 0:
            self._cell_data[: len(cells)] = cells

    def __len__(self) -> int:
        return self._ncells

    def __repr__(self) -> str:
        return f'CellList[{self._ncells}]'

    def __getitem__(self, index: int) -> CellType:
        if isinstance(index, str):
            raise TypeError('Expected an integer index, did you mean `cells.cell_data[key]`?')
        return self.cell_data[index]

    @property
    def cell_data(self) -> CellData:
        return self._cell_data[: self._ncells]

    @classmethod
    def create_from_seed(cls, grid, **kwargs) -> 'CellList':
        cell = cls.CellDataClass.create_cell(**kwargs)
        cell_data = cls.CellDataClass([cell])

        return cls(grid=grid, cell_data=cell_data)

    def append(self, cell: CellType) -> None:
        if len(self) >= self.max_cells:
            raise Exception('Not enough free space in cell tree')

        index = self._ncells
        object.__setattr__(self, '_ncells', self._ncells + 1)
        self._cell_data[index] = cell

    def extend(self, cells: CellData) -> None:
        for cell in cells:
            self.append(cell)

    def save(self, group: Group, name: str, metadata: dict) -> Group:
        composite_group = group.create_group(name)

        composite_group.attrs['type'] = 'CellList'
        composite_group.attrs['class'] = get_class_path(self)
        composite_group.attrs['max_cells'] = self.max_cells

        composite_group.create_dataset(name='cell_data', data=self.cell_data)
        return composite_group

    @classmethod
    def load(cls, global_state: State, group: Group, name: str, metadata: dict) -> 'CellList':
        composite_dataset = group[name]

        attrs = composite_dataset.attrs
        max_cells = attrs.get('max_cells', MAX_CELL_LIST_SIZE)
        cell_data = composite_dataset['cell_data'][:].view(cls.CellDataClass)

        return cls(max_cells=max_cells, grid=global_state.grid, cell_data=cell_data)
