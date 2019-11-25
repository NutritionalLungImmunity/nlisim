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
    """A low-level data contain for an array cells.

    This class is a subtype of
    [numpy.recarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html)
    containing the lowest level representation of a list of "cells" in a
    simulation.  The underlying data format of this type are identical to a
    simple array of C structures with the fields given in the static "dtype"
    variable.

    The base class contains only a single coordinate representing the location
    of the center of the cell.  Most implementations will want to override this
    class to append more fields.  Subclasses must also override the base
    implementation of `create_cell` to construct a single record containing
    the additional fields.

    For example, the following derived class adds an addition floating point value
    associated with each cell.

    ```python
    class DerivedCell(Cell):
        DERIVED_FIELDS = [
            ('iron_content', 'f8'),
        ]

        dtype = np.dtype(Cell.BASE_FIELDS + DERIVED_FIELDS,
                         align=True)

        @classmethod
        def create_cell(cls, point=None, iron_content=0):
            return np.rec.array(
                [(point, iron_content)],
                dtype=cls.dtype
            )[0]
    ```
    """

    BASE_FIELDS = [
        ('point', Point.dtype),
    ]
    """
    This variable contains the base fields that all subclasses should include
    in their derived data type.
    """

    # typing for dtype doesn't work correctly with this argument
    dtype = np.dtype(BASE_FIELDS, align=True)  # type: ignore
    """
    Subclasses **must** override this value to append custom fields into each
    cell record.
    """

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
        """Create a single record with type `cls.dtype`.

        Subclasses appending fields must override this with custom default
        values.
        """
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
    """A python view on top of a CellData array.

    This class represents a pythonic interface to the data contained in a
    CellData array.  Because the CellData class is a low-level object, it does
    not allow dynamically appending new elements.  Objects of this class get
    around this limitation by pre-allocating a large block of memory that is
    transparently available.  User-facing properties are sliced to make it
    appear as if the extra data is not there.

    Subclassed types are expected to set the `CellDataClass` attribute to
    a subclass of `CellData`.  This provides information about the underlying
    low-level array.

    Parameters
    ------
    grid : `simulation.state.RectangularGrid`
    max_cells : int, optional
    cells : `simulation.cell.CellData`, optional

    """

    CellDataClass: Type[CellData] = CellData
    """
    A class that overrides `CellData` that represents the format of the data
    contained in the list.
    """

    grid: RectangularGrid = attr.ib()
    max_cells: int = attr.ib(default=MAX_CELL_LIST_SIZE)
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
        """Return the portion of the underlying data array containing valid data."""
        return self._cell_data[: self._ncells]

    @classmethod
    def create_from_seed(cls, grid: RectangularGrid, **kwargs) -> 'CellList':
        """Create a new cell list initialized with a single cell.

        The kwargs provided are passed on to the `create_cell` method of the
        data array class.
        """
        cell = cls.CellDataClass.create_cell(**kwargs)
        cell_data = cls.CellDataClass([cell])

        return cls(grid=grid, cell_data=cell_data)

    def append(self, cell: CellType) -> None:
        """Append a new cell the the list."""
        if len(self) >= self.max_cells:
            raise Exception('Not enough free space in cell tree')

        index = self._ncells
        object.__setattr__(self, '_ncells', self._ncells + 1)
        self._cell_data[index] = cell

    def extend(self, cells: Iterable[CellData]) -> None:
        """Extend the cell list by multiple cells."""
        for cell in cells:
            self.append(cell)

    def save(self, group: Group, name: str, metadata: dict) -> Group:
        """Save the cell list.

        Save the list of cells as a new composite data structure inside
        an HDF5 group.  Subclasses should not need to over-ride this method.
        It will automatically create a new variable in the file with the
        correct data-type.  It will also create a reference to the original
        class so that it can be deserialized into the correct type.
        """
        composite_group = group.create_group(name)

        composite_group.attrs['type'] = 'CellList'
        composite_group.attrs['class'] = get_class_path(self)
        composite_group.attrs['max_cells'] = self.max_cells

        composite_group.create_dataset(name='cell_data', data=self.cell_data)
        return composite_group

    @classmethod
    def load(cls, global_state: State, group: Group, name: str, metadata: dict) -> 'CellList':
        """Load a cell list object.

        Load a `CellList` subclass from a composite group inside an HDF5 file.  As with
        `simulation.cell.CellList.save`, subclasses should not need to override this
        method.
        """
        composite_dataset = group[name]

        attrs = composite_dataset.attrs
        max_cells = attrs.get('max_cells', MAX_CELL_LIST_SIZE)
        cell_data = composite_dataset['cell_data'][:].view(cls.CellDataClass)

        return cls(max_cells=max_cells, grid=global_state.grid, cell_data=cell_data)
