from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple, Type, Union, cast

# noinspection PyPackageRequirements
from attr import attrib, attrs

# noinspection PyPackageRequirements
from h5py import Group

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
from numpy.typing import DTypeLike

from nlisim.coordinates import Point
from nlisim.grid import TetrahedralMesh
from nlisim.state import State, get_class_path

MAX_CELL_LIST_SIZE = 1_000_000

# the way numpy types single records is strange...
CellType = Any

CellFields = List[
    Union[
        Tuple[str, DTypeLike],
        Tuple[str, DTypeLike, int],
    ]
]


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
    class DerivedCell(CellData):
        FIELDS = CellData.FIELDS + [
            ('iron_content', 'f8')
        ]

        dtype = np.dtype(CellData.FIELDS, align=True)

        @classmethod
        def create_cell_tuple(cls, iron_content=0, **kwargs) -> Tuple:
            return CellData.create_cell_tuple(**kwargs) + (iron_content,)
    ```
    """

    FIELDS: CellFields = [
        ('point', Point.dtype),
        ('element_index', int),
        ('dead', 'b1'),
    ]
    """
    This variable contains the base fields that all subclasses should include
    in their derived data type.
    """

    # typing for dtype doesn't work correctly with this argument
    dtype = np.dtype(FIELDS, align=True)  # type: ignore
    """
    Subclasses **must** override this value to append custom fields into each
    cell record.
    """

    def __new__(cls, arg: Union[int, Iterable['CellData']], initialize: bool = False, **kwargs):
        if isinstance(arg, (int, np.int64, np.int32)):
            arg = cast(int, arg)
            array = np.ndarray(shape=(arg,), dtype=cls.dtype).view(cls)
            if initialize:
                for index in range(arg):
                    array[index] = cls.create_cell(**kwargs)
            return array

        return np.asarray(arg, dtype=cls.dtype).view(cls)

    @classmethod
    def create_cell_tuple(
        cls, *, point: Point = None, element_index: int = -1, dead: bool = False, **kwargs
    ) -> Tuple:
        """Create a tuple of fields attached to a single cell.

        The base class version of this method returns the fields associated with
        just the bare cell.  Subclasses that append additional attributes onto
        the cell must override this method to append their own fields to this
        tuple.  Care must be taken to ensure that the order of the tuple is
        identical to the order of the fields listed in `cls.dtype`.
        """
        if point is None:
            point = Point()

        return point, element_index, dead

    @classmethod
    def create_cell(cls, **kwargs) -> 'CellData':
        """Create a single record with type `cls.dtype`.

        Subclasses appending fields must override this with custom default
        values.
        """
        return np.rec.array([cls.create_cell_tuple(**kwargs)], dtype=cls.dtype)[0]


@attrs(kw_only=True, frozen=True, repr=False)
class CellList(object):
    # noinspection PyUnresolvedReferences
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
    mesh : `simulation.mesh.RectangularGrid`
    max_cells : int, optional
    cells : `simulation.cell.CellData`, optional

    """

    CellDataClass: Type[CellData] = CellData
    """
    A class that overrides `CellData` that represents the format of the data
    contained in the list.
    """

    mesh: TetrahedralMesh = attrib()
    max_cells: int = attrib(default=MAX_CELL_LIST_SIZE)
    _cell_data: CellData = attrib()
    _ncells: int = attrib(init=False)
    _cells_in_element_by_index: Dict[int, Set[int]] = attrib(
        init=False, factory=lambda: defaultdict(set)
    )

    # noinspection PyUnresolvedReferences
    @_cell_data.default
    def __set_default_cells(self) -> CellData:
        return self.CellDataClass(0)

    def __attrs_post_init__(self):
        cells = self._cell_data

        object.__setattr__(self, '_ncells', len(cells))
        object.__setattr__(self, '_cell_data', self.CellDataClass(self.max_cells))

        if len(cells) > 0:
            self._cell_data[: len(cells)] = cells

        self._compute_element_index()

    def __len__(self) -> int:
        return self._ncells

    def __repr__(self) -> str:
        return f'CellList[{self._ncells}]'

    def __getitem__(self, index: Union[int, slice]) -> CellType:
        if isinstance(index, str):
            raise TypeError('Expected an integer index, did you mean `cells.cell_data[key]`?')
        return self.cell_data[index]

    # Trick mypy into recognizing this class as iterable:
    #   https://github.com/python/mypy/issues/2220
    def __iter__(self) -> Iterator[CellType]:
        for index in range(len(self)):
            yield self[index]

    @property
    def cell_data(self) -> CellData:
        """Return the portion of the underlying data array containing valid data."""
        return self._cell_data[: self._ncells]

    @classmethod
    def create_from_seed(cls, mesh: TetrahedralMesh, **kwargs) -> 'CellList':
        """Create a new cell list initialized with a single cell.

        The kwargs provided are passed on to the `create_cell` method of the
        data array class.
        """
        cell = cls.CellDataClass.create_cell(**kwargs)
        cell_data = cls.CellDataClass([cell])

        return cls(mesh=mesh, cell_data=cell_data)

    # TODO: this is inconsistent with iterating over the whole CellList, why does this give indices
    #  while the other gives the records
    def alive(self, sample: Iterable = None) -> np.ndarray:
        """Get a list of indices containing cells that are alive.

        This method will filter out cells that are dead according to the
        value of the `dead` field.  Optionally, you can also pass in a boolean
        mask or index array.  This method will then filter the given list of
        cells rather than the full list.

        For example, to iterate over all living cells:
        ```python
        for index in cells.alive():
            cell = cells[index]
            # do something...
        ```

        To iterate over a sub-sample of living cells:
        ```python
        sample = [1, 10, 15]
        for index in cells.alive(sample):
            cell = cells[index]
            # do something...
        ```

        To iterate over a boolean mask of living cells:
        ```python
        sample = cells.cell_data['iron'] > 0.5
        for index in cells.alive(sample):
            cell = cells[index]
            # do something...
        ```
        """
        cell_data = self.cell_data
        if sample is None:
            return (cell_data['dead'] == False).nonzero()[0]  # noqa: E712

        sample_indices = np.asarray(sample)
        if sample_indices.dtype == 'b1':
            if sample_indices.shape != self.cell_data.shape:
                raise ValueError('Expected boolean mask the same size as the cell list')
            sample_indices = sample_indices.nonzero()[0]

        mask = (cell_data[sample_indices]['dead'] == False).nonzero()[0]  # noqa: E712
        return sample_indices[mask]

    def append(self, cell: CellType) -> int:
        """Append a new cell to the list."""
        if len(self) >= self.max_cells:
            raise Exception('Not enough free space in cell tree')

        cell_index = self._ncells
        object.__setattr__(self, '_ncells', self._ncells + 1)
        self._cell_data[cell_index] = cell
        element_index = cell['element_index']
        if element_index < 0:
            element_index = self.mesh.get_element_index(cell['point'])
        self._cells_in_element_by_index[element_index].add(cell_index)
        return cell_index

    def extend(self, cells: Iterable[CellData]) -> None:
        """Extend the cell list by multiple cells."""
        for cell in cells:
            self.append(cell)

    # noinspection PyUnusedLocal
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

    # noinspection PyUnusedLocal
    @classmethod
    def load(cls, global_state: State, group: Group, name: str, metadata: dict) -> 'CellList':
        """Load a cell list object.

        Load a `CellList` subclass from a composite group inside an HDF5 file.  As with
        `simulation.cell.CellList.save`, subclasses should not need to override this
        method.
        """
        composite_dataset = group[name]

        dataset_attrs = composite_dataset.attrs
        max_cells = dataset_attrs.get('max_cells', MAX_CELL_LIST_SIZE)
        cell_data = composite_dataset['cell_data'][:].view(cls.CellDataClass)

        return cls(max_cells=max_cells, mesh=global_state.mesh, cell_data=cell_data)

    def get_cells_in_element(self, element_index: int) -> np.ndarray:
        """Return a list of cell indices contained in a given element."""
        return np.asarray(sorted((self._cells_in_element_by_index[element_index])))

    def get_neighboring_cells(self, cell: CellData) -> np.ndarray:
        """Return a list of cells indices in the same element."""
        return self.get_cells_in_element(cell['element_index'])

    def update_element_index(self, indices: Iterable = None):
        """Update the embedded element index.

        This method will update the element indices for a given list of cells,
        or if no parameter is provided, for all of the cells.  Currently,
        calling this method is only required if the `point` field of a cell
        is changed... i.e. if the cell is moved to a potentially different
        element.
        """
        if indices is None:
            self._cells_in_element_by_index.clear()
            self._compute_element_index()
            return

        for index in indices:
            cell = self[index]
            old_element = cell['element_index']
            new_element = self.mesh.get_element_index(cell['point'])
            assert new_element > 0, f"{old_element=} {new_element=}"
            if old_element != new_element:
                self._cells_in_element_by_index[old_element].remove(index)
                self._cells_in_element_by_index[new_element].add(index)
                cell['element_index'] = new_element

    def _compute_element_index(self):
        """Generate a dictionary mapping element index to cell index.

        This index exists to maintain efficient (sub-linear) access to cells contained
        in a single element.  This method is called automatically on initialization.
        """
        for cell_index in range(len(self)):
            cell = self[cell_index]
            element = self.mesh.get_element_index(cell['point'])
            assert element >= 0
            self._cells_in_element_by_index[element].add(cell_index)
            cell['element_index'] = element
