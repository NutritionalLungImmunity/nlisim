from functools import reduce
from typing import List, Tuple

import attr
from h5py import File as H5File
import numpy as np

from simulation.coordinates import Point, Voxel

ShapeType = Tuple[int, int, int]
SpacingType = Tuple[float, float, float]

_dtype_float64 = np.dtype('float64')


@attr.s(auto_attribs=True, repr=False)
class RectangularGrid(object):
    """A class representation of a rectangular grid."""

    # cell centered coordinates (1-d arrays)
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # vertex coordinates (1-d arrays)
    xv: np.ndarray
    yv: np.ndarray
    zv: np.ndarray

    @classmethod
    def _make_coordinate_arrays(cls, size: int, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        vertex = np.arange(size + 1) * spacing
        cell = spacing / 2 + vertex[:-1]
        vertex.flags['WRITEABLE'] = False
        cell.flags['WRITEABLE'] = False
        return cell, vertex

    @classmethod
    def construct_uniform(cls, shape: ShapeType, spacing: SpacingType) -> 'RectangularGrid':
        """Create a rectangular grid with uniform spacing in each axis."""
        nz, ny, nx = shape
        dz, dy, dx = spacing
        x, xv = cls._make_coordinate_arrays(nx, dx)
        y, yv = cls._make_coordinate_arrays(ny, dy)
        z, zv = cls._make_coordinate_arrays(nz, dz)
        return cls(x=x, y=y, z=z, xv=xv, yv=yv, zv=zv)

    @property
    def meshgrid(self) -> List[np.ndarray]:
        """Return the coordinate grid representation.

        This returns three 3D arrays containing the z, y, x coordinates
        respectively.  For example,

        >>> Z, Y, X = grid.meshgrid()

        X[zi, yi, xi] is is the x-coordinate of the point at indices (xi, yi,
        zi).  The data returned is a read-only view into the coordinate arrays
        and is efficient to compute on demand.
        """
        return np.meshgrid(self.z, self.y, self.x, indexing='ij', copy=False)

    def delta(self, axis: int) -> np.ndarray:
        """Return grid spacing along the given axis."""
        if axis == 0:
            meshgrid = np.meshgrid(self.zv, self.y, self.x, indexing='ij', copy=False)[axis]
        elif axis == 1:
            meshgrid = np.meshgrid(self.z, self.yv, self.x, indexing='ij', copy=False)[axis]
        elif axis == 2:
            meshgrid = np.meshgrid(self.z, self.y, self.xv, indexing='ij', copy=False)[axis]
        else:
            raise ValueError('Invalid axis provided')

        return np.diff(meshgrid, axis=axis)

    @property
    def shape(self) -> ShapeType:
        return (len(self.z), len(self.y), len(self.x))

    def __len__(self):
        return reduce(lambda x, y: x * y, self.shape, 1)

    def allocate_variable(self, dtype: np.dtype = _dtype_float64) -> np.ndarray:
        """Allocate a numpy array defined over this grid."""
        return np.zeros(self.shape, dtype=dtype)

    def __repr__(self):
        shp = self.shape
        return f'RectangularGrid(nx={shp[2]}, ny={shp[1]}, nz={shp[0]})'

    def save(self, file: H5File) -> None:
        """Save the grid state into an HDF5 file."""
        for dim in ('x', 'xv', 'y', 'yv', 'z', 'zv'):
            d = file.create_dataset(dim, data=getattr(self, dim))
            d.make_scale(dim)

    @classmethod
    def load(cls, file: H5File) -> 'RectangularGrid':
        """Generate a grid object from an existing HDF5 file."""
        kwargs = {}
        for dim in ('x', 'xv', 'y', 'yv', 'z', 'zv'):
            kwargs[dim] = file[dim][:]
        return cls(**kwargs)

    @classmethod
    def _find_dimension_index(cls, vertices: np.ndarray, coordinate: float) -> int:
        indices = (vertices >= coordinate).nonzero()[0]
        if len(indices) == 0:
            return -1
        return indices[0]

    def get_voxel(self, point: Point) -> Voxel:
        """Return the voxel containing the given point.

        For points outside of the grid, this method will return invalid
        indices.  For example, given vertex coordinates [1.5, 2.7, 6.5] and point
        -1.5 or 7.1, this method will return -1 and 3, respectively.  Call the
        the `is_valid_voxel` method to determine if the voxel is valid.
        """
        # For some reason, extracting fields from a recordarray results in a
        # transposed point object (shape (1,3) rather than (3,)).  This code
        # ensures the representation is as expected.
        point = point.ravel().view(Point)
        assert len(point) == 3, 'This method does not handle arrays of points'

        ix = self._find_dimension_index(self.xv, point.x)
        iy = self._find_dimension_index(self.yv, point.y)
        iz = self._find_dimension_index(self.zv, point.z)
        return Voxel(x=ix, y=iy, z=iz)

    def is_valid_voxel(self, voxel: Voxel) -> bool:
        """Return whether or not a voxel index is valid."""
        v = voxel
        return 0 <= v.x <= len(self.x) and 0 <= v.y <= len(self.y) and 0 <= v.z <= len(self.z)
