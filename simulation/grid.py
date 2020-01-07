r"""
Domain discretization interface.

This module contains defines a common interface for representing the
[discretization](https://en.wikipedia.org/wiki/Discretization) of the 3D
simulation domain.  In this context, the "grid" is a discrete representation of
the region of 3D space where the simulation occurs (the domain).  The code
will generally assume the domain is the cartesian product of intervals,
\[
    \Omega = [x_0, x_1] \times [y_0, y_1] \times [z_0, z_1].
\]
Users should assume that the units of these quantities are in physical quantities
(nanometers) so that, for example,  \( x_1 - x_0 \) is the length of the `x`-axis
of the domain in nanometers.  There is also no requirement that the lower left
corner of the domain is aligned with the origin.

The grid breaks the continuous domain up into discrete "voxels" each of which
will be centered about a specific point inside the domain.  In general, the
geometry of these voxels is arbitrary and unstructured, but currently only
[rectangular grids](https://en.wikipedia.org/wiki/Regular_grid) are
implemented.  For these grids, all voxels are hyper-rectangles and are aligned
along the domains axes.  See the `simulation.grid.RectangularGrid` implementation
for details.
"""
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
    r"""
    A class representation of a rectangular grid.

    This class breaks the simulation domain into a \(n_x \times n_y \times
    n_z\) array of hyper-rectangles.  As is the case for the full domain, each
    grid element is cartesian product of intervals,
    \[
        \Omega_{i,j,k} = [x_i, x_{i+1}] \times [y_j, y_{j+1}] \times [z_k, z_{k+1}].
    \]
    In addition, there is a "center" for each grid cell contained within the
    grid element,
    \[
        (\bar{x}_i, \bar{y}_j, \bar{z}_k) \in \Omega_{i,j,k}.
    \]
    For a perticular function defined over the domain, \(f\), the
    descritization is defined relative to this point,
    \[
        f_{i,j,k} := f(\bar{x}_i, \bar{y}_j, \bar{z}_k)
    \]
    This center is usually (but not required to be) the true center of the
    interval.

    As a concrete example, if you are representing a "gridded variable"
    such as the iron concentration throughout the domain as a numpy array, `iron`,
    then the value of `iron[k, j, i]` would be the iron concentration at the point
    \((\bar{x}_i, \bar{y}_j, \bar{z}_k)\).

    This class is initialized with several arrays of numbers representing these
    coordinates.  Considering just the `x`-axis, the argument `x` is an array
    of length \(n_x\).  The element of this array at index `i` represents the
    value \(\bar{x}_i\).  In other words, this array contains coordinate of the
    center of all cells in the `x` direction.

    The `xv` argument contains the coordinates of the edges of each cell.  This
    array should be of length \(n_x + 1\) or one element larger than `x`.  The
    element at index `i` in this array represents the value \(x_i\) or the left
    edge of the element.  Because the elements are all aligned, this is also the
    right edge of the previous element.

    Parameters
    ----------
    x : np.ndarray
        The `x`-coordinates of the centers of each grid cell.
    y : np.ndarray
        The `y`-coordinates of the centers of each grid cell.
    z : np.ndarray
        The `z`-coordinates of the centers of each grid cell.
    xv : np.ndarray
        The `x`-coordinates of the edges of each grid cell.
    yv : np.ndarray
        The `y`-coordinates of the edges of each grid cell.
    zv : np.ndarray
        The `z`-coordinates of the edges of each grid cell.

    """

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

        `X[zi, yi, xi]` is is the x-coordinate of the point at indices `(xi, yi,
        zi)`.  The data returned is a read-only view into the coordinate arrays
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
        indices.  For example, given vertex coordinates `[1.5, 2.7, 6.5]` and point
        `-1.5` or `7.1`, this method will return `-1` and `3`, respectively.  Call the
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
