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
from itertools import product
from typing import Iterable, Iterator, List, Tuple, cast

import attr
from h5py import File as H5File
import numpy as np

from nlisim.coordinates import Point, Voxel

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
    For a particular function defined over the domain, \(f\), the
    discretization is defined relative to this point,
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
        return indices[0] - 1

    def get_flattened_index(self, voxel: Voxel):
        """Return the flattened index of a voxel inside the grid.

        This is a convenience method that wraps numpy.ravel_multi_index.
        """
        return np.ravel_multi_index(cast(Tuple[int, int, int], voxel), self.shape)

    def voxel_from_flattened_index(self, index: int) -> 'Voxel':
        """Create a Voxel from flattened index of the grid.

        This is a convenience method that wraps numpy.unravel_index.
        """
        z, y, x = np.unravel_index(index, self.shape)
        return Voxel(x=float(x), y=float(y), z=float(z))

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

    def get_voxel_center(self, voxel: Voxel) -> Point:
        """Get the coordinates of the center point of a voxel."""
        return Point(x=self.x[voxel.x], y=self.y[voxel.y], z=self.z[voxel.z])

    def is_valid_voxel(self, voxel: Voxel) -> bool:
        """Return whether or not a voxel index is valid."""
        v = voxel
        return 0 <= v.x < len(self.x) and 0 <= v.y < len(self.y) and 0 <= v.z < len(self.z)

    def is_point_in_domain(self, point: Point) -> bool:
        """Return whether or not a point in inside the domain."""
        return (
            (self.xv[0] <= point.x <= self.xv[-1])
            and (self.yv[0] <= point.y <= self.yv[-1])
            and (self.zv[0] <= point.z <= self.zv[-1])
        )

    def get_adjacent_voxels(self, voxel: Voxel, corners: bool = False) -> Iterator[Voxel]:
        """Return an iterator over all neighbors of a given voxel.

        Parameters
        ----------
        voxel : simulation.coordinates.Voxel
            The target voxel
        corners : bool
            Include voxels sharing corners and edges in addition to those sharing sides.

        """
        dirs: Iterable[Tuple[int, int, int]]
        if corners:
            dirs = filter(lambda x: x != (0, 0, 0), product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
        else:
            dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

        for di, dj, dk in dirs:
            i = voxel.x + di
            j = voxel.y + dj
            k = voxel.z + dk
            neighbor = Voxel(x=i, y=j, z=k)

            if self.is_valid_voxel(neighbor):
                yield neighbor

    def get_nearest_voxel(self, point: Point) -> Voxel:
        """Return the nearest voxel to a given point.

        Parameters
        ----------
        point : simulation.coordinates.Point
            The target point

        """
        if not self.is_point_in_domain(point):
            raise NotImplementedError(
                'Getting the closest domain voxel to a point outside the domain is not implemented'
            )
        return self.get_voxel(point)

    def get_voxels_in_range(self, point: Point, distance: float) -> Iterator[Tuple[Voxel, float]]:
        """Return an iterator of voxels within a given distance of a point.

        The values returned by the iterator are tuples of `(Voxel, distance)`
        pairs.  For example,

            voxel, distance = next(self.get_voxels_in_range(point, 1))

        where `distance` is the distance from `point` to the center of `voxel`.

        Note: no guarantee is given to the order over which the voxels are
        iterated.

        Parameters
        ----------
        point : simulation.coordinates.Point
            The center point
        distance : float
            Return all voxels with centers less than the distance from the center point

        """
        # Get a hyper-square containing a superset of what we want.  This
        # restricts the set of points that we need to explicitly compute.
        dp = Point(x=distance, y=distance, z=distance)
        z0, y0, x0 = point - dp
        z1, y1, x1 = point + dp

        x0 = max(x0, self.x[0])
        x1 = min(x1, self.x[-1])

        y0 = max(y0, self.y[0])
        y1 = min(y1, self.y[-1])

        z0 = max(z0, self.z[0])
        z1 = min(z1, self.z[-1])

        # get voxel indices of the lower left and upper right corners
        k0, j0, i0 = self.get_voxel(Point(x=x0, y=y0, z=z0))
        k1, j1, i1 = self.get_voxel(Point(x=x1, y=y1, z=z1))

        # get a distance matrix over all voxels in the candidate range
        z, y, x = self.meshgrid
        dx = x[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1] - point.x
        dy = y[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1] - point.y
        dz = z[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1] - point.z
        distances = np.sqrt(dx * dx + dy * dy + dz * dz)

        # iterate over all voxels and yield those in range
        for k in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                for i in range(distances.shape[2]):
                    d = distances[k, j, i]
                    if d <= distance:
                        yield Voxel(x=(i + i0), y=(j + j0), z=(k + k0)), d
