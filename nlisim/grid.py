r"""
Domain discretization interface.

This module contains defines a common interface for representing the
[discretization](https://en.wikipedia.org/wiki/Discretization) of the 3D
simulation domain.  In this context, the "mesh" is a discrete representation of
the region of 3D space where the simulation occurs (the domain).  The code
will generally assume the domain is the cartesian product of intervals,
\[
    \Omega = [x_0, x_1] \times [y_0, y_1] \times [z_0, z_1].
\]
Users should assume that the units of these quantities are in physical quantities
(nanometers) so that, for example,  \( x_1 - x_0 \) is the length of the `x`-axis
of the domain in nanometers.  There is also no requirement that the lower left
corner of the domain is aligned with the origin.

The mesh breaks the continuous domain up into discrete "voxels" each of which
will be centered about a specific point inside the domain.  In general, the
geometry of these voxels is arbitrary and unstructured, but currently only
[rectangular grids](https://en.wikipedia.org/wiki/Regular_grid) are
implemented.  For these grids, all voxels are hyper-rectangles and are aligned
along the domains axes.  See the `simulation.mesh.RectangularGrid` implementation
for details.
"""
from collections import defaultdict
from enum import IntEnum
from functools import reduce
from itertools import product
from typing import Iterable, Iterator, List, Tuple, Union, cast

from attr import attrib, attrs
from h5py import File as H5File
import numpy as np
from vtkmodules.all import VTK_TETRA, vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

from nlisim.coordinates import Point, Voxel

ShapeType = Tuple[int, int, int]
SpacingType = Tuple[float, float, float]

_dtype_float64 = np.dtype('float64')


class TissueType(IntEnum):
    BRONCHIOLAR_EPITHELIUM = 0
    CAPILLARY = 1
    ALVEOLAR_EPITHELIUM = 2
    ALVEOLAR_SURFACTANT = 3


@attrs(auto_attribs=True, repr=False)
class TetrahedralMesh(object):
    """
    A class representation of a tetrahedral mesh.

    points is an (N,3) array of points in euclidean 3-space

    element_point_indices is an (M,4) numpy array of integers where each row encodes the points
     of a tetrahedron via their index in the points array. M = number of tetrahedra

    element_neighbors is an (M,) numpy array of integers where the row i is a list of the tetrahedra
     which share a face with tetrahedron i via their integer index.
     M = number of tetrahedra

    element_tissue_type is an (M,) numpy array of integers which records what type of tissue is
     present in the geometric cell. M = number of tetrahedra
     e.g. bronchiolar or alveolar epithelium, capillary. Also, "tissue" such as surfactant or air.
     See the TissueType IntEnum for values.

    element_volumes is an (M,) numpy array of floats which records the volumes of each tetrahedron
     M = number of tetrahedra

    point_dual_volumes is an (N,) numpy array of floats which records "the part of the volume
     incident to a point" What we really mean here is the Hodge star operator on 0-forms. That is,
     the map *_0: C_0 ↦ C^3 from functions to 3-forms. Why we need this: to compute the integral
     of functions.
    """

    points: np.ndarray = attrib()
    element_point_indices: np.ndarray = attrib()
    element_neighbors: np.ndarray = attrib()
    element_tissue_type: np.ndarray = attrib()
    element_volumes: np.ndarray = attrib()
    total_volume: float = attrib()
    point_dual_volumes: np.ndarray = attrib()

    @classmethod
    def load(cls, filename: str) -> 'TetrahedralMesh':
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        # noinspection PyArgumentList
        reader.Update()

        data = reader.GetOutput()

        element_type = vtk_to_numpy(data.GetCellTypesArray())
        assert np.all(element_type == VTK_TETRA), f"{filename} is not a tetrahedral mesh!"

        points = vtk_to_numpy(data.GetPoints().GetData())
        points.flags['WRITEABLE'] = False

        element_tissue_type = vtk_to_numpy(data.GetCellData().GetArray("tissue-type"))
        element_tissue_type.flags['WRITEABLE'] = False

        # tetrahedral meshes look like (4, pt, pt, pt, pt, 4, pt, pt, pt, pt, 4, ...)
        element_point_indices = vtk_to_numpy(data.GetCells().GetData()).reshape((-1, 5))[:, 1:]
        element_point_indices.flags['WRITEABLE'] = False

        # element_neighbors: np.ndarray
        # if data.GetCells().HasArray("neighbors") == 1:
        #     # use any precomputed dual
        #     element_neighbors = vtk_to_numpy(data.GetCellData().GetArray("neighbors"))
        # else:

        # TODO: see if vtk has this
        # if we don't have a precomputed 1-skeleton of the dual, compute it now
        # collect all faces and their incident tetrahedra
        face_tets = defaultdict(list)
        for tet_index in range(element_point_indices.shape[0]):
            point_indices = np.array(sorted(element_point_indices[tet_index, :]))
            for omitted_idx in range(4):
                face = tuple(point_indices[[k for k in range(4) if k != omitted_idx]])
                tet_list = face_tets[face]
                tet_list.append(tet_index)
        # read the tetrahedra incidence lists from the faces
        element_neighbors = np.full((element_point_indices.shape[0], 4), -1)
        for face, tets in face_tets.items():
            assert 0 < len(tets) <= 2, f"Not a manifold at face: {face}"
            if len(tets) == 1:
                continue
            tet_a, tet_b = tets
            # insert tet_b into incidence list for tet_a
            idx = np.argmin(element_neighbors[tet_a])
            element_neighbors[tet_a, idx] = tet_b
            # insert tet_a into incidence list for tet_b
            idx = np.argmin(element_neighbors[tet_b])
            element_neighbors[tet_b, idx] = tet_a
        element_neighbors.flags['WRITEABLE'] = False

        # precompute element volumes
        tet_points = points[element_point_indices, :]
        element_volumes = np.abs(
            np.linalg.det((tet_points[:, 1:, :].T - tet_points[:, 0, :].T).T) / 6.0
        )
        element_volumes.flags['WRITEABLE'] = False

        total_volume = float(np.sum(element_volumes))

        # distribute the volume of tetrahedra evenly amongst its points
        point_dual_volumes = np.sum(element_volumes[element_point_indices], axis=1) / 4.0

        return cls(
            points=points,
            element_point_indices=element_point_indices,
            element_neighbors=element_neighbors,
            element_tissue_type=element_tissue_type,
            element_volumes=element_volumes,
            total_volume=total_volume,
            point_dual_volumes=point_dual_volumes,
        )

    def integrate_point_function(self, point_function: np.ndarray) -> Union[np.ndarray, float]:
        """
        Integrate a point function over the mesh.

        Parameters
        ----------
        point_function: np.ndarray
            a function defined on points, expressed as an (N,) or (N,k) numpy array.
            N = number of points

        Returns
        -------
        integral of the point function. Returns as a float if point_function is (N,) and as an
        (k,) numpy array if point_function is (N,k)
        """
        assert (
            point_function.shape[0] == self.point_dual_volumes.shape[0]
        ), f"Dimension mismatch! {point_function.shape} and {self.point_dual_volumes.shape}"

        if len(point_function.shape) == 1:
            return float(np.sum(point_function * self.point_dual_volumes, axis=0))
        else:
            return np.sum(
                point_function
                * np.expand_dims(
                    self.point_dual_volumes,
                    axis=[ax for ax in range(len(point_function.shape)) if ax != 0],
                ),
                axis=0,
            )

    def integrate_point_function_single_element(
        self, element_index: Union[int, np.ndarray], point_function: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Integrate a point function over a single element of the mesh.

        Parameters
        ----------
        element_index: int or np.ndarray
            an element of the mesh or elements if given as an (L,) numpy array of ints
        point_function: np.ndarray
            a function defined on points, expressed as an (N,) or (N,k) numpy array.
            N = number of points

        Returns
        -------
        integral of the point function over the given element(s). If a single element is given,
         returns as a float when point_function is (N,) and as an (k,) numpy array if
         point_function is (N,k). When an (L,) array of elements are passed, an (L,) or (L,k)
         array is returned, respectively.
        """
        assert (
            point_function.shape[0] == self.point_dual_volumes.shape[0]
        ), f"Dimension mismatch! {point_function.shape} and {self.point_dual_volumes.shape}"

        if len(point_function.shape) == 1:
            value = point_function[element_index] * self.point_dual_volumes[element_index]
            if isinstance(element_index, int):
                return float(value)
            else:
                return value
        else:
            return np.sum(
                point_function[element_index]
                * np.expand_dims(
                    self.point_dual_volumes[element_index],
                    axis=[ax for ax in range(len(point_function.shape)) if ax != 0],
                ),
                axis=0,
            )

    def integrate_element_function(self, element_function: np.ndarray) -> Union[np.ndarray, float]:
        """
        Integrate an element function over the mesh.

        Parameters
        ----------
        element_function: np.ndarray
            a function defined on elements, expressed as an (M,) or (M,k) numpy array.
            M = number of elements

        Returns
        -------
        integral of the element function. Returns as a float if element_function is (M,) and as an
        (k,) numpy array if element_function is (M,k)
        """
        assert (
            element_function.shape[0] == self.element_volumes.shape[0]
        ), f"Dimension mismatch! {element_function.shape} and {self.element_volumes.shape}"

        if len(element_function.shape) == 1:
            return float(np.sum(element_function, axis=0))
        else:
            return np.sum(element_function, axis=0)

    def is_in_element(self, element_index: int, point: Point) -> bool:
        """Determine if a given point is in a given element."""
        tet_points = self.points[self.element_point_indices[element_index, :], :]

        try:
            # find position of the point in tetrahedral coordinates
            sln = np.linalg.solve(tet_points[1:, :] - tet_points[0, :], point - tet_points[0, :])
            return np.all(0.0 <= sln) and np.all(sln <= 1.0) and np.sum(sln) <= 1.0
        except np.linalg.LinAlgError:
            assert False, "Bad mesh: contains a singular tetrahedron"

    def get_element_index(self, point: Point) -> int:
        """
        Get the index label of the element containing the point `point`.

        Parameters
        ----------
        point: Point
            a point in 3-space

        Returns
        -------
        integer label of the element, -1 if not in an element
        """
        # TODO: This is the most naïve algorithm, replace it.
        for tet_index in range(self.element_point_indices.shape[0]):
            if self.is_in_element(tet_index, point):
                return True
        return False

    def get_element_tissue_type(self, element_index: int) -> TissueType:
        """
        Get tissue type of an element.

        e.g. blood, epithelium, ...

        Parameters
        ----------
        element_index: int
            integer label of the element

        Returns
        -------
        TissueType (IntEnum) representing the tissue type of the element
        """
        return self.element_tissue_type[element_index]

    def element_volume(self, element_index: int) -> float:
        """
        Get unsigned volume of an element.

        Parameters
        ----------
        element_index: int
            integer label of the element

        Returns
        -------
        float with units equal to (linear units)^3
        """
        return self.element_volumes[element_index]

    def allocate_point_variable(self, dtype: np.DTypeLike = _dtype_float64) -> np.ndarray:
        """Allocate a numpy array defined on the points of this mesh."""
        return np.zeros(self.points.shape, dtype=dtype)

    def allocate_volume_variable(self, dtype: np.DTypeLike = _dtype_float64) -> np.ndarray:
        """Allocate a numpy array defined on the 3-dimensional elements of this mesh."""
        return np.zeros(self.element_point_indices.shape[0], dtype=dtype)

    def get_adjacent_elements(self, element_index: int) -> Iterator[int]:
        """Return all 3-dimensional elements which share a face with the given element."""
        return (idx for idx in self.element_neighbors[element_index, :] if idx != -1)

    def tetrahedral_proportions(self, element_index: int, point: Point) -> np.ndarray:
        tet_points = self.points[self.element_point_indices[element_index, :], :]

        ortho_coords = np.min(
            1.0,
            np.max(
                0.0, np.linalg.solve(tet_points[1:, :] - tet_points[0, :], point - tet_points[0, :])
            ),
        )

        assert 0.0 <= np.sum(ortho_coords) <= 1.0 and np.isclose(
            ((tet_points[1:, :] - tet_points[0, :]) @ ortho_coords) + tet_points[0, :], point
        ), f"Point does not seem to be in the element. {point=} {element_index=} {tet_points=}"

        proportional_coords = np.array(
            [
                1 - ortho_coords[0] - ortho_coords[1] - ortho_coords[2],
                ortho_coords[0],
                ortho_coords[1],
                ortho_coords[2],
            ]
        )

        return proportional_coords


@attrs(auto_attribs=True, repr=False)
class RectangularGrid(object):
    r"""
    A class representation of a rectangular mesh.

    This class breaks the simulation domain into a \(n_x \times n_y \times
    n_z\) array of hyper-rectangles.  As is the case for the full domain, each
    mesh element is cartesian product of intervals,
    \[
        \Omega_{i,j,k} = [x_i, x_{i+1}] \times [y_j, y_{j+1}] \times [z_k, z_{k+1}].
    \]
    In addition, there is a "center" for each mesh cell contained within the
    mesh element,
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
        The `x`-coordinates of the centers of each mesh cell.
    y : np.ndarray
        The `y`-coordinates of the centers of each mesh cell.
    z : np.ndarray
        The `z`-coordinates of the centers of each mesh cell.
    xv : np.ndarray
        The `x`-coordinates of the edges of each mesh cell.
    yv : np.ndarray
        The `y`-coordinates of the edges of each mesh cell.
    zv : np.ndarray
        The `z`-coordinates of the edges of each mesh cell.

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
        """Create a rectangular mesh with uniform spacing in each axis."""
        nz, ny, nx = shape
        dz, dy, dx = spacing
        x, xv = cls._make_coordinate_arrays(nx, dx)
        y, yv = cls._make_coordinate_arrays(ny, dy)
        z, zv = cls._make_coordinate_arrays(nz, dz)
        return cls(x=x, y=y, z=z, xv=xv, yv=yv, zv=zv)

    @property
    def meshgrid(self) -> List[np.ndarray]:
        # noinspection PyUnresolvedReferences
        """Return the coordinate mesh representation.

        This returns three 3D arrays containing the z, y, x coordinates
        respectively.  For example,

        >>> Z, Y, X = mesh.meshgrid()

        `X[zi, yi, xi]` is is the x-coordinate of the point at indices `(xi, yi,
        zi)`.  The data returned is a read-only view into the coordinate arrays
        and is efficient to compute on demand.
        """
        # noinspection PyTypeChecker
        return np.meshgrid(self.z, self.y, self.x, indexing='ij', copy=False)

    def delta(self, axis: int) -> np.ndarray:
        """Return mesh spacing along the given axis."""
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
        return len(self.z), len(self.y), len(self.x)

    def __len__(self):
        return reduce(lambda x, y: x * y, self.shape, 1)

    def allocate_variable(self, dtype: np.dtype = _dtype_float64) -> np.ndarray:
        """Allocate a numpy array defined over this mesh."""
        return np.zeros(self.shape, dtype=dtype)

    def __repr__(self):
        shp = self.shape
        return f'RectangularGrid(nx={shp[2]}, ny={shp[1]}, nz={shp[0]})'

    def save(self, file: H5File) -> None:
        """Save the mesh state into an HDF5 file."""
        for dim in ('x', 'xv', 'y', 'yv', 'z', 'zv'):
            d = file.create_dataset(dim, data=getattr(self, dim))
            d.make_scale(dim)

    @classmethod
    def load(cls, file: H5File) -> 'RectangularGrid':
        """Generate a mesh object from an existing HDF5 file."""
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
        """Return the flattened index of a voxel inside the mesh.

        This is a convenience method that wraps numpy.ravel_multi_index.
        """
        return np.ravel_multi_index(cast(Tuple[int, int, int], voxel), self.shape)

    def voxel_from_flattened_index(self, index: int) -> 'Voxel':
        """Create a Voxel from flattened index of the mesh.

        This is a convenience method that wraps numpy.unravel_index.
        """
        z, y, x = np.unravel_index(index, self.shape)
        return Voxel(x=float(x), y=float(y), z=float(z))

    def get_voxel(self, point: Point) -> Voxel:
        """Return the voxel containing the given point.

        For points outside of the mesh, this method will return invalid
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
