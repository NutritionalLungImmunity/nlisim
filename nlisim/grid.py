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
from itertools import combinations, product
from typing import Dict, Iterable, Iterator, List, Set, Tuple, Union, cast, overload

from attr import attrib, attrs
from h5py import File as H5File
import meshio
import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_matrix, dia_matrix, diags, dok_matrix
from vtkmodules.all import VTK_TETRA, vtkUnstructuredGridReader, vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

from nlisim.coordinates import Point, Voxel
from nlisim.random import rg
from nlisim.tetsearch import TetrahedronSearchTree, TreeNode
from nlisim.util import logger

ShapeType = Tuple[int, int, int]
SpacingType = Tuple[float, float, float]

_dtype_float64 = np.dtype('float64')


class TissueType(IntEnum):
    AIR = -1
    BRONCHIOLAR_EPITHELIUM = 0
    CAPILLARY = 1
    ALVEOLAR_EPITHELIUM = 2
    ALVEOLAR_SURFACTANT = 3


@attrs(auto_attribs=True, repr=False)
class TetrahedralMesh(object):
    """
    A class representation of a tetrahedral mesh.

    points is an (N,3) array of points in euclidean 3-space. All x,y,z coordinates are in µm

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

    laplacian is a sparse (N,N) matrix o floats which encodes the discrete laplace operator on
     functions (0-forms)
    """

    points: np.ndarray = attrib()
    element_point_indices: np.ndarray = attrib()
    element_neighbors: np.ndarray = attrib()
    element_tissue_type: np.ndarray = attrib()
    element_volumes: np.ndarray = attrib()
    total_volume: float = attrib()
    point_dual_volumes: np.ndarray = attrib()
    laplacian: csr_matrix = attrib()
    hodge_star_0: dia_matrix = attrib()
    point_search_tree: TetrahedronSearchTree = attrib()

    @classmethod
    def load_hdf5(cls, file: H5File) -> 'TetrahedralMesh':
        points = np.array(file['points'], dtype=_dtype_float64)
        element_point_indices = np.array(file['element_point_indices'], dtype=int)
        element_tissue_type = np.array(file['element_tissue_type'], dtype=int)
        (
            element_neighbors,
            element_volumes,
            point_dual_volumes,
            total_volume,
            laplacian,
            hodge_star_0,
        ) = cls.compute_derived_quantities(element_point_indices, points)
        return cls(
            points=points,
            element_point_indices=element_point_indices,
            element_neighbors=element_neighbors,
            element_tissue_type=element_tissue_type,
            element_volumes=element_volumes,
            total_volume=total_volume,
            point_dual_volumes=point_dual_volumes,
            laplacian=laplacian,
            hodge_star_0=hodge_star_0,
            point_search_tree=TetrahedronSearchTree(
                points=points, element_point_indices=element_point_indices
            ),
        )

    def save_hdf5(self, file: H5File) -> None:
        for dim in ('points', 'element_point_indices', 'element_tissue_type'):
            file.create_dataset(dim, data=getattr(self, dim))

    def as_meshio(self) -> meshio.Mesh:
        """
        Create a meshio Mesh for this mesh.

        No fields are added.

        Returns
        -------
        a meshio.Mesh
        """
        mesh = meshio.Mesh(points=self.points, cells={'tetra': self.element_point_indices})

        return mesh

    @classmethod
    def load_vtk(cls, filename: str) -> 'TetrahedralMesh':
        if filename[-4:] != ".vtu":
            logger.warning(
                f"{filename} is not a vtu file, assuming it is a vtk unstructured grid. "
                "This might be wrong!"
            )
            reader = vtkUnstructuredGridReader()
        else:
            reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
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

        (
            element_neighbors,
            element_volumes,
            point_dual_volumes,
            total_volume,
            laplacian,
            hodge_star_0,
        ) = cls.compute_derived_quantities(element_point_indices, points)

        return cls(
            points=points,
            element_point_indices=element_point_indices,
            element_neighbors=element_neighbors,
            element_tissue_type=element_tissue_type,
            element_volumes=element_volumes,
            total_volume=total_volume,
            point_dual_volumes=point_dual_volumes,
            laplacian=laplacian,
            hodge_star_0=hodge_star_0,
            point_search_tree=TetrahedronSearchTree(
                points=points, element_point_indices=element_point_indices
            ),
        )

    @classmethod
    def compute_derived_quantities(
        cls, element_point_indices: np.ndarray, points
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, csr_matrix, dia_matrix]:

        # TODO: see if vtk has this
        # if we don't have a precomputed 1-skeleton of the dual, compute it now

        # collect all faces and their incident tetrahedra
        # face_tets: dictionary from 3-tuples of point-indices representing triangles to
        # the indices of tetrahedra which contain them.
        face_tets: Dict = defaultdict(list)
        for tet_index in range(element_point_indices.shape[0]):
            point_indices = np.array(sorted(element_point_indices[tet_index, :]))
            for omitted_idx in range(4):
                face = tuple(point_indices[[k for k in range(4) if k != omitted_idx]])
                tet_list = face_tets[face]
                tet_list.append(tet_index)

        # read the tetrahedra incidence lists from the faces. each tetrahedron may be incident
        # to up to 4 other tetrahedra
        # element_neighbors[i] = (4,) array of indices of tetrahedra incident to i-th tetrahedron,
        # filled with -1's if fewer than 4 neighbors
        element_neighbors = np.full((element_point_indices.shape[0], 4), -1)
        for face, tets in face_tets.items():
            assert (
                0 < len(tets) <= 2
            ), f"Not a manifold at face {face}: {len(tets)} tetrahedra share this face"
            if len(tets) == 1:
                continue
            tet_a, tet_b = tets
            # insert tet_b into incidence list for tet_a
            idx = np.argmin(element_neighbors[tet_a])
            element_neighbors[tet_a, idx] = tet_b
            # insert tet_a into incidence list for tet_b
            idx = np.argmin(element_neighbors[tet_b])
            element_neighbors[tet_b, idx] = tet_a
        element_neighbors.flags['WRITEABLE'] = False  # type: ignore

        # precompute element volumes
        # tet_points: (M,4,3) array
        # with indices: tetrahedral index, index of point _in_ tetrahedron, xyz
        tet_points = points[element_point_indices, :]
        element_volumes = np.abs(
            np.linalg.det((tet_points[:, 1:, :].T - tet_points[:, 0, :].T).T) / 6.0
        )  # unit note: (µm)^3
        element_volumes.flags['WRITEABLE'] = False

        total_volume = float(np.sum(element_volumes))

        # distribute the volume of tetrahedra evenly amongst its points
        point_dual_volumes = np.zeros(points.shape[0], dtype=np.float64)
        np.add.at(point_dual_volumes, element_point_indices.T, element_volumes / 4.0)

        # For details of following diagonal Hodge Star operators, see 4.8.4 of
        # https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf

        # sparse diagonal Hodge star matrix *_0: Ω^0 → Ω^{0*}
        hodge_star_0 = diags(point_dual_volumes)  # unit note: (µm)^3

        # inverse of *_0
        hodge_star_0_inv = diags(1.0 / point_dual_volumes)  # unit note: 1/(µm)^3

        # 1-forms are defined on edges
        edge_indices, edges = cls.get_edges(element_point_indices)

        # compute diagonal Hodge star operator on 1-forms, *_1,
        # in terms of edge lengths and dual areas
        edge_lengths = np.linalg.norm(
            points[edges[:, 0]] - points[edges[:, 1]], axis=1
        )  # unit note: µm
        edge_dual_areas = cls.get_edge_dual_areas(
            edge_indices, edges, element_point_indices, points
        )  # unit note: (µm)^2
        # sparse diagonal Hodge star matrix *_1: Ω^1 → Ω^{1*}
        hodge_star_1 = diags(edge_dual_areas / edge_lengths)

        # d_0, exterior derivative on functions (d_0: Ω^0 → Ω^1)
        d_0 = dok_matrix((edges.shape[0], points.shape[0]), dtype=np.float64)
        for edge_idx, edge in enumerate(edges):
            d_0[edge_idx, edge[1]] = 1.0
            d_0[edge_idx, edge[0]] = -1.0

        laplacian: csr_matrix = hodge_star_0_inv @ d_0.transpose() @ hodge_star_1 @ d_0

        return (
            element_neighbors,
            element_volumes,
            point_dual_volumes,
            total_volume,
            laplacian,
            hodge_star_0,
        )

    @classmethod
    def get_edge_dual_areas(cls, edge_indices, edges, element_point_indices, points):
        # go through the tetrahedra to compute the edge duals
        edge_dual_areas: np.ndarray = np.zeros(shape=edges.shape[0], dtype=np.float64)
        for tet_index, idx_pair in product(
            range(element_point_indices.shape[0]), combinations(range(4), 2)
        ):
            idx_pair_complement = tuple(sorted(set(range(4)) - set(idx_pair)))
            edge = element_point_indices[tet_index, idx_pair]
            edge_midpoint = np.mean(points[edge], axis=0)
            # compute the area of the triangle whose points are 1) the midpoint of the edge and
            # 2) the other two points of the tetrahedron
            area = (
                np.linalg.norm(
                    np.cross(
                        *(
                            points[element_point_indices[tet_index, idx_pair_complement]]
                            - edge_midpoint
                        )
                    )
                )
                / 2.0  # unit note: (µm)^2
            )
            edge_dual_areas[edge_indices[tuple(edge)]] += area / 3.0
        return edge_dual_areas

    @classmethod
    def get_edges(cls, element_point_indices: np.ndarray):
        edge_set: Set[Tuple] = set()
        for idx_pair in combinations(range(4), 2):
            edge_set.update(map(tuple, element_point_indices[:, idx_pair]))
        edges = np.array(sorted(list(edge_set)))
        edge_indices = {tuple(e): n for n, e in enumerate(edges)}  # edge index lookup
        return edge_indices, edges

    @overload
    def evaluate_point_function(
        self, *, point_function: np.ndarray, point: Point, element_index: int
    ) -> float:
        ...

    @overload
    def evaluate_point_function(
        self, *, point_function: np.ndarray, point: Point, element_index: np.ndarray
    ) -> np.ndarray:
        ...

    def evaluate_point_function(
        self, *, point_function: np.ndarray, point: Point, element_index: Union[int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Evaluate a point function on the interior of a tetrahedral element.

        Parameters
        ----------
        point_function: np.ndarray
            a function defined on the points of the mesh
        point: Point
            a point in a tetrahedron. no checking that this is in the element is performed.
        element_index: int
            the index of the tetrahedral element.

        Returns
        -------
        Value of the function, defined using linear interpolation.
        """
        proportions = self.tetrahedral_proportions(element_index=element_index, point=point)

        if isinstance(element_index, np.ndarray):
            return np.einsum(
                'ij,ij->i', point_function[self.element_point_indices[element_index]], proportions
            )
        else:
            return point_function[self.element_point_indices[element_index]] @ proportions

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
            self.hodge_star_0.shape[1] == point_function.shape[0]
        ), f"Dimension mismatch! {self.hodge_star_0.shape} and {point_function.shape}"

        result = np.sum(self.hodge_star_0 @ point_function, axis=0) / 3.0

        # convert to float, if possible
        if len(point_function.shape) > 1:
            return result
        else:
            return float(result)

    def integrate_point_function_in_element(
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
            self.hodge_star_0.shape[1] == point_function.shape[0]
        ), f"Dimension mismatch! {self.hodge_star_0.shape} and {point_function.shape}"

        if isinstance(element_index, np.ndarray) and element_index.shape == (1,):
            element_index = int(element_index)

        if isinstance(element_index, (int, np.integer)):
            point_indices = self.element_point_indices[element_index]  # (4,)
            values = point_function[point_indices]  # (4,)
            return float(np.mean(values) * self.element_volumes[element_index])
        else:
            point_indices = self.element_point_indices[element_index]  # (L,4)
            values = point_function[point_indices]  # (L,4)
            return np.mean(values, axis=1) * self.element_volumes[element_index]

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

    def elements_incident_to(self, points: Union[int, np.ndarray]) -> np.ndarray:
        """
        Compute an array of element indices that are incident to a collection of points.

        Parameters
        ----------
        points: int or np.ndarray
            a point index or points indices

        Returns
        -------
        the indices of any tetrahedral elements that include at least one of the points given
          in `points`

        """
        if isinstance(points, int) or points.shape == () or points.shape == (1,):
            return np.where(np.any(self.element_point_indices == points, axis=1))[0]
        else:
            return np.where(
                np.any(self.element_point_indices[:, :, np.newaxis] == points, axis=(1, 2))
            )[0]

    def _find_tet(
        self, point: Union[Point, np.ndarray], tree_node: TreeNode, dimension: int = 0
    ) -> int:
        if tree_node.leaf:
            assert tree_node.tetrahedra_indices is not None
            for element_idx in tree_node.tetrahedra_indices:
                if self.in_element(element_idx, point=point):
                    return element_idx
            return -1

        assert tree_node.left is not None
        assert tree_node.center is not None
        assert tree_node.right is not None

        if tree_node.left.min <= point[dimension] <= tree_node.left.max:
            left_result = self._find_tet(
                point=point, tree_node=tree_node.left.node, dimension=(dimension + 1) % 3
            )
            if left_result >= 0:
                return left_result

        if tree_node.center.min <= point[dimension] <= tree_node.center.max:
            center_result = self._find_tet(
                point=point, tree_node=tree_node.center.node, dimension=(dimension + 1) % 3
            )
            if center_result >= 0:
                return center_result

        if tree_node.right.min <= point[dimension] <= tree_node.right.max:
            right_result = self._find_tet(
                point=point, tree_node=tree_node.right.node, dimension=(dimension + 1) % 3
            )
            if right_result >= 0:
                return right_result

        return -1

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
        return self._find_tet(np.squeeze(point), self.point_search_tree.sorted_tree, dimension=0)

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

    def allocate_point_variable(self, dtype: DTypeLike = _dtype_float64) -> np.ndarray:
        """Allocate a numpy array defined on the points of this mesh."""
        return np.zeros(self.points.shape[0], dtype=dtype)

    def allocate_volume_variable(self, dtype: DTypeLike = _dtype_float64) -> np.ndarray:
        """Allocate a numpy array defined on the 3-dimensional elements of this mesh."""
        return np.zeros(self.element_point_indices.shape[0], dtype=dtype)

    def get_adjacent_elements(self, element_index: int) -> Iterator[int]:
        """Return all 3-dimensional elements which share a face with the given element."""
        return (idx for idx in self.element_neighbors[element_index, :] if idx != -1)

    def tetrahedral_proportions(
        self, element_index: Union[int, np.ndarray], point: Point
    ) -> np.ndarray:
        # TODO: good candidate for tests
        try:
            element_index = int(element_index)
        except TypeError:
            pass

        tet_points = self.points[self.element_point_indices[element_index]]

        try:
            if isinstance(element_index, (int, np.integer)):
                ortho_coords = np.linalg.solve(
                    (tet_points[1:, :] - tet_points[0, :]).T, point - tet_points[0, :]
                )

                proportional_coords = np.array(
                    [
                        1 - np.sum(ortho_coords, axis=0),
                        *ortho_coords,
                    ]
                )

                return proportional_coords
            else:
                ortho_coords = np.linalg.solve(
                    np.transpose(
                        tet_points[:, 1:, :] - np.expand_dims(tet_points[:, 0, :], axis=1),
                        axes=[0, 2, 1],
                    ),
                    point - tet_points[:, 0, :],
                )
                proportional_coords = np.array(
                    [
                        1 - np.sum(ortho_coords, axis=1),
                        ortho_coords[:, 0],
                        ortho_coords[:, 1],
                        ortho_coords[:, 2],
                    ]
                ).T

                return proportional_coords
        except np.linalg.LinAlgError as e:
            raise AssertionError(str(e) + "; Bad mesh: contains a singular tetrahedron")

    # removed as a de-duplication, TODO: permanent removal
    # def is_in_element(self, element_index: int, point: Point) -> bool:
    #     """Determine if a given point is in a given element."""
    #     tet_points = self.points[self.element_point_indices[element_index, :], :]
    #
    #     try:
    #         # find position of the point in tetrahedral coordinates
    #         sln = np.linalg.solve((tet_points[1:, :] - tet_points[0, :]).T,
    #                               point - tet_points[0, :])
    #         return np.all(0.0 <= sln) and np.all(sln <= 1.0) and np.sum(sln) <= 1.0
    #     except np.linalg.LinAlgError:
    #         raise AssertionError("Bad mesh: contains a singular tetrahedron")

    def in_element(self, element_index: int, point: Point, interior: bool = False) -> bool:
        tet_proportions = self.tetrahedral_proportions(element_index, point)
        # by construction, np.sum(tet_proportions) == 1.0 (approx), so failure to be in the
        # tetrahedron will only happen when one of the coords is negative

        return np.all(0.0 <= tet_proportions) and (
            (not interior)
            or (np.alltrue(0.0 < tet_proportions) and np.alltrue(tet_proportions < 1.0))
        )

    @classmethod
    def construct_uniform(cls, shape: ShapeType, spacing: SpacingType) -> 'TetrahedralMesh':
        xs = np.linspace(0, spacing[0] * shape[0], shape[0] + 1)
        ys = np.linspace(0, spacing[1] * shape[1], shape[1] + 1)
        zs = np.linspace(0, spacing[2] * shape[2], shape[2] + 1)
        points = np.array(list(product(xs, ys, zs)), dtype=np.float64)

        def pt_idx(i_idx: int, j_idx: int, k_idx: int) -> int:
            return k_idx + (shape[2] + 1) * (j_idx + (shape[1] + 1) * i_idx)

        element_point_indices_list = []
        for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
            # the [0,1]^3 cube splits into 6 tetrahedra:
            # 1) [0,0,0], [0,0,1], [0,1,1], [1,1,1]
            element_point_indices_list.append(
                [
                    pt_idx(i + 0, j + 0, k + 0),
                    pt_idx(i + 0, j + 0, k + 1),
                    pt_idx(i + 0, j + 1, k + 1),
                    pt_idx(i + 1, j + 1, k + 1),
                ]
            )
            # 2) [0,0,0], [0,0,1], [1,0,1], [1,1,1]
            element_point_indices_list.append(
                [
                    pt_idx(i + 0, j + 0, k + 0),
                    pt_idx(i + 0, j + 0, k + 1),
                    pt_idx(i + 1, j + 0, k + 1),
                    pt_idx(i + 1, j + 1, k + 1),
                ]
            )
            # 3) [0,0,0], [0,1,0], [0,1,1], [1,1,1]
            element_point_indices_list.append(
                [
                    pt_idx(i + 0, j + 0, k + 0),
                    pt_idx(i + 0, j + 1, k + 0),
                    pt_idx(i + 0, j + 1, k + 1),
                    pt_idx(i + 1, j + 1, k + 1),
                ]
            )
            # 4) [0,0,0], [0,1,0], [1,1,0], [1,1,1]
            element_point_indices_list.append(
                [
                    pt_idx(i + 0, j + 0, k + 0),
                    pt_idx(i + 0, j + 1, k + 0),
                    pt_idx(i + 1, j + 1, k + 0),
                    pt_idx(i + 1, j + 1, k + 1),
                ]
            )
            # 5) [0,0,0], [1,0,0], [1,0,1], [1,1,1]
            element_point_indices_list.append(
                [
                    pt_idx(i + 0, j + 0, k + 0),
                    pt_idx(i + 1, j + 0, k + 0),
                    pt_idx(i + 1, j + 0, k + 1),
                    pt_idx(i + 1, j + 1, k + 1),
                ]
            )
            # 6) [0,0,0], [1,0,0], [1,1,0], [1,1,1]
            element_point_indices_list.append(
                [
                    pt_idx(i + 0, j + 0, k + 0),
                    pt_idx(i + 1, j + 0, k + 0),
                    pt_idx(i + 1, j + 1, k + 0),
                    pt_idx(i + 1, j + 1, k + 1),
                ]
            )
            # you can divide the cube into fewer, but this set match up as triangles on the faces.
        element_point_indices = np.array(element_point_indices_list, dtype=int)

        element_tissue_type = np.full(
            shape=element_point_indices.shape[0], fill_value=TissueType.ALVEOLAR_EPITHELIUM
        )
        (
            element_neighbors,
            element_volumes,
            point_dual_volumes,
            total_volume,
            laplacian,
            hodge_star_0,
        ) = cls.compute_derived_quantities(element_point_indices, points)
        return cls(
            points=points,
            element_point_indices=element_point_indices,
            element_neighbors=element_neighbors,
            element_tissue_type=element_tissue_type,
            element_volumes=element_volumes,
            total_volume=total_volume,
            point_dual_volumes=point_dual_volumes,
            laplacian=laplacian,
            hodge_star_0=hodge_star_0,
            point_search_tree=TetrahedronSearchTree(
                points=points, element_point_indices=element_point_indices
            ),
        )


@attrs(auto_attribs=True, repr=False)
class RectangularGrid(object):
    # noinspection PyUnresolvedReferences
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
        vertex.flags['WRITEABLE'] = False  # type: ignore
        cell.flags['WRITEABLE'] = False  # type: ignore
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

        For points outside the mesh, this method will return invalid
        indices.  For example, given vertex coordinates `[1.5, 2.7, 6.5]` and point
        `-1.5` or `7.1`, this method will return `-1` and `3`, respectively.  Call the
        `is_valid_voxel` method to determine if the voxel is valid.
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
        """Return whether a voxel index is valid."""
        v = voxel
        return 0 <= v.x < len(self.x) and 0 <= v.y < len(self.y) and 0 <= v.z < len(self.z)

    def is_point_in_domain(self, point: Point) -> bool:
        """Return whether a point in inside the domain."""
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


def choose_voxel_by_prob(
    voxels: Tuple[Voxel, ...], default_value: Voxel, weights: np.ndarray
) -> Voxel:
    """
    Choose a voxels using a non-normalized probability distribution.

    If weights are all zero, the default value is chosen.

    Parameters
    ----------
    voxels
        an tuple of voxels
    default_value
        default return value for when weights are uniformly zero
    weights
        an array of non-negative (unchecked) un-normalized probabilities/weights for the voxels

    Returns
    -------
    a Voxel, from voxels, chosen by the probability distribution, or the default
    """
    normalization_constant = np.sum(weights)
    if normalization_constant <= 0:
        # e.g. if all neighbors are air
        return default_value

    # prepend a zero to detect 'failure by zero' in the argmax below
    normalized_weights = np.concatenate((np.array([0.0]), weights / normalization_constant))

    # sample from distribution given by normalized weights
    random_voxel_idx: int = int(np.argmax(np.cumsum(normalized_weights) - rg.uniform() > 0.0) - 1)
    if random_voxel_idx < 0:
        # the only way the 0th could be chosen is by argmax failing
        return default_value
    else:
        return voxels[random_voxel_idx]


def secrete_in_element(
    *,
    mesh: TetrahedralMesh,
    point_field: np.ndarray,
    element_index: Union[int, np.ndarray],
    point: Union[Point, np.ndarray],
    amount: Union[float, np.ndarray],  # units: atto-mol
) -> None:
    proportions = mesh.tetrahedral_proportions(element_index, point)
    points = mesh.element_point_indices[element_index]
    # new pt concentration = (old pt amount + new amount) / pt dual volume
    #    = (old conc * pt dual volume + new amount) / pt dual volume
    #    = old conc + (new amount / pt dual volume)
    np.add.at(
        point_field, points, proportions * amount / mesh.point_dual_volumes[points]
    )  # units: prop * atto-mol / L = atto-M


def uptake_proportionally(
    *,
    mesh: TetrahedralMesh,
    point_field: np.ndarray,  # units: atto-M
    element_index: Union[int, np.ndarray],
    point: Union[Point, np.ndarray],
    k: float,  # units: L * cell^-1 * step^-1
) -> float:
    points = mesh.element_point_indices[element_index]
    amount_around_points = point_field[points] * mesh.point_dual_volumes[points]
    point_field_proportions = mesh.tetrahedral_proportions(element_index=element_index, point=point)
    uptake_amounts = k * amount_around_points * point_field_proportions

    np.subtract.at(
        point_field, points, uptake_amounts / mesh.point_dual_volumes[points]
    )  # units: atto-mol / L = atto-M

    return float(np.sum(uptake_amounts))


def uptake_in_element(
    *,
    mesh: TetrahedralMesh,
    point_field: np.ndarray,  # units: atto-M
    element_index: Union[int, np.ndarray],
    point: Union[Point, np.ndarray],
    amount: Union[float, np.ndarray],  # units: atto-mol
) -> None:
    points = mesh.element_point_indices[element_index]
    # TODO: justify
    point_field_proportions = mesh.tetrahedral_proportions(element_index=element_index, point=point)

    # new pt concentration = (old pt amount + new amount) / pt dual volume
    #    = (old conc * pt dual volume + new amount) / pt dual volume
    #    = old conc + (new amount / pt dual volume)
    # state.log.debug(f"{point_field_proportions=}")
    # state.log.debug(f"{amount=}")
    # state.log.debug(f"{mesh.point_dual_volumes[points]=}")
    assert np.all(0.0 <= point_field_proportions) and np.all(
        point_field_proportions <= 1.0
    ), f"{point_field_proportions=}"

    np.subtract.at(
        point_field, points, point_field_proportions * amount / mesh.point_dual_volumes[points]
    )  # units: proportion * atto-mol / L = atto-M


def sample_point_from_simplex(num_points: int = 1, dimension: int = 3) -> np.ndarray:
    """
    Generate a uniformly distributed random point from a simplex in probability coordinates.

    Parameters
    ----------
    num_points: int
        The number of sample points to return
    dimension: int
        The dimension of the simplex. e.g. a triangle has dimension 2 and tetrahedron has
        dimension 3. Default value is 3.

    Returns
    -------
    A shape (dimension+1,) or (dimension+1,num_points) np.ndarray of floats between 0.0 and 1.0
     which sum to 1.0.

    """
    if num_points == 1:
        return np.diff(np.sort(rg.random(dimension)), prepend=0.0, append=1.0)
    else:
        return np.diff(
            np.sort(rg.random((dimension, num_points)), axis=0),
            prepend=0.0,
            append=1.0,
            axis=0,
        )


def tetrahedral_gradient(*, field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of a (linear) function defined at the points of a tetrahedron

    Parameters
    ----------
    points : a shape=(4,3) np.ndarray of points of a tetrahedron
    field : a shape=(4,) np.ndarray of point values of a function at the points of the tetrahedron

    Returns
    -------
    the gradient of the function as an (3,) np.ndarray
    """
    base_point = points[0, :]
    basis_vectors = points[1:, :] - base_point
    dfield = np.linalg.solve(basis_vectors, field[1:] - field[0])
    return dfield
