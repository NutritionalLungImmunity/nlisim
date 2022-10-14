from dataclasses import dataclass
from functools import cmp_to_key
from typing import List, Optional

from attr import define, field
import numpy as np


@dataclass
class TreeNodeInterval:
    node: 'TreeNode'
    min: float = float('-inf')
    max: float = float('inf')


@define(kw_only=True, auto_attribs=False, repr=False)
class TreeNode:
    leaf: bool = field()
    left: Optional[TreeNodeInterval] = field(default=None)
    center: Optional[TreeNodeInterval] = field(default=None)
    right: Optional[TreeNodeInterval] = field(default=None)
    tetrahedra_indices: Optional[np.ndarray] = field(default=None)


def create_tree(
    points: np.ndarray,
    element_point_indices: np.ndarray,
    dimension: int = 0,
    sorted_tetrahedra: Optional[List[int]] = None,
):
    if sorted_tetrahedra is None:
        sorted_tetrahedra = list(range(element_point_indices.shape[0]))
    assert sorted_tetrahedra is not None
    # don't proceed to parse into sub-trees below a threshold
    if len(sorted_tetrahedra) < 5:
        return TreeNode(leaf=True, tetrahedra_indices=np.array(sorted_tetrahedra))

    def fine_cmp(tet_a: int, tet_b: int) -> int:
        points_a = points[element_point_indices[tet_a], dimension]
        points_b = points[element_point_indices[tet_b], dimension]
        if np.max(points_a) < np.min(points_b):
            return -1
        elif np.min(points_a) > np.max(points_b):
            return 1
        else:
            mean_a = np.mean(points_a)
            mean_b = np.mean(points_b)
            if mean_a < mean_b:
                return -1
            elif mean_a > mean_b:
                return 1
            else:
                return 0

    sorted_tetrahedra.sort(key=cmp_to_key(fine_cmp))
    center_low = len(sorted_tetrahedra) // 2
    center_high = center_low + 1
    center_idx = sorted_tetrahedra[center_low]

    def rough_cmp(tet_a: int, tet_b: int) -> int:
        points_a = points[element_point_indices[tet_a], dimension]
        points_b = points[element_point_indices[tet_b], dimension]
        if np.max(points_a) < np.min(points_b):
            return -1
        elif np.min(points_a) > np.max(points_b):
            return 1
        else:
            return 0

    while center_low > 0 and rough_cmp(center_idx, sorted_tetrahedra[center_low - 1]) == 0:
        center_low -= 1
    while (
        center_high < len(sorted_tetrahedra)
        and rough_cmp(center_idx, sorted_tetrahedra[center_high]) == 0
    ):
        center_high += 1

    if center_low == 0 and center_high == len(sorted_tetrahedra):
        return TreeNode(leaf=True, tetrahedra_indices=np.array(sorted_tetrahedra))

    left_min = np.min(
        points[element_point_indices[sorted_tetrahedra[:center_low]], dimension],
        initial=float('inf'),
    )
    left_max = np.max(
        points[element_point_indices[sorted_tetrahedra[:center_low]], dimension],
        initial=float('-inf'),
    )
    left_tree = create_tree(
        sorted_tetrahedra=sorted_tetrahedra[:center_low],
        points=points,
        element_point_indices=element_point_indices,
        dimension=(dimension + 1) % 3,
    )

    center_min = np.min(
        points[element_point_indices[sorted_tetrahedra[center_low:center_high]], dimension],
        initial=float('inf'),
    )
    center_max = np.max(
        points[element_point_indices[sorted_tetrahedra[center_low:center_high]], dimension],
        initial=float('-inf'),
    )
    center_tree = create_tree(
        sorted_tetrahedra=sorted_tetrahedra[center_low:center_high],
        points=points,
        element_point_indices=element_point_indices,
        dimension=(dimension + 1) % 3,
    )

    right_min = np.min(
        points[element_point_indices[sorted_tetrahedra[center_high:]], dimension],
        initial=float('inf'),
    )
    right_max = np.max(
        points[element_point_indices[sorted_tetrahedra[center_high:]], dimension],
        initial=float('-inf'),
    )
    right_tree = create_tree(
        sorted_tetrahedra=sorted_tetrahedra[center_high:],
        points=points,
        element_point_indices=element_point_indices,
        dimension=(dimension + 1) % 3,
    )
    return TreeNode(
        leaf=False,
        left=TreeNodeInterval(node=left_tree, min=left_min, max=left_max),
        center=TreeNodeInterval(node=center_tree, min=center_min, max=center_max),
        right=TreeNodeInterval(node=right_tree, min=right_min, max=right_max),
    )


@define(auto_attribs=True, repr=False)
class TetrahedronSearchTree:
    points: np.ndarray = field()
    element_point_indices: np.ndarray = field()
    sorted_tree: TreeNode = field(init=False)

    def __attrs_post_init__(self):
        self.sorted_tree = create_tree(
            points=self.points, element_point_indices=self.element_point_indices
        )
