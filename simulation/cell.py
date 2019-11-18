from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.coordinates import Point
from simulation.state import RectangularGrid

BOOLEAN_NETWORK_LENGTH = 23
GROWTH_SCALE_FACTOR = 0.02  # from original code


class CellArray(np.ndarray):
    dtype = np.dtype([
        ('point', Point.dtype),
        ('growth', Point.dtype),
        ('boolean_network', 'b1', BOOLEAN_NETWORK_LENGTH),
        ('state', 'u1'),
        ('status', 'u1'),
        ('growable', 'b1'),
        ('switched', 'b1'),
        ('branchable', 'b1'),
        ('iron_pool', 'f8'),
        ('iron', 'b1'),
        ('iteration', 'i4')
    ], align=True)

    class Status(IntEnum):
        RESTING_CONIDIA = 0
        SWELLING_CONIDIA = 1
        HYPHAE = 2
        DYING = 3
        DEAD = 4

    class State(IntEnum):
        FREE = 0
        INTERNALIZING = 1
        RELEASING = 2

    def __new__(cls, length: int, **kwargs):
        return np.asarray([
            cls.create(**kwargs) for _ in range(length)
        ], dtype=cls.dtype).view(cls)

    @classmethod
    def create(cls, point: Point = None, iron_pool: float = 0,
               status: Status = Status.RESTING_CONIDIA,
               state: State = State.FREE) -> np.record:

        if point is None:
            point = Point()
        growth = GROWTH_SCALE_FACTOR * Point.from_array(2 * np.random.rand(3) - 1)
        network = cls.initial_boolean_network()
        growable = True
        switched = False
        branchable = False
        iteration = 0
        iron = False

        return np.rec.array([
            (point, growth, network, state, status, growable,
             switched, branchable, iron_pool, iron, iteration)
        ], dtype=cls.dtype)[0]

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        return np.asarray([
            True, False, True, False, True, True, True, True, True, False, False, False,
            True, False, False, False, False, False, False, False, False, False, False
        ])

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

    def elongate(self, grid: RectangularGrid) -> 'CellArray':
        mask = (
            self['growable'] &
            (self['status'] == self.Status.HYPHAE) &
            self.point_mask(self['point'] + self['growth'], grid)
        )

        self['growable'][mask] = False
        self['branchable'][mask] = True

        children = CellArray(mask.sum())
        self['iron_pool'][mask] /= 2
        children['iron_pool'] = self['iron_pool'][mask]
        children['point'] = self['point'][mask] + self['growth'][mask]
        children['growth'] = self['growth'][mask]

        a = np.append(self, children).view(CellArray)
        return a

    def branch(self, branch_probability: float, grid: RectangularGrid) -> 'CellArray':
        indices = (
            self['branchable'] &
            (self['status'] == self.Status.HYPHAE) &
            (np.random.rand(*self.shape) < branch_probability)
        ).nonzero()[0]

        if len(indices) == 0:
            return self

        children = CellArray(len(indices))
        children['growth'] = np.apply_along_axis(
            self.random_branch_direction, 1, children['growth'])

        children['point'] = self['point'][indices] + children['growth'][indices]
        children['iron_pool'] = self['iron_pool'][indices] / 2

        # filter out children lying outside of the domain
        indices = indices[self.point_mask(children['point'], grid)]

        # set iron content for branched cells
        self['iron_pool'][indices] /= 2

        # set properties
        self['branchable'][indices] = False
        children['growable'] = True
        children['branchable'] = False

        return np.append(self, children).view(CellArray)
