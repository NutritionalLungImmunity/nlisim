from enum import IntEnum

import numpy as np

from simulation.cell import CellArray, CellTree
from simulation.coordinates import Point

BOOLEAN_NETWORK_LENGTH = 23


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


class AfumigatusCellArray(CellArray):
    AFUMIGATUS_FIELDS = [
        ('boolean_network', 'b1', BOOLEAN_NETWORK_LENGTH),
        ('state', 'u1'),
        ('status', 'u1'),
        ('iron_pool', 'f8'),
        ('iron', 'b1'),
        ('iteration', 'i4')
    ]

    dtype = np.dtype(CellArray.BASE_FIELDS + AFUMIGATUS_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell(cls, point: Point = None, iron_pool: float = 0,
                    status: Status = Status.RESTING_CONIDIA,
                    state: State = State.FREE, **kwargs) -> np.record:

        if point is None:
            point = Point()

        growth = cls.GROWTH_SCALE_FACTOR * Point.from_array(2 * np.random.rand(3) - 1)
        network = cls.initial_boolean_network()
        growable = True
        switched = False
        branchable = False
        iteration = 0
        iron = False

        return np.rec.array([
            (point, growth, growable, switched, branchable,
             network, state, status, iron_pool, iron, iteration)
        ], dtype=cls.dtype)[0]

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        return np.asarray([
            True, False, True, False, True, True, True, True, True, False, False, False,
            True, False, False, False, False, False, False, False, False, False, False
        ])


class AfumigatusCellTree(CellTree):
    CellArrayClass = AfumigatusCellArray

    def append(self, cell, parent=None):
        if parent is not None:
            iron_pool = self.cells[parent]['iron_pool'] / 2
            self.cells[parent]['iron_pool'] = cell['iron_pool'] = iron_pool
        return super().append(cell, parent)

    def is_growable(self):
        return (
            super().is_growable() &
            (self.cells['status'] == Status.HYPHAE)
        )

    def is_branchable(self, branch_probability):
        return (
            super().is_branchable(branch_probability) &
            (self.cells['status'] == Status.HYPHAE)
        )
