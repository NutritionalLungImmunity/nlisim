from enum import IntEnum

import attr
import random
import numpy as np

from simulation.cell import CellList, CellData
from simulation.coordinates import Point
from simulation.state import State, RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import GeometryState, TissueTypes

class MacrophageCellData(CellData):
    BOOLEAN_NETWORK_LENGTH = 23

    class Status(IntEnum):
        RESTING = 0
        ACTIVE = 1
        INTERACTING = 2
        SECRETING = 3
        SYNERGIC = 4
        APOPTOTIC = 5
        NECROTIC = 6
        DEAD = 7

    MACROPHAGE_FIELDS = [
        ('boolean_network', 'b1', BOOLEAN_NETWORK_LENGTH),
        ('status', 'u1'),
        ('iron_pool', 'f8'),
        ('iteration', 'i4'),
    ]

    dtype = np.dtype(CellData.BASE_FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell(
        cls,
        point: Point = None,
        iron_pool: float = 0,
        status: Status = Status.RESTING,
        **kwargs,
    ) -> np.record:

        if point is None:
            point = Point()

        network = cls.initial_boolean_network()
        iteration = 0

        return np.rec.array(
            [
                (
                    point,
                    network,
                    status,
                    iron_pool,
                    iteration,
                )
            ],
            dtype=cls.dtype,
        )[0]

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        return np.asarray(
            [
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData

def cell_list_factory(self: 'MacrophageState'):
    return MacrophageCellList(grid=self.global_state.grid)

@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_num: int

class Macrophage(Module):
    name = 'macrophage'
    defaults = {
        'cells': '',
        'init_num': '0',
    }
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        tissue: GeometryState = state.geometry.lung_tissue

        # macrophage.recruit_rate = self.config.getfloat('rec_rate')
        macrophage.init_num = self.config.getint('init_num')

        if(macrophage.init_num > 0):
            indices = np.argwhere(tissue == TissueTypes.SURFACTANT.value)
            np.random.shuffle(indices)

            for i in range(0, macrophage.init_num):
                x = indices[i][2] + (random.uniform(-0.5, 0.5))
                y = indices[i][1] + (random.uniform(-0.5, 0.5))
                z = indices[i][0] + (random.uniform(-0.5, 0.5))
                point = Point(x=x, y=y, z=z)

                if(i == 0):
                    # create one cell
                    macrophage.cells = MacrophageCellList.create_from_seed(grid=grid, point=point, status=MacrophageCellData.Status.RESTING)
                else:
                    macrophage.cells.append(MacrophageCellData.create_cell(point=point, status=MacrophageCellData.Status.RESTING))

        return state

    def advance(self, state: State, previous_time: float):
        return state

