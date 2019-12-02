from enum import IntEnum

import attr
import numpy as np

from simulation.cell import CellList
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

    dtype = np.dtype(CellList.BASE_FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

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

@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib()
    init_num: int

class Macrophage(Module):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid


        # macrophage.recruit_rate = self.config.getfloat('rec_rate')
        macrophage.init_num = self.config.getint('init_num')


        if(macrophage.init_num > 0):
            # create one cell 
            macrophage.cells = MacrophageCellList.create_from_seed(grid=grid, point=point, status=MacrophageCellData.Status.RESTING)

        return state

    def advance(self, state: State, previous_time: float):

        return state

    def get_surfactant_layer(self):
        return 

