from enum import auto, IntEnum, unique

import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.phagocyte import PhagocyteModel, PhagocyteState, PhagocyteStatus
from nlisim.state import State

MAX_CONIDIA = 100


class MacrophageLifecycle(IntEnum):
    APOPTOTIC = 0
    NECROTIC = auto()
    DEAD = auto()


# TODO: naming
@unique
class MacrophageNetworkSpecies(IntEnum):
    Dectin1 = 0
    TNFa = auto()
    IL6 = auto()
    Ft = auto()
    DMT1 = auto()
    LIP = auto()
    TFR = auto()
    Fe2 = auto()
    IRP = auto()
    Hep = auto()
    Fpn = auto()
    TFBI = auto()
    Bglucan = auto()


class MacrophageCellData(CellData):
    MACROPHAGE_FIELDS = [
        # ('iteration', 'i4'),
        ('boolean_network', 'b1', len(MacrophageNetworkSpecies)),
        # ('phagosome', (np.int32, MAX_CONIDIA)),
        ('status', np.uint8),
        ('state', np.uint8),
        ('fpn', np.bool),
        ('fpn_iteration', np.int64),
        ('tf', np.bool),
        ('max_move_step', np.object),  # TODO: figure out what this is
        ('tnfa', np.bool),
        ('engaged', np.bool),
        ]

    dtype = np.dtype(CellData.FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'boolean_network': kwargs.get('boolean_network', cls.initial_boolean_network()),
            'status'         : kwargs.get('status', PhagocyteStatus.RESTING),
            'state'          : kwargs.get('state', PhagocyteState.FREE),
            'fpn'            : kwargs.get('fpn', True),
            'fpn_iteration'  : kwargs.get('fpn_iteration', 0),
            'tf'             : kwargs.get('tf', False),
            'max_move_step'  : kwargs.get('max_move_step', None),  # TODO: no none for numeric arrays
            'tnfa'           : kwargs.get('tnfa', False),
            'engaged'        : kwargs.get('engaged', False),
            }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + \
               [initializer[key] for key, tyype in MacrophageCellData.MACROPHAGE_FIELDS]

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        initMacrophageBooleanState = {MacrophageNetworkSpecies.Dectin1: 1,
                                      MacrophageNetworkSpecies.TNFa   : 0,
                                      MacrophageNetworkSpecies.IL6    : 1,
                                      MacrophageNetworkSpecies.Ft     : 0,
                                      MacrophageNetworkSpecies.DMT1   : 1,
                                      MacrophageNetworkSpecies.LIP    : 1,
                                      MacrophageNetworkSpecies.TFR    : 1,
                                      MacrophageNetworkSpecies.Fe2    : 1,
                                      MacrophageNetworkSpecies.IRP    : 1,
                                      MacrophageNetworkSpecies.Hep    : 0,
                                      MacrophageNetworkSpecies.Fpn    : 0,
                                      MacrophageNetworkSpecies.TFBI   : 0,
                                      MacrophageNetworkSpecies.Bglucan: 1}
        return np.asarray([initMacrophageBooleanState[species] for species in MacrophageNetworkSpecies], dtype=np.bool)


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData


def cell_list_factory(self: 'MacrophageState'):
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))

class MacrophageModel(PhagocyteModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid


        return state

    def advance(self, state: State, previous_time: float):
        macrophage: MacrophageState = state.macrophage

        return state


