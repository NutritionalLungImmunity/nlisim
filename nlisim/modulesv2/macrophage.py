import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.afumigatus import AfumigatusState
from nlisim.modulesv2.afumigatus import FungalState
from nlisim.modulesv2.phagocyte import PhagocyteModel, PhagocyteState, PhagocyteStatus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function

MAX_CONIDIA = 50  # note: this the max that we can set the max to. i.e. not an actual model parameter


class MacrophageCellData(CellData):
    MACROPHAGE_FIELDS = [
        ('phagosome', (np.int32, MAX_CONIDIA)),
        ('has_conidia', np.bool),
        ('status', np.uint8),
        ('state', np.uint8),
        ('fpn', np.bool),
        ('fpn_iteration', np.int64),
        ('tf', np.bool),  # TODO: descriptive name, transferrin?
        ('max_move_step', np.float),  # TODO: double check, might be int
        ('tnfa', np.bool),
        ('engaged', np.bool),
        ('iron_pool', np.float),
        ('status_iteration', np.uint)
        ]

    dtype = np.dtype(CellData.FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'phagosome'       : kwargs.get('phagosome',
                                           -1 * np.ones(MAX_CONIDIA, dtype=np.int)),
            'has_conidia'     : kwargs.get('has_conidia',
                                           False),
            'status'          : kwargs.get('status',
                                           PhagocyteStatus.RESTING),
            'state'           : kwargs.get('state',
                                           PhagocyteState.FREE),
            'fpn'             : kwargs.get('fpn',
                                           True),
            'fpn_iteration'   : kwargs.get('fpn_iteration',
                                           0),
            'tf'              : kwargs.get('tf',
                                           False),
            'max_move_step'   : kwargs.get('max_move_step',
                                           1.0),  # TODO: reasonable default?
            'tnfa'            : kwargs.get('tnfa',
                                           False),
            'engaged'         : kwargs.get('engaged',
                                           False),
            'iron_pool'       : kwargs.get('iron_pool',
                                           0.0),
            'status_iteration': kwargs.get('status_iteration',
                                           0)
            }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + \
               [initializer[key] for key, tyype in MacrophageCellData.MACROPHAGE_FIELDS]


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData


def cell_list_factory(self: 'MacrophageState') -> MacrophageCellList:
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    max_conidia: int
    move_rate_rest: float
    move_rate_act: float
    iter_to_rest: int
    iter_to_change_state: int
    ma_internal_iron: float
    kd_ma_iron: float
    ma_vol: float
    ma_half_life: float
    min_ma: int


class MacrophageModel(PhagocyteModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid

        macrophage.max_conidia = self.config.getint('max_conidia')
        macrophage.move_rate_act = self.config.getfloat('move_rate_act')
        macrophage.move_rate_rest = self.config.getfloat('move_rate_rest')
        macrophage.iter_to_rest = self.config.getint('iter_to_rest')
        macrophage.iter_to_change_state = self.config.getint('iter_to_change_state')
        macrophage.ma_internal_iron = self.config.getfloat('ma_internal_iron')
        macrophage.kd_ma_iron = self.config.getfloat('kd_ma_iron')
        macrophage.ma_vol = self.config.getfloat('ma_vol')
        macrophage.ma_half_life = self.config.getfloat('ma_half_life')
        macrophage.min_ma = self.config.getint('min_ma')

        return state

    def advance(self, state: State, previous_time: float):
        macrophage: MacrophageState = state.macrophage
        afumigatus: AfumigatusState = state.afumigatus

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell = macrophage.cells[macrophage_cell_index]

            num_cells_in_phagosome = np.sum(macrophage_cell['phagosome'] >= 0)

            if macrophage_cell['status'] == PhagocyteStatus.NECROTIC:
                # TODO: what about APOPTOTIC?
                for fungal_cell_index in macrophage_cell['phagosome']:
                    if fungal_cell_index == -1:
                        continue
                    afumigatus.cells[fungal_cell_index]['state'] = FungalState.RELEASING

            elif num_cells_in_phagosome > macrophage.max_conidia:
                # TODO: how do we get here?
                macrophage_cell['status'] = PhagocyteStatus.NECROTIC

            elif macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                if macrophage_cell['status_iteration'] >= macrophage.iter_to_rest:
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['tnfa'] = False
                    macrophage_cell['status'] = PhagocyteStatus.RESTING
                else:
                    macrophage_cell['status_iteration'] += 1

            elif macrophage_cell['status'] == PhagocyteStatus.INACTIVE:
                if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['status'] = PhagocyteStatus.RESTING
                else:
                    macrophage_cell['status_iteration'] += 1

            elif macrophage_cell['status'] == PhagocyteStatus.ACTIVATING:
                if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['status'] = PhagocyteStatus.ACTIVE
                else:
                    macrophage_cell['status_iteration'] += 1

            elif macrophage_cell['status'] == PhagocyteStatus.INACTIVATING:
                if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['status'] = PhagocyteStatus.INACTIVE
                else:
                    macrophage_cell['status_iteration'] += 1

            elif macrophage_cell['status'] == PhagocyteStatus.ANERGIC:
                if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['status'] = PhagocyteStatus.RESTING
                else:
                    macrophage_cell['status_iteration'] += 1

            if macrophage_cell['status'] not in {PhagocyteStatus.DEAD,
                                                 PhagocyteStatus.NECROTIC,
                                                 PhagocyteStatus.APOPTOTIC}:
                if rg() < activation_function(x=macrophage_cell['iron_pool'] - macrophage.ma_internal_iron,
                                              kd=macrophage.kd_ma_iron,
                                              h=state.simulation.time_step_size / 60,
                                              volume=macrophage.ma_vol):
                    macrophage_cell['status'] = PhagocyteStatus.ANERGIC
                    macrophage_cell['status_iteration'] = 0

            macrophage_cell['engaged'] = False  # TODO: find out what 'engaged' means

            # TODO: this usage suggests 'half life' should be 'prob death', real half life is 1/prob
            if num_cells_in_phagosome == 0 and \
                    rg() < macrophage.ma_half_life and \
                    len(macrophage.cells.alive()) > macrophage.min_ma:
                macrophage_cell['status'] = PhagocyteStatus.DEAD
                macrophage_cell['dead'] = True

            if not macrophage_cell['fpn']:
                if macrophage_cell['fpn_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['fpn_iteration'] = 0
                    macrophage_cell['fpn'] = True
                else:
                    macrophage_cell['fpn_iteration'] += 1

            macrophage_cell['move_step'] = 0
            # TODO: -1 below was 'None'. this looks like something which needs to be reworked
            macrophage_cell['max_move_step'] = -1

        return state
