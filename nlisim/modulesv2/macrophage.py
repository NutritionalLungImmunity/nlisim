import math
import random

import attr
from attr import attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point
from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.phagocyte import PhagocyteCellData, PhagocyteModel, PhagocyteState, PhagocyteStatus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function


class MacrophageCellData(PhagocyteCellData):
    MACROPHAGE_FIELDS = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('fpn', bool),
        ('fpn_iteration', np.int64),
        ('tf', bool),  # TODO: descriptive name, transferrin?
        ('move_step', np.float64),
        ('max_move_step', np.float64),  # TODO: double check, might be int
        ('tnfa', bool),
        ('engaged', bool),
        ('iron_pool', np.float64),
        ('status_iteration', np.uint)
        ]

    # TODO: this implementation of inheritance is not so slick, redo with super?
    dtype = np.dtype(CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + MACROPHAGE_FIELDS,
                     align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'status':           kwargs.get('status',
                                           PhagocyteStatus.RESTING),
            'state':            kwargs.get('state',
                                           PhagocyteState.FREE),
            'fpn':              kwargs.get('fpn',
                                           True),
            'fpn_iteration':    kwargs.get('fpn_iteration',
                                           0),
            'tf':               kwargs.get('tf',
                                           False),
            'move_step':        kwargs.get('move_step',
                                           0.0),  # TODO: reasonable default?
            'max_move_step':    kwargs.get('max_move_step',
                                           1.0),  # TODO: reasonable default?
            'tnfa':             kwargs.get('tnfa',
                                           False),
            'engaged':          kwargs.get('engaged',
                                           False),
            'iron_pool':        kwargs.get('iron_pool',
                                           0.0),
            'status_iteration': kwargs.get('status_iteration',
                                           0)
            }

        # ensure that these come in the correct order
        return \
            PhagocyteCellData.create_cell_tuple(**kwargs) + \
            tuple([initializer[key] for key, *_ in MacrophageCellData.MACROPHAGE_FIELDS])


@attrs(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData


def cell_list_factory(self: 'MacrophageState') -> MacrophageCellList:
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    move_rate_rest: float
    move_rate_act: float
    iter_to_rest: int
    time_to_change_state: float
    iter_to_change_state: int
    ma_internal_iron: float
    kd_ma_iron: float
    ma_vol: float
    ma_half_life: float
    min_ma: int
    init_num_macrophages: int


class Macrophage(PhagocyteModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        geometry: GeometryState = state.geometry
        time_step_size: float = self.time_step

        macrophage.max_conidia = self.config.getint('max_conidia')
        macrophage.iter_to_rest = self.config.getint('iter_to_rest')
        macrophage.time_to_change_state = self.config.getint('time_to_change_state')
        macrophage.ma_internal_iron = self.config.getfloat('ma_internal_iron')
        macrophage.kd_ma_iron = self.config.getfloat('kd_ma_iron')
        macrophage.ma_vol = self.config.getfloat('ma_vol')
        macrophage.min_ma = self.config.getint('min_ma')

        macrophage.init_num_macrophages = self.config.getint('init_num_macrophages')

        # computed values
        macrophage.move_rate_act = self.config.getfloat('move_rate_act') / time_step_size / 40  # TODO: 40?
        macrophage.move_rate_rest = self.config.getfloat('move_rate_rest') / time_step_size / 40

        macrophage.iter_to_change_state = int(macrophage.time_to_change_state * (60 / time_step_size))  # 4

        macrophage.ma_half_life = - math.log(0.5) / (
                self.config.getfloat('ma_half_life') * (60 / time_step_size))

        # initialize cells, placing them randomly TODO: can we do anything more specific?
        z_range = geometry.lung_tissue.shape[0]
        y_range = geometry.lung_tissue.shape[1]
        x_range = geometry.lung_tissue.shape[2]
        for _ in range(macrophage.init_num_macrophages):
            z = random.randint(0, z_range - 1)
            y = random.randint(0, y_range - 1)
            x = random.randint(0, x_range - 1)
            macrophage.cells.append(MacrophageCellData.create_cell(point=Point(x=x, y=y, z=z)))

        return state

    def advance(self, state: State, previous_time: float):
        from nlisim.modulesv2.afumigatus import AfumigatusCellState, AfumigatusState

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
                    afumigatus.cells[fungal_cell_index]['state'] = AfumigatusCellState.RELEASING

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
                if rg.uniform() < activation_function(x=macrophage_cell['iron_pool'] - macrophage.ma_internal_iron,
                                                      kd=macrophage.kd_ma_iron,
                                                      h=self.time_step / 60,
                                                      volume=macrophage.ma_vol):
                    macrophage_cell['status'] = PhagocyteStatus.ANERGIC
                    macrophage_cell['status_iteration'] = 0

            macrophage_cell['engaged'] = False  # TODO: find out what 'engaged' means

            # TODO: this usage suggests 'half life' should be 'prob death', real half life is 1/prob
            if num_cells_in_phagosome == 0 and \
                    rg.uniform() < macrophage.ma_half_life and \
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
