import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.afumigatus import AfumigatusState
from nlisim.modulesv2.afumigatus import FungalState
from nlisim.modulesv2.erythrocyte import ErythrocyteState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.hemoglobin import HemoglobinState
from nlisim.modulesv2.hemolysin import HemolysinState
from nlisim.modulesv2.macrophage import MacrophageState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.phagocyte import PhagocyteModel, PhagocyteState, PhagocyteStatus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function

MAX_CONIDIA = 50  # note: this the max that we can set the max to. i.e. not an actual model parameter


class NeutrophilCellData(CellData):
    NEUTROPHIL_FIELDS = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('iron_pool', np.float),
        ('max_move_step', np.float),  # TODO: double check, might be int
        ('tnfa', np.bool),
        ('engaged', np.bool),
        ('status_iteration', np.uint),
        # XXX
        ('phagosome', (np.int32, MAX_CONIDIA)),
        ('has_conidia', np.bool),
        ('fpn', np.bool),
        ('fpn_iteration', np.int64),
        ('tf', np.bool),  # TODO: descriptive name, transferrin?


        ]

    dtype = np.dtype(CellData.FIELDS + NEUTROPHIL_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'status'          : kwargs.get('status',
                                           PhagocyteStatus.RESTING),
            'state'           : kwargs.get('state',
                                           PhagocyteState.FREE),
            'iron_pool'       : kwargs.get('iron_pool',
                                           0.0),
            'max_move_step'   : kwargs.get('max_move_step',
                                           1.0),  # TODO: reasonable default?
            'tnfa'            : kwargs.get('tnfa',
                                           False),
            'engaged'         : kwargs.get('engaged',
                                           False),
            'status_iteration': kwargs.get('status_iteration',
                                           0),
            # XXX
            'phagosome'       : kwargs.get('phagosome',
                                           -1 * np.ones(MAX_CONIDIA, dtype=np.int)),
            'has_conidia'     : kwargs.get('has_conidia',
                                           False),
            'fpn'             : kwargs.get('fpn',
                                           True),
            'fpn_iteration'   : kwargs.get('fpn_iteration',
                                           0),
            'tf'              : kwargs.get('tf',
                                           False),
            }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + \
               [initializer[key] for key, tyype in NeutrophilCellData.NEUTROPHIL_FIELDS]


@attrs(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData


def cell_list_factory(self: 'NeutrophilState') -> NeutrophilCellList:
    return NeutrophilCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class NeutrophilState(ModuleState):
    cells: NeutrophilCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    half_life: float
    iter_to_change_state:int


class NeutrophilModel(PhagocyteModel):
    name = 'neutrophil'
    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid

        neutrophil.half_life = self.config.getfloat('half_life') # TODO: not a real half life
        neutrophil.iter_to_change_state = self.config.getint('iter_to_change_state')

        return state

    def advance(self, state: State, previous_time: float):
        neutrophil: NeutrophilState = state.neutrophil
        erythrocyte: ErythrocyteState = state.erythrocyte
        molecules: MoleculesState = state.molecules
        hemoglobin: HemoglobinState = state.hemoglobin
        hemolysin: HemolysinState = state.hemolysin
        macrophage: MacrophageState = state.macrophage
        afumigatus: AfumigatusState = state.afumigatus
        geometry: GeometryState = state.geometry
        grid: RectangularGrid = state.grid

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = neutrophil.cells[neutrophil_cell_index]

            num_cells_in_phagosome = np.sum(neutrophil_cell['phagosome'] >= 0)

            if neutrophil_cell['status'] in {PhagocyteStatus.NECROTIC, PhagocyteStatus.APOPTOTIC}:
                for fungal_cell_index in neutrophil_cell['phagosome']:
                    if fungal_cell_index == -1:
                        continue
                    afumigatus.cells[fungal_cell_index]['state'] = FungalState.RELEASING

            elif rg() < neutrophil.half_life:
                neutrophil_cell['status'] = PhagocyteStatus.APOPTOTIC

            elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['tnfa'] = False
                    neutrophil_cell['status'] = PhagocyteStatus.RESTING
                    neutrophil_cell['state'] = PhagocyteState.FREE # TODO: was not part of macrophage
                else:
                    neutrophil_cell['status_iteration'] += 1

            elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVATING:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                else:
                    neutrophil_cell['status_iteration'] += 1

            neutrophil_cell['move_step'] = 0
            # TODO: -1 below was 'None'. this looks like something which needs to be reworked
            neutrophil_cell['max_move_step'] = -1
            neutrophil_cell['engaged'] = False # TODO: find out what 'engaged' means

            # interactions

            # interact with iron

        # elif itype is Iron:
        #     if self.status == Neutrophil.NECROTIC or self.status == Neutrophil.APOPTOTIC or self.status == Neutrophil.DEAD:
        #         interactable.inc(self.iron_pool)
        #         self.inc_iron_pool(-self.iron_pool)
        #     return False  # TODO: ACK Is this supposed to be false?
        # elif itype is Afumigatus:
        #     if self.engaged:
        #         return True
        #     if self.status != Neutrophil.APOPTOTIC and self.status != Neutrophil.NECROTIC and self.status != Neutrophil.DEAD:
        #         if interactable.status == Afumigatus.HYPHAE or interactable.status == Afumigatus.GERM_TUBE:
        #             if random() < Constants.PR_N_HYPHAE:
        #                 Phagocyte.int_aspergillus(self, interactable)
        #                 interactable.status = Afumigatus.DYING
        #             else:
        #                 self.engaged = True
        #         elif interactable.status == Afumigatus.SWELLING_CONIDIA:
        #             if random() < Constants.PR_N_PHAG:
        #                 Phagocyte.int_aspergillus(self, interactable)
        #             else:
        #                 pass
        #                 # interactable.status = Afumigatus.STERILE_CONIDIA
        #                 # Afumigatus.total_sterile_conidia = Afumigatus.total_sterile_conidia + 1
        #     return True
        # elif itype is Macrophage:
        #     if self.status == Neutrophil.APOPTOTIC and len(interactable.phagosome.agents) == 0:
        #         interactable.phagosome.agents = self.phagosome.agents
        #         interactable.inc_iron_pool(self.iron_pool)
        #         self.inc_iron_pool(self.iron_pool)
        #         self.die()
        #         interactable.status = Macrophage.INACTIVE
        #     return True
        # elif itype is ROS and interactable.state == Neutrophil.INTERACTING:
        #     if self.status == Neutrophil.ACTIVE:
        #         interactable.inc(0)
        #     return True
        # else:
        #     return interactable.interact(self)

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = neutrophil.cells[neutrophil_cell_index]

            num_cells_in_phagosome = np.sum(neutrophil_cell['phagosome'] >= 0)

            if neutrophil_cell['status'] == PhagocyteStatus.NECROTIC:
                for fungal_cell_index in neutrophil_cell['phagosome']:
                    if fungal_cell_index == -1:
                        continue
                    afumigatus.cells[fungal_cell_index]['state'] = FungalState.RELEASING

            elif num_cells_in_phagosome > neutrophil.max_conidia:
                # TODO: how do we get here?
                neutrophil_cell['status'] = PhagocyteStatus.NECROTIC

            elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_rest:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['tnfa'] = False
                    neutrophil_cell['status'] = PhagocyteStatus.RESTING
                else:
                    neutrophil_cell['status_iteration'] += 1

            elif neutrophil_cell['status'] == PhagocyteStatus.INACTIVE:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['status'] = PhagocyteStatus.RESTING
                else:
                    neutrophil_cell['status_iteration'] += 1

            elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVATING:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                else:
                    neutrophil_cell['status_iteration'] += 1

            elif neutrophil_cell['status'] == PhagocyteStatus.INACTIVATING:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['status'] = PhagocyteStatus.INACTIVE
                else:
                    neutrophil_cell['status_iteration'] += 1

            elif neutrophil_cell['status'] == PhagocyteStatus.ANERGIC:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['status'] = PhagocyteStatus.RESTING
                else:
                    neutrophil_cell['status_iteration'] += 1

            if neutrophil_cell['status'] not in {PhagocyteStatus.DEAD,
                                                 PhagocyteStatus.NECROTIC,
                                                 PhagocyteStatus.APOPTOTIC}:
                if rg() < activation_function(x=neutrophil_cell['iron_pool'] - neutrophil.ma_internal_iron,
                                              kd=neutrophil.kd_ma_iron,
                                              h=state.simulation.time_step_size / 60,
                                              volume=neutrophil.ma_vol):
                    neutrophil_cell['status'] = PhagocyteStatus.ANERGIC
                    neutrophil_cell['status_iteration'] = 0

            neutrophil_cell['engaged'] = False

            # TODO: this usage suggests 'half life' should be 'prob death', real half life is 1/prob
            if num_cells_in_phagosome == 0 and \
                    rg() < neutrophil.ma_half_life and \
                    len(neutrophil.cells.alive()) > neutrophil.min_ma:
                neutrophil_cell['status'] = PhagocyteStatus.DEAD

            if not neutrophil_cell['fpn']:
                if neutrophil_cell['fpn_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['fpn_iteration'] = 0
                    neutrophil_cell['fpn'] = True
                else:
                    neutrophil_cell['fpn_iteration'] += 1

            neutrophil_cell['move_step'] = 0
            # TODO: -1 below was 'None'. this looks like something which needs to be reworked
            neutrophil_cell['max_move_step'] = -1

        return state

# def get_max_move_steps(self):  ##REVIEW
#     if self.max_move_step is None:
#         if self.status == Macrophage.ACTIVE:
#             self.max_move_step = np.random.poisson(Constants.MA_MOVE_RATE_REST)
#         else:
#             self.max_move_step = np.random.poisson(Constants.MA_MOVE_RATE_REST)
#     return self.max_move_step
