import math

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.phagocyte import internalize_aspergillus, PhagocyteCellData, PhagocyteModel, PhagocyteState, \
    PhagocyteStatus
from nlisim.random import rg
from nlisim.state import State

MAX_CONIDIA = 100  # note: this the max that we can set the max to. i.e. not an actual model parameter


class NeutrophilCellData(PhagocyteCellData):
    NEUTROPHIL_FIELDS = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('iron_pool', np.float64),
        ('max_move_step', np.float64),  # TODO: double check, might be int
        ('tnfa', bool),
        ('engaged', bool),
        ('status_iteration', np.uint),
        ]

    dtype = np.dtype(CellData.FIELDS + NEUTROPHIL_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'status':           kwargs.get('status',
                                           PhagocyteStatus.RESTING),
            'state':            kwargs.get('state',
                                           PhagocyteState.FREE),
            'iron_pool':        kwargs.get('iron_pool',
                                           0.0),
            'max_move_step':    kwargs.get('max_move_step',
                                           1.0),  # TODO: reasonable default?
            'tnfa':             kwargs.get('tnfa',
                                           False),
            'engaged':          kwargs.get('engaged',
                                           False),
            'status_iteration': kwargs.get('status_iteration',
                                           0),
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
    time_to_change_state: float
    iter_to_change_state: int
    pr_n_hyphae: float
    pr_n_phag: float


class Neutrophil(PhagocyteModel):
    name = 'neutrophil'
    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume
        time_step_size: float = self.time_step

        neutrophil.time_to_change_state = self.config.getfloat('time_to_change_state')
        neutrophil.max_conidia = self.config.getint('max_conidia')

        # computed values
        # TODO: not a real half life
        neutrophil.half_life = - math.log(0.5) / (
                self.config.getfloat('half_life') * (60 / time_step_size))

        neutrophil.iter_to_change_state = int(neutrophil.time_to_change_state * 60 / time_step_size)

        rel_n_hyphae_int_unit_t = time_step_size / 60  # per hour # TODO: not like this
        neutrophil.pr_n_hyphae = 1 - math.exp(-rel_n_hyphae_int_unit_t / (
                voxel_volume * self.config.getfloat('pr_n_hyphae_param')))  # TODO: -exp1m

        rel_phag_afnt_unit_t = time_step_size / 60  # TODO: not like this
        neutrophil.pr_n_phag = 1 - math.exp(
            -rel_phag_afnt_unit_t / (voxel_volume * self.config.getfloat('pr_n_phag_param')))  # TODO: -exp1m

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        from nlisim.modulesv2.afumigatus import AfumigatusCellData, AfumigatusCellState, AfumigatusCellStatus, \
            AfumigatusState
        from nlisim.modulesv2.iron import IronState
        from nlisim.modulesv2.macrophage import MacrophageCellData, MacrophageState

        neutrophil: NeutrophilState = state.neutrophil
        macrophage: MacrophageState = state.macrophage
        afumigatus: AfumigatusState = state.afumigatus
        iron: IronState = state.iron
        grid: RectangularGrid = state.grid

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])

            if neutrophil_cell['status'] in {PhagocyteStatus.NECROTIC, PhagocyteStatus.APOPTOTIC}:
                for fungal_cell_index in neutrophil_cell['phagosome']:
                    if fungal_cell_index == -1:
                        continue
                    afumigatus.cells[fungal_cell_index]['state'] = AfumigatusCellState.RELEASING

            elif rg.uniform() < neutrophil.half_life:
                neutrophil_cell['status'] = PhagocyteStatus.APOPTOTIC

            elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['tnfa'] = False
                    neutrophil_cell['status'] = PhagocyteStatus.RESTING
                    neutrophil_cell['state'] = PhagocyteState.FREE  # TODO: was not part of macrophage
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
            neutrophil_cell['engaged'] = False  # TODO: find out what 'engaged' means

            # ---------- interactions

            # dead and dying cells release iron
            # TODO: can move this to a numpy operation if it ends up more performant
            if neutrophil_cell['status'] in {PhagocyteStatus.NECROTIC, PhagocyteStatus.APOPTOTIC, PhagocyteStatus.DEAD}:
                iron.grid[tuple(neutrophil_cell_voxel)] += neutrophil_cell['iron_pool']
                neutrophil_cell['iron_pool'] = 0

            # interact with fungus
            if not neutrophil_cell['engaged'] and neutrophil_cell['status'] not in {PhagocyteStatus.APOPTOTIC,
                                                                                    PhagocyteStatus.NECROTIC,
                                                                                    PhagocyteStatus.DEAD}:
                # get fungal cells in this voxel
                local_aspergillus = afumigatus.cells.get_cells_in_voxel(neutrophil_cell_voxel)
                for aspergillus_index in local_aspergillus:
                    aspergillus_cell: AfumigatusCellData = afumigatus.cells[aspergillus_index]
                    if aspergillus_cell['dead']: continue

                    if aspergillus_cell['status'] in {AfumigatusCellStatus.HYPHAE, AfumigatusCellStatus.GERM_TUBE}:
                        # possibly internalize the fungal cell
                        if rg.uniform() < neutrophil.pr_n_hyphae:
                            internalize_aspergillus(phagocyte_cell=neutrophil_cell,
                                                    aspergillus_cell=aspergillus_cell,
                                                    aspergillus_cell_index=aspergillus_index,
                                                    phagocyte=neutrophil)
                            aspergillus_cell['status'] = AfumigatusCellStatus.DYING
                        else:
                            neutrophil_cell['engaged'] = True

                    elif aspergillus_cell['status'] == AfumigatusCellStatus.SWELLING_CONIDIA:
                        if rg.uniform() < neutrophil.pr_n_phag:
                            internalize_aspergillus(phagocyte_cell=neutrophil_cell,
                                                    aspergillus_cell=aspergillus_cell,
                                                    aspergillus_cell_index=aspergillus_index,
                                                    phagocyte=neutrophil)
                        else:
                            pass

            # interact with macrophages:
            # if we are apoptotic, give our iron and phagosome to a nearby present macrophage (if empty)
            if neutrophil_cell['status'] == PhagocyteStatus.APOPTOTIC:
                local_macrophages = macrophage.cells.get_cells_in_voxel(neutrophil_cell_voxel)
                for macrophage_index in local_macrophages:
                    macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_index]
                    macrophage_num_cells_in_phagosome = np.sum(macrophage_cell['phagosome'] >= 0)
                    # TODO: Henrique, why only if empty?
                    if macrophage_num_cells_in_phagosome == 0:
                        macrophage_cell['phagosome'] = neutrophil_cell['phagosome']
                        macrophage_cell['iron_pool'] += neutrophil_cell['iron_pool']
                        neutrophil_cell['iron_pool'] = 0.0  # TODO: verify, Henrique's code looks odd
                        neutrophil_cell['status'] = PhagocyteStatus.DEAD
                        macrophage_cell['status'] = PhagocyteStatus.INACTIVE

        return state

# def get_max_move_steps(self):  ##REVIEW
#     if self.max_move_step is None:
#         if self.status == Macrophage.ACTIVE:
#             self.max_move_step = np.random.poisson(Constants.MA_MOVE_RATE_REST)
#         else:
#             self.max_move_step = np.random.poisson(Constants.MA_MOVE_RATE_REST)
#     return self.max_move_step
