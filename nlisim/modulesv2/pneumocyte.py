import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.modulesv2.geometry import GeometryState, TissueType
from nlisim.modulesv2.phagocyte import PhagocyteCellData, PhagocyteModel, PhagocyteState, \
    PhagocyteStatus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function


class PneumocyteCellData(PhagocyteCellData):
    PNEUMOCYTE_FIELDS = [
        ('status', np.uint8),
        ('iteration', np.uint),
        ('tnfa', bool),
        ]

    dtype = np.dtype(CellData.FIELDS + PNEUMOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'status':    kwargs.get('status',
                                    PhagocyteStatus.RESTING),
            'iteration': kwargs.get('iteration',
                                    0),
            'tnfa':      kwargs.get('tnfa',
                                    False),
            }

        # ensure that these come in the correct order
        return \
            CellData.create_cell_tuple(**kwargs) + \
            [initializer[key] for key, _ in PneumocyteCellData.PNEUMOCYTE_FIELDS]


@attrs(kw_only=True, frozen=True, repr=False)
class PneumocyteCellList(CellList):
    CellDataClass = PneumocyteCellData


def cell_list_factory(self: 'PneumocyteState') -> PneumocyteCellList:
    return PneumocyteCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class PneumocyteState(PhagocyteState):
    cells: PneumocyteCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    iter_to_rest: int
    time_to_change_state: float
    iter_to_change_state: int
    p_il6_qtty: float
    p_il8_qtty: float
    p_tnf_qtty: float


class PneumocyteModel(PhagocyteModel):
    name = 'pneumocyte'
    StateClass = PneumocyteState

    def initialize(self, state: State):
        pneumocyte: PneumocyteState = state.pneumocyte
        geometry: GeometryState = state.geometry
        time_step_size: float = self.time_step

        pneumocyte.max_conidia = self.config.getint('max_conidia')
        pneumocyte.time_to_rest = self.config.getint('time_to_rest')
        pneumocyte.iter_to_change_state = self.config.getint('iter_to_change_state')

        pneumocyte.p_il6_qtty = self.config.getfloat('p_il6_qtty')
        pneumocyte.p_il8_qtty = self.config.getfloat('p_il8_qtty')
        pneumocyte.p_tnf_qtty = self.config.getfloat('p_tnf_qtty')

        # computed values
        pneumocyte.iter_to_rest = int(pneumocyte.time_to_rest * (60 / time_step_size))

        # initialize cells, placing one per epithelial voxel
        # TODO: Any changes due to ongoing conversation with Henrique
        for z, y, x in zip(*np.where(geometry.lung_tissue == TissueType.EPITHELIUM)):
            pneumocyte.cells.append(PneumocyteCellData.create_cell(point=Point(x=x, y=y, z=z)))

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        from nlisim.modulesv2.il6 import IL6State
        from nlisim.modulesv2.il8 import IL8State
        from nlisim.modulesv2.tnfa import TNFaState
        from nlisim.modulesv2.afumigatus import AfumigatusCellData, AfumigatusCellStatus, AfumigatusState

        pneumocyte: PneumocyteState = state.pneumocyte
        afumigatus: AfumigatusState = state.afumigatus
        il6: IL6State = state.il6
        il8: IL8State = state.il8
        tnfa: TNFaState = state.tnfa
        geometry: GeometryState = state.geometry
        grid: RectangularGrid = state.grid

        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell = pneumocyte.cells[pneumocyte_cell_index]
            pneumocyte_cell_voxel: Voxel = grid.get_voxel(pneumocyte_cell['point'])

            # self update
            if pneumocyte_cell['status'] == PhagocyteStatus.ACTIVE:
                if pneumocyte_cell['iteration'] >= pneumocyte.iter_to_rest:
                    pneumocyte_cell['iteration'] = 0
                    pneumocyte_cell['status'] = PhagocyteStatus.RESTING
                    pneumocyte_cell['tnfa'] = False
                else:
                    pneumocyte_cell['iteration'] += 1

            elif pneumocyte_cell['status'] == PhagocyteStatus.ACTIVATING:
                if pneumocyte_cell['iteration'] >= pneumocyte.iter_to_change_state:
                    pneumocyte_cell['iteration'] = 0
                    pneumocyte_cell['status'] = PhagocyteStatus.ACTIVE
                else:
                    pneumocyte_cell['iteration'] += 1

            # ----------- interactions

            # interact with fungus
            if pneumocyte_cell['status'] not in {PhagocyteStatus.APOPTOTIC,
                                                 PhagocyteStatus.NECROTIC,
                                                 PhagocyteStatus.DEAD}:
                local_aspergillus = afumigatus.cells.get_cells_in_voxel(pneumocyte_cell_voxel)
                for aspergillus_index in local_aspergillus:
                    aspergillus_cell: AfumigatusCellData = afumigatus.cells[aspergillus_index]

                    # skip resting conidia
                    if aspergillus_cell['status'] == AfumigatusCellStatus.RESTING_CONIDIA:
                        continue

                    if pneumocyte_cell['status'] != PhagocyteStatus.ACTIVE:
                        if rg() < pneumocyte.pr_p_int:
                            pneumocyte_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        # TODO: I don't get this, looks like it zeros out the iteration when activating
                        pneumocyte_cell['iteration'] = 0

            # secrete IL6
            if pneumocyte_cell['status'] == PhagocyteStatus.ACTIVE:
                il6.grid[pneumocyte_cell_voxel] += pneumocyte.p_il6_qtty

            # secrete IL8
            if pneumocyte_cell['tnfa']:  # TODO: and active?
                il8.grid[pneumocyte_cell_voxel] += pneumocyte.p_il8_qtty

            # interact with TNFa
            if pneumocyte_cell['status'] == PhagocyteStatus.ACTIVE and \
                    rg() < activation_function(x=tnfa.grid,
                                               kd=tnfa.k_d,
                                               h=self.time_step / 60,
                                               volume=geometry.voxel_volume):
                pneumocyte_cell['iteration'] = 0
                pneumocyte_cell['tnfa'] = True

            # secrete TNFa
            if pneumocyte_cell['status'] == PhagocyteStatus.ACTIVE:
                tnfa.grid[pneumocyte_cell_voxel] += pneumocyte.p_tnf_qtty

        return state
