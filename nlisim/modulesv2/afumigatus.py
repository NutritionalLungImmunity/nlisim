from enum import auto, IntEnum, unique

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Voxel
from nlisim.module import ModuleModel, ModuleState
from nlisim.modulesv2.macrophage import MacrophageCellData, MacrophageLifecycle, MacrophageState
from nlisim.state import State


# TODO: naming
@unique
class AFumigatusForm(IntEnum):
    RESTING_CONIDIA = 0
    SWELLING_CONIDIA = auto()
    GERM_TUBE = auto()
    HYPHAE = auto()
    DYING = auto()
    DEAD = auto()
    STERILE_CONIDIA = auto()


# TODO: naming
@unique
class NetworkSpecies(IntEnum):
    hapX = 0       # gene
    sreA = auto()  # gene
    HapX = auto()  # protein
    SreA = auto()  # protein
    RIA = auto()
    EstB = auto()
    MirB = auto()
    SidA = auto()
    TAFC = auto()
    ICP = auto()
    LIP = auto()
    CccA = auto()
    FC0fe = auto()
    FC1fe = auto()
    VAC = auto()
    ROS = auto()
    Yap1 = auto()
    SOD2_3 = auto()
    Cat1_2 = auto()
    ThP = auto()
    Fe = auto()
    O = auto()


# TODO: naming
@unique
class AFumigatusState(IntEnum):
    FREE = 0
    INTERNALIZING = auto()
    RELEASING = auto()


class AfumigatusCellData(CellData):
    AFUMIGATUS_FIELDS = [
        ('iron_pool', np.float64),
        ('state', np.uint8),
        ('status', np.uint8),
        ('is_root', np.bool),
        ('root', np.float64, 3),
        ('tip', np.float64, 3),
        ('vec', np.float64, 3),
        ('growable', np.bool),
        ('branchable', np.bool),
        ('activation_iteration', np.int64),
        ('growth_iteration', np.int64),
        ('boolean_network', 'b1', len(NetworkSpecies)),
        ('next_branch', np.int64),
        ('next_septa', np.int64),
        ('previous_septa', np.int64),
        ('bn_iteration', np.int64),
        ]

    FIELDS = CellData.FIELDS + AFUMIGATUS_FIELDS
    dtype = np.dtype(FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs) -> np.record:
        initializer = {
            'iron_pool'           : kwargs.get('iron_pool', 0),
            'state'               : kwargs.get('state', AFumigatusState.FREE),
            'status'              : kwargs.get('status', AFumigatusForm.RESTING_CONIDIA),
            'is_root'             : kwargs.get('is_root', True),
            'root'                : kwargs.get('root', np.ndarray([0.0, 0.0, 0.0], dtype=np.float64)),
            'tip'                 : kwargs.get('tip', np.ndarray([0.0, 0.0, 0.0], dtype=np.float64)),
            'vec'                 : kwargs.get('vec', np.ndarray([0.0, 0.0, 0.0], dtype=np.float64)),  # dx, dy, dz
            'growable'            : kwargs.get('growable', True),
            'branchable'          : kwargs.get('branchable', False),
            'activation_iteration': kwargs.get('activation_iteration', 0),
            'growth_iteration'    : kwargs.get('growth_iteration', 0),
            'boolean_network'     : kwargs.get('boolean_network', cls.initial_boolean_network()),
            'bn_iteration'        : kwargs.get('bn_iteration', 0),
            'next_branch'         : kwargs.get('next_branch', -1),
            'next_septa'          : kwargs.get('next_septa', -1),
            'previous_septa'      : kwargs.get('previous_septa', -1),
            }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + \
               [initializer[key] for key, tyype in AfumigatusCellData.AFUMIGATUS_FIELDS]

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        initAfumigatusBooleanSpecies = {NetworkSpecies.hapX  : True,
                                        NetworkSpecies.sreA  : False,
                                        NetworkSpecies.HapX  : True,
                                        NetworkSpecies.SreA  : False,
                                        NetworkSpecies.RIA   : True,
                                        NetworkSpecies.EstB  : True,
                                        NetworkSpecies.MirB  : True,
                                        NetworkSpecies.SidA  : True,
                                        NetworkSpecies.TAFC  : True,
                                        NetworkSpecies.ICP   : False,
                                        NetworkSpecies.LIP   : False,
                                        NetworkSpecies.CccA  : False,
                                        NetworkSpecies.FC0fe : True,
                                        NetworkSpecies.FC1fe : False,
                                        NetworkSpecies.VAC   : False,
                                        NetworkSpecies.ROS   : False,
                                        NetworkSpecies.Yap1  : False,
                                        NetworkSpecies.SOD2_3: False,
                                        NetworkSpecies.Cat1_2: False,
                                        NetworkSpecies.ThP   : False,
                                        NetworkSpecies.Fe    : False,
                                        NetworkSpecies.O     : False,
                                        # NetworkSpecies.TAFCBI:False TODO: I'm assuming ?
                                        # There was an extra in the source material
                                        }
        return np.asarray([initAfumigatusBooleanSpecies[species] for species in NetworkSpecies], dtype=np.bool)


@attrs(kw_only=True, frozen=True, repr=False)
class AfumigatusCellList(CellList):
    CellDataClass = AfumigatusCellData


def cell_list_factory(self: 'AfumigatusState'):
    return AfumigatusCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class AfumigatusState(ModuleState):
    cells: AfumigatusCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    pr_ma_hyphae: float
    pr_ma_phag: float
    pr_branch: float
    steps_to_bn_eval: int


class Afumigatus(ModuleModel):
    name = 'afumigatus'
    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus

        afumigatus.pr_ma_hyphae = self.config.getfloat('pr_ma_hyphae')
        afumigatus.pr_ma_phag = self.config.getfloat('pr_ma_phag')

        afumigatus.pr_branch = self.config.getfloat('pr_branch')
        afumigatus.steps_to_bn_eval = self.config.getint('steps_to_bn_eval')

        return state

    def advance(self, state: State, previous_time: float) -> State:
        afumigatus: AfumigatusState = state.afumigatus
        macrophage: MacrophageState = state.macrophage
        grid = state.grid
        tissue = state.geometry.lung_tissue

        # itype = type(interactable)
        # if itype is Afumigatus:
        #     return False

        # iterate over live cells
        for afumigatus_index in afumigatus.cells.alive():
            # get cell and voxel-position
            cell: AfumigatusCellData = afumigatus.cells[afumigatus_index]
            vox: Voxel = grid.get_voxel(cell['point'])

            if cell['state'] == AFumigatusForm.RESTING_CONIDIA:
                continue

            nearby_macrophage_indices = macrophage.cells.get_cells_in_voxel(vox)
            for macrophage_index in nearby_macrophage_indices:
                individual_macrophage: MacrophageCellData = macrophage.cells[macrophage_index]
                if individual_macrophage.lifecycle not in {MacrophageLifecycle.APOPTOTIC,
                                                           MacrophageLifecycle.NECROTIC,
                                                           MacrophageLifecycle.DEAD}:
                    pr_interact = afumigatus.pr_ma_hyphae if cell.form == AFumigatusForm.HYPHAE \
                        else afumigatus.pr_ma_phag

            # elif itype is Macrophage:
            #     if interactable.engaged:
            #         return True
            #     if interactable.status != Macrophage.APOPTOTIC and interactable.status != Macrophage.NECROTIC and interactable.status != Macrophage.DEAD:
            #         if self.status != Afumigatus.RESTING_CONIDIA:
            #             pr_interact = Constants.PR_MA_HYPHAE if self.status == Afumigatus.HYPHAE else Constants.PR_MA_PHAG
            #             if random() < pr_interact:
            #                 Phagocyte.int_aspergillus(interactable, self, self.status != Afumigatus.HYPHAE)
            #                 if self.status == Afumigatus.HYPHAE and interactable.status == Macrophage.ACTIVE:
            #                     self.status = Afumigatus.DYING
            #                     if self.next_septa is not None:
            #                         self.next_septa.is_root = True
            #                     if self.next_branch is not None:
            #                         self.next_branch.is_root = True
            #                 else:
            #                     if self.status == Afumigatus.HYPHAE and interactable.status == Macrophage.ACTIVE:
            #                         interactable.engaged = True

        # TODO: move to dying process
        # elif itype is Iron:
        #     if self.status == Afumigatus.DYING or self.status == Afumigatus.DEAD:
        #         interactable.inc(self.iron_pool)
        #         self.inc_iron_pool(-self.iron_pool)
        #     return True

        return state
