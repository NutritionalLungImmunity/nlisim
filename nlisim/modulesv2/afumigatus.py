from enum import auto, IntEnum, unique
from queue import SimpleQueue

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Voxel
from nlisim.module import ModuleModel, ModuleState
from nlisim.modulesv2.iron import IronState
from nlisim.modulesv2.macrophage import MacrophageCellData, MacrophageLifecycle, MacrophageState
from nlisim.random import rg
from nlisim.state import State


# TODO: naming
@unique
class FungalForm(IntEnum):
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
    hapX = 0  # gene
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
            'status'              : kwargs.get('status', FungalForm.RESTING_CONIDIA),
            'is_root'             : kwargs.get('is_root', True),
            'root'                : kwargs.get('root',
                                               np.ndarray([0.0, 0.0, 0.0], dtype=np.float64)),
            'tip'                 : kwargs.get('tip',
                                               np.ndarray([0.0, 0.0, 0.0], dtype=np.float64)),
            'vec'                 : kwargs.get('vec',  # dx, dy, dz
                                               np.ndarray([0.0, 0.0, 0.0], dtype=np.float64)),
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
        return np.asarray([initAfumigatusBooleanSpecies[species]
                           for species in NetworkSpecies], dtype=np.bool)


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
    hyphae_volume: float
    kd_lip: float
    iter_to_swelling: float
    iter_to_germinate: float
    pr_aspergillus_change: float


class Afumigatus(ModuleModel):
    name = 'afumigatus'
    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus

        afumigatus.pr_ma_hyphae = self.config.getfloat('pr_ma_hyphae')
        afumigatus.pr_ma_phag = self.config.getfloat('pr_ma_phag')

        afumigatus.pr_branch = self.config.getfloat('pr_branch')
        afumigatus.steps_to_bn_eval = self.config.getint('steps_to_bn_eval')
        afumigatus.hyphae_volume = self.config.getint('hyphae_volume')
        afumigatus.kd_lip = self.config.getint('kd_lip')

        afumigatus.iter_to_swelling = self.config.getint('iter_to_swelling')
        afumigatus.pr_aspergillus_change = self.config.getfloat('pr_aspergillus_change')
        afumigatus.iter_to_germinate = self.config.getint('iter_to_germinate')

        return state

    def advance(self, state: State, previous_time: float) -> State:
        afumigatus: AfumigatusState = state.afumigatus
        macrophage: MacrophageState = state.macrophage
        grid = state.grid
        tissue = state.geometry.lung_tissue

        # update live cells
        for cell_index in afumigatus.cells.alive():
            cell: AfumigatusCellData = afumigatus.cells[cell_index]

            cell['activation_iteration'] += 1

            # resting conidia become swelling conidia after a number of iterations
            # (with some probability)
            if (cell['status'] == FungalForm.RESTING_CONIDIA and
                    cell['activation_iteration'] >= afumigatus.iter_to_swelling and
                    rg.random() < afumigatus.pr_aspergillus_change):
                cell['status'] = FungalForm.SWELLING_CONIDIA
                cell['activation_iteration'] = 0
            elif (cell['status'] == FungalForm.SWELLING_CONIDIA and
                  cell['activation_iteration'] >= afumigatus.iter_to_germinate):
                cell['status'] = FungalForm.GERM_TUBE
                cell['activation_iteration'] = 0
            elif cell['status'] == FungalForm.DYING:
                # TODO: Henrique said something about the DYING state not being necessary. First glance in code
                #  suggests that this update only removes the cells from live counts
                cell['status'] = FungalForm.DEAD

            # TODO: this looks redundant/unnecessary
            if cell['next_septa'] == -1:
                cell['growable'] = True

            # TODO: verify this, 1 turn on internalizing then free?
            if cell['state'] in {AFumigatusState.INTERNALIZING, AFumigatusState.RELEASING}:
                cell['state'] = AFumigatusState.FREE

            # self.diffuse_iron()
            # if self.next_branch is None:
            #     self.growable = True

        # itype = type(interactable)
        # if itype is Afumigatus:
        #     return False

        # iterate over live cells
        for afumigatus_index in afumigatus.cells.alive():
            # get cell and voxel-position
            cell: AfumigatusCellData = afumigatus.cells[afumigatus_index]
            vox: Voxel = grid.get_voxel(cell['point'])

            if cell['state'] == FungalForm.RESTING_CONIDIA:
                continue

            nearby_macrophage_indices = macrophage.cells.get_cells_in_voxel(vox)
            for macrophage_index in nearby_macrophage_indices:
                individual_macrophage: MacrophageCellData = macrophage.cells[macrophage_index]
                if individual_macrophage.lifecycle not in {MacrophageLifecycle.APOPTOTIC,
                                                           MacrophageLifecycle.NECROTIC,
                                                           MacrophageLifecycle.DEAD}:
                    pr_interact = afumigatus.pr_ma_hyphae if cell.form == FungalForm.HYPHAE \
                        else afumigatus.pr_ma_phag
                    if rg.random() < pr_interact:
                        Phagocyte.int_aspergillus(interactable, self, self.status != Afumigatus.HYPHAE)
                        if self.status == Afumigatus.HYPHAE and interactable.status == Macrophage.ACTIVE:
                            self.status = Afumigatus.DYING
                            if self.next_septa is not None:
                                self.next_septa.is_root = True
                            if self.next_branch is not None:
                                self.next_branch.is_root = True
                        else:
                            if self.status == Afumigatus.HYPHAE and interactable.status == Macrophage.ACTIVE:
                                interactable.engaged = True

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

    def process_boolean_network(self,
                                state: State,
                                boolean_network: np.ndarray,
                                bn_iteration: np.ndarray,
                                steps_to_eval: int):
        bn_iteration += 1
        bn_iteration %= steps_to_eval

        boolean_network_active: np.ndarray = boolean_network[bn_iteration == 0, :]
        temp: np.ndarray = np.zeros(shape=boolean_network_active.shape, dtype=np.bool)

        # TODO: verify array shape
        temp[:, NetworkSpecies.hapX] = ~boolean_network_active[:, NetworkSpecies.SreA]
        temp[:, NetworkSpecies.sreA] = ~boolean_network_active[:, NetworkSpecies.HapX]
        temp[:, NetworkSpecies.HapX] = boolean_network_active[:, NetworkSpecies.hapX] & \
                                       ~boolean_network_active[:, NetworkSpecies.LIP]
        temp[:, NetworkSpecies.SreA] = boolean_network_active[:, NetworkSpecies.sreA] & \
                                       boolean_network_active[:, NetworkSpecies.LIP]
        temp[:, NetworkSpecies.RIA] = ~boolean_network_active[:, NetworkSpecies.SreA]
        temp[:, NetworkSpecies.EstB] = ~boolean_network_active[:, NetworkSpecies.SreA]
        temp[:, NetworkSpecies.MirB] = boolean_network_active[:, NetworkSpecies.HapX] & \
                                       ~boolean_network_active[:, NetworkSpecies.SreA]
        temp[:, NetworkSpecies.SidA] = boolean_network_active[:, NetworkSpecies.HapX] & \
                                       ~boolean_network_active[:, NetworkSpecies.SreA]
        temp[:, NetworkSpecies.TAFC] = boolean_network_active[:, NetworkSpecies.SidA]
        temp[:, NetworkSpecies.ICP] = ~boolean_network_active[:, NetworkSpecies.HapX] & \
                                      (boolean_network_active[:, NetworkSpecies.VAC] |
                                       boolean_network_active[:, NetworkSpecies.FC1fe])
        temp[:, NetworkSpecies.LIP] = (boolean_network_active[:, NetworkSpecies.Fe] &
                                       boolean_network_active[:, NetworkSpecies.RIA]) | \
                                      self._lip_activation(state=state,
                                                           shape=temp.shape)
        temp[:, NetworkSpecies.CccA] = ~boolean_network_active[:, NetworkSpecies.HapX]
        temp[:, NetworkSpecies.FC0fe] = boolean_network_active[:, NetworkSpecies.SidA]
        temp[:, NetworkSpecies.FC1fe] = boolean_network_active[:, NetworkSpecies.LIP] & \
                                        boolean_network_active[:, NetworkSpecies.FC0fe]
        temp[:, NetworkSpecies.VAC] = boolean_network_active[:, NetworkSpecies.LIP] & \
                                      boolean_network_active[:, NetworkSpecies.CccA]
        temp[:, NetworkSpecies.ROS] = (boolean_network_active[:, NetworkSpecies.O] &
                                       ~(boolean_network_active[:, NetworkSpecies.SOD2_3] &
                                         boolean_network_active[:, NetworkSpecies.ThP] &
                                         boolean_network_active[:, NetworkSpecies.Cat1_2])) | \
                                      (boolean_network_active[:, NetworkSpecies.ROS] &
                                       ~(boolean_network_active[:, NetworkSpecies.SOD2_3] &
                                         (boolean_network_active[:, NetworkSpecies.ThP] |
                                          boolean_network_active[:, NetworkSpecies.Cat1_2])))
        temp[:, NetworkSpecies.Yap1] = boolean_network_active[:, NetworkSpecies.ROS]
        temp[:, NetworkSpecies.SOD2_3] = boolean_network_active[:, NetworkSpecies.Yap1]
        temp[:, NetworkSpecies.Cat1_2] = boolean_network_active[:, NetworkSpecies.Yap1] & \
                                         ~boolean_network_active[:, NetworkSpecies.HapX]
        temp[:, NetworkSpecies.ThP] = boolean_network_active[:, NetworkSpecies.Yap1]
        temp[:, NetworkSpecies.Fe] = 0  # might change according to iron environment?
        temp[:, NetworkSpecies.O] = 0

        # noinspection PyUnusedLocal
        boolean_network_active = temp

    def _lip_activation(self, state: State, shape) -> np.ndarray:
        afumigatus: AfumigatusState = state.afumigatus
        iron: IronState = state.iron

        molar_concentration = iron.grid / afumigatus.hyphae_volume
        activation = 1 - np.exp(-molar_concentration / afumigatus.kd_lip)
        return np.random.rand(*shape) < activation

    def diffuse_iron(self, root_cell_index: int, afumigatus: AfumigatusState) -> None:
        """
        Evenly distributes iron amongst fungal cells in a tree

        Parameters
        ----------
        root_cell_index : int
            index of tree root, function is a noop on non-root cells
        afumigatus : AfumigatusState
            state class for fungus
        Returns
        -------

        """
        if not afumigatus.cells[root_cell_index]['is_root']:
            return

        tree_cells = set()
        total_iron: float = 0.0

        # walk along the tree, collecting iron
        q = SimpleQueue()
        q.put(root_cell_index)
        while not q.empty():
            next_cell_index = q.get()
            tree_cells.add(next_cell_index)

            next_cell = afumigatus.cells[next_cell_index]
            total_iron += next_cell['iron']

            if next_cell['next_branch'] >= 0:
                q.put(next_cell['next_branch'])
            if next_cell['next_septa'] >= 0:
                q.put(next_cell['next_septa'])

        # distribute the iron evenly
        iron_per_cell: float = total_iron / len(tree_cells)
        for tree_cell_index in tree_cells:
            afumigatus.cells[tree_cell_index]['iron'] = iron_per_cell
