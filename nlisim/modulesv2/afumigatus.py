from enum import auto, IntEnum, unique
from queue import SimpleQueue
import random

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.module import ModuleModel, ModuleState
from nlisim.modulesv2.geometry import GeometryState, TissueType
from nlisim.modulesv2.iron import IronState
from nlisim.modulesv2.macrophage import MacrophageCellData, MacrophageState, PhagocyteStatus
from nlisim.modulesv2.phagocyte import internalize_aspergillus
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
    Oxygen = auto()


# TODO: naming
@unique
class FungalState(IntEnum):
    FREE = 0
    INTERNALIZING = auto()
    RELEASING = auto()


class AfumigatusCellData(CellData):
    AFUMIGATUS_FIELDS = [
        ('iron_pool', np.float64),
        ('state', np.uint8),
        ('status', np.uint8),
        ('is_root', bool),
        ('root', np.float64, 3),
        ('tip', np.float64, 3),
        ('vec', np.float64, 3),
        ('growable', bool),
        ('branchable', bool),
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
            'iron_pool':            kwargs.get('iron_pool',
                                               0),
            'state':                kwargs.get('state',
                                               FungalState.FREE),
            'status':               kwargs.get('status',
                                               FungalForm.RESTING_CONIDIA),
            'is_root':              kwargs.get('is_root',
                                               True),
            'root':                 kwargs.get('root',
                                               np.zeros(3, dtype=np.float64)),
            'tip':                  kwargs.get('tip',
                                               np.zeros(3, dtype=np.float64)),
            'vec':                  kwargs.get('vec',  # dx, dy, dz
                                               np.zeros(3, dtype=np.float64)),
            'growable':             kwargs.get('growable',
                                               True),
            'branchable':           kwargs.get('branchable',
                                               False),
            'activation_iteration': kwargs.get('activation_iteration',
                                               0),
            'growth_iteration':     kwargs.get('growth_iteration',
                                               0),
            'boolean_network':      kwargs.get('boolean_network',
                                               cls.initial_boolean_network()),
            'bn_iteration':         kwargs.get('bn_iteration',
                                               0),
            'next_branch':          kwargs.get('next_branch',
                                               -1),
            'next_septa':           kwargs.get('next_septa',
                                               -1),
            'previous_septa':       kwargs.get('previous_septa',
                                               -1),
            }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + [initializer[key] for key, _ in
                                                       AfumigatusCellData.AFUMIGATUS_FIELDS]

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        init_afumigatus_boolean_species = {NetworkSpecies.hapX:   True,
                                           NetworkSpecies.sreA:   False,
                                           NetworkSpecies.HapX:   True,
                                           NetworkSpecies.SreA:   False,
                                           NetworkSpecies.RIA:    True,
                                           NetworkSpecies.EstB:   True,
                                           NetworkSpecies.MirB:   True,
                                           NetworkSpecies.SidA:   True,
                                           NetworkSpecies.TAFC:   True,
                                           NetworkSpecies.ICP:    False,
                                           NetworkSpecies.LIP:    False,
                                           NetworkSpecies.CccA:   False,
                                           NetworkSpecies.FC0fe:  True,
                                           NetworkSpecies.FC1fe:  False,
                                           NetworkSpecies.VAC:    False,
                                           NetworkSpecies.ROS:    False,
                                           NetworkSpecies.Yap1:   False,
                                           NetworkSpecies.SOD2_3: False,
                                           NetworkSpecies.Cat1_2: False,
                                           NetworkSpecies.ThP:    False,
                                           NetworkSpecies.Fe:     False,
                                           NetworkSpecies.Oxygen: False,
                                           }
        return np.asarray([init_afumigatus_boolean_species[species]
                           for species in NetworkSpecies], dtype=bool)


@attrs(kw_only=True, frozen=True, repr=False)
class AfumigatusCellList(CellList):
    CellDataClass = AfumigatusCellData


def cell_list_factory(self: 'AfumigatusState') -> AfumigatusCellList:
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
    init_iron: float
    conidia_vol: float


class Afumigatus(ModuleModel):
    name = 'afumigatus'
    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus
        geometry: GeometryState = state.geometry

        afumigatus.pr_ma_hyphae = self.config.getfloat('pr_ma_hyphae')
        afumigatus.pr_ma_phag = self.config.getfloat('pr_ma_phag')

        afumigatus.pr_branch = self.config.getfloat('pr_branch')
        afumigatus.steps_to_bn_eval = self.config.getint('steps_to_bn_eval')

        afumigatus.conidia_vol = self.config.getfloat('conidia_vol')
        afumigatus.hyphae_volume = self.config.getint('hyphae_volume')
        afumigatus.kd_lip = self.config.getint('kd_lip')

        afumigatus.iter_to_swelling = self.config.getint('iter_to_swelling')
        afumigatus.pr_aspergillus_change = self.config.getfloat('pr_aspergillus_change')
        afumigatus.iter_to_germinate = self.config.getint('iter_to_germinate')

        # computed values
        afumigatus.init_iron = afumigatus.kd_lip * afumigatus.conidia_vol

        # place cells for initial infection
        # TODO: 'smart' placement should be checked
        # current initial positions: any air voxel which is in a Moore neighborhood of an epithelial voxel
        # https://en.wikipedia.org/wiki/Moore_neighborhood
        epithelium_mask = geometry.lung_tissue == TissueType.EPITHELIUM
        epithelium_mask |= np.roll(epithelium_mask, 1, axis=0) | np.roll(epithelium_mask, -1, axis=0)
        epithelium_mask |= np.roll(epithelium_mask, 1, axis=1) | np.roll(epithelium_mask, -1, axis=1)
        epithelium_mask |= np.roll(epithelium_mask, 1, axis=2) | np.roll(epithelium_mask, -1, axis=2)
        locations = list(zip(*np.where(np.logical_and(epithelium_mask, geometry.lung_tissue == TissueType.AIR))))
        for z, y, x in random.sample(locations, self.config.getint('init_infection_num')):
            afumigatus.cells.append(AfumigatusCellData.create_cell(point=Point(x=x, y=y, z=z),
                                                                   iron_pool=afumigatus.init_iron))

        return state

    def advance(self, state: State, previous_time: float) -> State:
        afumigatus: AfumigatusState = state.afumigatus
        macrophage: MacrophageState = state.macrophage
        iron: IronState = state.iron
        grid = state.grid

        # update live cells
        for afumigatus_index in afumigatus.cells.alive():
            # get cell and voxel position
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_index]
            voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])

            cell_self_update(afumigatus, afumigatus_cell, afumigatus_index)

            # ------------ interactions after this point

            # TODO: this should never be reached?! Make sure that we release iron when we kill the fungal cell
            #  release cell's iron pool back to voxel
            if afumigatus_cell['status'] in {FungalForm.DYING, FungalForm.DEAD}:
                # TODO: verify zyx (vs xyz)
                iron.grid[voxel.z, voxel.y, voxel.x] += afumigatus_cell['iron_pool']
                afumigatus_cell['iron_pool'] = 0.0
                afumigatus_cell['dead'] = True

            # interact with macrophages, possibly internalizing the aspergillus cell
            for macrophage_index in macrophage.cells.get_cells_in_voxel(voxel):
                macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_index]

                # Only healthy macrophages can internalize
                if macrophage_cell['status'] in {PhagocyteStatus.APOPTOTIC, PhagocyteStatus.NECROTIC,
                                                 PhagocyteStatus.DEAD}:
                    continue

                fungus_macrophage_interaction(afumigatus, afumigatus_cell, macrophage, macrophage_cell)

            # -----------

        return state

    # TODO: originally called in voxel
    # def elongate(self):
    #     septa = None
    #     if self.growable and self.boolean_network[Afumigatus.LIP] == 1:
    #         if self.status == Afumigatus.HYPHAE:
    #             if self.growth_iteration >= Constants.ITER_TO_GROW:
    #                 self.growth_iteration = 0
    #                 self.growable = False
    #                 self.branchable = True
    #                 self.iron_pool = self.iron_pool / 2.0
    #                 self.next_septa = Afumigatus(x_root=self.x_tip, y_root=self.y_tip, z_root=self.z_tip,
    #                                              x_tip=self.x_tip + self.dx, y_tip=self.y_tip + self.dy,
    #                                              z_tip=self.z_tip + self.dz,
    #                                              dx=self.dx, dy=self.dy, dz=self.dz, growth_iteration=0,
    #                                              iron_pool=0, status=Afumigatus.HYPHAE, state=self.state,
    #                                              is_root=False)
    #                 self.next_septa.previous_septa = self
    #                 self.next_septa.iron_pool = self.iron_pool
    #                 septa = self.next_septa
    #             else:
    #                 self.growth_iteration = self.growth_iteration + 1
    #         elif self.status == Afumigatus.GERM_TUBE:
    #             if self.growth_iteration >= Constants.ITER_TO_GROW:
    #                 self.status = Afumigatus.HYPHAE
    #                 self.x_tip = self.x_root + self.dx
    #                 self.y_tip = self.y_root + self.dy
    #                 self.z_tip = self.z_root + self.dz
    #             else:
    #                 self.growth_iteration = self.growth_iteration + 1
    #     return septa


def fungus_macrophage_interaction(afumigatus: AfumigatusState,
                                  afumigatus_cell: AfumigatusCellData,
                                  macrophage: MacrophageState,
                                  macrophage_cell: MacrophageCellData):
    probability_of_interaction = afumigatus.pr_ma_hyphae \
        if afumigatus_cell['status'] == FungalForm.HYPHAE \
        else afumigatus.pr_ma_phag

    if rg.random() < probability_of_interaction:
        internalize_aspergillus(macrophage_cell,
                                afumigatus_cell,
                                macrophage,
                                phagocytize=afumigatus_cell['status'] != FungalForm.HYPHAE)

        # unlink the fungal cell from its tree
        if afumigatus_cell['status'] == FungalForm.HYPHAE and \
                macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
            afumigatus_cell['status'] = FungalForm.DYING
            if afumigatus_cell['next_septa'] != -1:
                afumigatus.cells[afumigatus_cell['next_septa']]['is_root'] = True
            if afumigatus_cell['next_branch'] != -1:
                afumigatus.cells[afumigatus_cell['next_branch']]['is_root'] = True

            # TODO: what if the cell isn't a root? adding this. Will these be growable after a
            #  macrophage gets them? I haven't done anything with that.
            # TODO: this really should be spun off into its own method
            parent_id = afumigatus_cell['previous_septa']
            if parent_id != -1:
                parent_cell: AfumigatusCellData = afumigatus.cells[parent_id]
                if parent_cell['next_septa'] == afumigatus_cell:
                    parent_cell['next_septa'] = -1
                elif parent_cell['next_branch'] == afumigatus_cell:
                    parent_cell['next_branch'] = -1
                else:
                    assert False, "The fungal tree structure must be screwed up somehow"

        if afumigatus_cell['status'] == FungalForm.HYPHAE and \
                macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
            afumigatus_cell['status'] = FungalForm.DYING
            # TODO: careful cell tree deletion
            if afumigatus_cell['next_septa'] == -1:
                afumigatus_cell['next_septa']['is_root'] = True
                afumigatus_cell['next_septa']['previous_septa'] = -1
            if afumigatus_cell['next_branch'] == -1:
                afumigatus_cell['next_branch']['is_root'] = True
                afumigatus_cell['next_branch']['previous_septa'] = -1

        # TODO: Ask about this dead code.
        # else:
        #     if self.status == Afumigatus.HYPHAE and interactable.status == Macrophage.ACTIVE:
        #         interactable.engaged = True


def cell_self_update(afumigatus: AfumigatusState,
                     afumigatus_cell: AfumigatusCellData,
                     afumigatus_index: int) -> None:
    afumigatus_cell['activation_iteration'] += 1

    # resting conidia become swelling conidia after a number of iterations
    # (with some probability)
    if (afumigatus_cell['status'] == FungalForm.RESTING_CONIDIA and
            afumigatus_cell['activation_iteration'] >= afumigatus.iter_to_swelling and
            rg.random() < afumigatus.pr_aspergillus_change):
        afumigatus_cell['status'] = FungalForm.SWELLING_CONIDIA
        afumigatus_cell['activation_iteration'] = 0

    elif (afumigatus_cell['status'] == FungalForm.SWELLING_CONIDIA and
          afumigatus_cell['activation_iteration'] >= afumigatus.iter_to_germinate):
        afumigatus_cell['status'] = FungalForm.GERM_TUBE
        afumigatus_cell['activation_iteration'] = 0

    elif afumigatus_cell['status'] == FungalForm.DYING:
        # TODO: Henrique said something about the DYING state not being necessary. First glance in code
        #  suggests that this update only removes the cells from live counts
        afumigatus_cell['status'] = FungalForm.DEAD

    # TODO: this looks redundant/unnecessary. well, as long as we are careful about pruning the tree
    if afumigatus_cell['next_septa'] == -1:
        afumigatus_cell['growable'] = True

    # TODO: verify this, 1 turn on internalizing then free?
    if afumigatus_cell['state'] in {FungalState.INTERNALIZING, FungalState.RELEASING}:
        afumigatus_cell['state'] = FungalState.FREE

    # Note: called for every cell, but a no-op on non-root cells.
    diffuse_iron(afumigatus_index, afumigatus)

    # TODO: verify necessity. see above for next_septa
    if afumigatus_cell['next_branch'] == -1:
        afumigatus_cell['growable'] = True


def process_boolean_network(state: State,
                            boolean_network: np.ndarray,
                            bn_iteration: np.ndarray,
                            steps_to_eval: int):
    bn_iteration += 1
    bn_iteration %= steps_to_eval

    active_bool_net: np.ndarray = boolean_network[bn_iteration == 0, :]
    temp: np.ndarray = np.zeros(shape=active_bool_net.shape, dtype=bool)

    # TODO: verify array shape
    temp[:, NetworkSpecies.hapX] = ~active_bool_net[:, NetworkSpecies.SreA]
    temp[:, NetworkSpecies.sreA] = ~active_bool_net[:, NetworkSpecies.HapX]
    temp[:, NetworkSpecies.HapX] = active_bool_net[:, NetworkSpecies.hapX] & ~active_bool_net[:, NetworkSpecies.LIP]
    temp[:, NetworkSpecies.SreA] = active_bool_net[:, NetworkSpecies.sreA] & active_bool_net[:, NetworkSpecies.LIP]
    temp[:, NetworkSpecies.RIA] = ~active_bool_net[:, NetworkSpecies.SreA]
    temp[:, NetworkSpecies.EstB] = ~active_bool_net[:, NetworkSpecies.SreA]
    temp[:, NetworkSpecies.MirB] = active_bool_net[:, NetworkSpecies.HapX] & ~active_bool_net[:, NetworkSpecies.SreA]
    temp[:, NetworkSpecies.SidA] = active_bool_net[:, NetworkSpecies.HapX] & ~active_bool_net[:, NetworkSpecies.SreA]
    temp[:, NetworkSpecies.TAFC] = active_bool_net[:, NetworkSpecies.SidA]
    temp[:, NetworkSpecies.ICP] = ~active_bool_net[:, NetworkSpecies.HapX] & (active_bool_net[:, NetworkSpecies.VAC] |
                                                                              active_bool_net[:, NetworkSpecies.FC1fe])
    temp[:, NetworkSpecies.LIP] = \
        (active_bool_net[:, NetworkSpecies.Fe] & active_bool_net[:, NetworkSpecies.RIA]) | \
        lip_activation(state=state,
                       shape=temp.shape)
    temp[:, NetworkSpecies.CccA] = ~active_bool_net[:, NetworkSpecies.HapX]
    temp[:, NetworkSpecies.FC0fe] = active_bool_net[:, NetworkSpecies.SidA]
    temp[:, NetworkSpecies.FC1fe] = active_bool_net[:, NetworkSpecies.LIP] & active_bool_net[:, NetworkSpecies.FC0fe]
    temp[:, NetworkSpecies.VAC] = active_bool_net[:, NetworkSpecies.LIP] & active_bool_net[:, NetworkSpecies.CccA]
    temp[:, NetworkSpecies.ROS] = (active_bool_net[:, NetworkSpecies.Oxygen] &
                                   ~(active_bool_net[:, NetworkSpecies.SOD2_3] &
                                     active_bool_net[:, NetworkSpecies.ThP] &
                                     active_bool_net[:, NetworkSpecies.Cat1_2])) | \
                                  (active_bool_net[:, NetworkSpecies.ROS] &
                                   ~(active_bool_net[:, NetworkSpecies.SOD2_3] &
                                     (active_bool_net[:, NetworkSpecies.ThP] |
                                      active_bool_net[:, NetworkSpecies.Cat1_2])))
    temp[:, NetworkSpecies.Yap1] = active_bool_net[:, NetworkSpecies.ROS]
    temp[:, NetworkSpecies.SOD2_3] = active_bool_net[:, NetworkSpecies.Yap1]
    temp[:, NetworkSpecies.Cat1_2] = active_bool_net[:, NetworkSpecies.Yap1] & ~active_bool_net[:, NetworkSpecies.HapX]
    temp[:, NetworkSpecies.ThP] = active_bool_net[:, NetworkSpecies.Yap1]
    temp[:, NetworkSpecies.Fe] = 0  # might change according to iron environment?
    temp[:, NetworkSpecies.Oxygen] = 0

    # noinspection PyUnusedLocal
    active_bool_net = temp


def diffuse_iron(root_cell_index: int, afumigatus: AfumigatusState) -> None:
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


def lip_activation(state: State, shape) -> np.ndarray:
    afumigatus: AfumigatusState = state.afumigatus
    iron: IronState = state.iron

    molar_concentration = iron.grid / afumigatus.hyphae_volume
    activation = 1 - np.exp(-molar_concentration / afumigatus.kd_lip)
    return np.random.rand(*shape) < activation
