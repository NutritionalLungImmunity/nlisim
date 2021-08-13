from enum import IntEnum, unique
import math
from queue import SimpleQueue
import random
from typing import Any, Dict, List, Tuple, Union

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.iron import IronState
from nlisim.modules.phagocyte import internalize_aspergillus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType


@unique
class AfumigatusCellStatus(IntEnum):
    RESTING_CONIDIA = 0
    SWELLING_CONIDIA = 1
    GERM_TUBE = 2
    HYPHAE = 3
    DYING = 4
    DEAD = 5
    STERILE_CONIDIA = 6


@unique
class NetworkSpecies(IntEnum):
    hapX = 0  # gene # noqa: N815
    sreA = 1  # gene # noqa: N815
    HapX = 2  # protein
    SreA = 3  # protein
    RIA = 4
    EstB = 5
    MirB = 6
    SidA = 7
    TAFC = 8
    ICP = 9
    LIP = 10
    CccA = 11
    FC0fe = 12
    FC1fe = 13
    VAC = 14
    ROS = 15
    Yap1 = 16
    SOD2_3 = 17
    Cat1_2 = 18
    ThP = 19
    Fe = 20
    Oxygen = 21


@unique
class AfumigatusCellState(IntEnum):
    FREE = 0
    INTERNALIZING = 1
    RELEASING = 2


def random_sphere_point() -> np.ndarray:
    """Generate a random point on the 2-sphere in R^3 using Marsaglia's method"""
    # generate vector in unit disc
    u: np.ndarray = rg.random(size=2)
    while np.linalg.norm(u) > 1.0:
        u = rg.random(size=2)

    normsq_u = float(np.dot(u, u))
    return np.array(
        [
            2 * u[0] * np.sqrt(1 - normsq_u),
            2 * u[1] * np.sqrt(1 - normsq_u),
            1 - 2 * normsq_u,
        ],
        dtype=np.float64,
    )


class AfumigatusCellData(CellData):
    AFUMIGATUS_FIELDS: List[Union[Tuple[str, Any], Tuple[str, Any, Any]]] = [
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
    def create_cell_tuple(cls, **kwargs) -> Tuple:
        initializer = {
            'iron_pool': kwargs.get('iron_pool', 0),
            'state': kwargs.get('state', AfumigatusCellState.FREE),
            'status': kwargs.get('status', AfumigatusCellStatus.RESTING_CONIDIA),
            'is_root': kwargs.get('is_root', True),
            'root': kwargs.get('root', np.zeros(3, dtype=np.float64)),
            'tip': kwargs.get('tip', np.zeros(3, dtype=np.float64)),
            'vec': kwargs.get('vec', random_sphere_point()),  # dx, dy, dz
            'growable': kwargs.get('growable', True),
            'branchable': kwargs.get('branchable', False),
            'activation_iteration': kwargs.get('activation_iteration', 0),
            'growth_iteration': kwargs.get('growth_iteration', 0),
            'boolean_network': kwargs.get('boolean_network', cls.initial_boolean_network()),
            'bn_iteration': kwargs.get('bn_iteration', 0),
            'next_branch': kwargs.get('next_branch', -1),
            'next_septa': kwargs.get('next_septa', -1),
            'previous_septa': kwargs.get('previous_septa', -1),
        }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in AfumigatusCellData.AFUMIGATUS_FIELDS]
        )

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        init_afumigatus_boolean_species = {
            NetworkSpecies.hapX: True,
            NetworkSpecies.sreA: False,
            NetworkSpecies.HapX: True,
            NetworkSpecies.SreA: False,
            NetworkSpecies.RIA: True,
            NetworkSpecies.EstB: True,
            NetworkSpecies.MirB: True,
            NetworkSpecies.SidA: True,
            NetworkSpecies.TAFC: True,
            NetworkSpecies.ICP: False,
            NetworkSpecies.LIP: False,
            NetworkSpecies.CccA: False,
            NetworkSpecies.FC0fe: True,
            NetworkSpecies.FC1fe: False,
            NetworkSpecies.VAC: False,
            NetworkSpecies.ROS: False,
            NetworkSpecies.Yap1: False,
            NetworkSpecies.SOD2_3: False,
            NetworkSpecies.Cat1_2: False,
            NetworkSpecies.ThP: False,
            NetworkSpecies.Fe: False,
            NetworkSpecies.Oxygen: False,
        }
        return np.asarray(
            [init_afumigatus_boolean_species[species] for species in NetworkSpecies], dtype=bool
        )


@attrs(kw_only=True, frozen=True, repr=False)
class AfumigatusCellList(CellList):
    CellDataClass = AfumigatusCellData


def cell_list_factory(self: 'AfumigatusState') -> AfumigatusCellList:
    return AfumigatusCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class AfumigatusState(ModuleState):
    cells: AfumigatusCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    pr_ma_hyphae: float
    pr_ma_hyphae_param: float
    pr_ma_phag: float
    pr_ma_phag_param: float
    pr_branch: float
    steps_to_bn_eval: int
    hyphae_volume: float
    kd_lip: float
    time_to_swelling: float
    iter_to_swelling: int
    time_to_germinate: float
    iter_to_germinate: int
    time_to_grow: float
    iter_to_grow: int
    pr_aspergillus_change: float
    init_iron: float
    conidia_vol: float
    rel_n_hyphae_int_unit_t: float
    rel_phag_affinity_unit_t: float
    phag_affinity_t: float
    aspergillus_change_half_life: float


class Afumigatus(ModuleModel):
    name = 'afumigatus'
    StateClass = AfumigatusState

    from nlisim.modules.macrophage import MacrophageCellData, MacrophageState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus
        voxel_volume = state.voxel_volume
        lung_tissue = state.lung_tissue
        time_step_size: float = self.time_step

        afumigatus.pr_ma_hyphae_param = self.config.getfloat('pr_ma_hyphae_param')
        afumigatus.pr_ma_phag_param = self.config.getfloat('pr_ma_phag_param')

        afumigatus.pr_branch = self.config.getfloat('pr_branch')
        afumigatus.steps_to_bn_eval = self.config.getint('steps_to_bn_eval')

        afumigatus.conidia_vol = self.config.getfloat('conidia_vol')
        afumigatus.hyphae_volume = self.config.getfloat('hyphae_volume')
        afumigatus.kd_lip = self.config.getfloat('kd_lip')

        afumigatus.time_to_swelling = self.config.getfloat('time_to_swelling')
        afumigatus.time_to_germinate = self.config.getfloat('time_to_germinate')
        afumigatus.time_to_grow = self.config.getfloat('time_to_grow')
        afumigatus.aspergillus_change_half_life = self.config.getfloat(
            'aspergillus_change_half_life'
        )

        afumigatus.phag_affinity_t = self.config.getfloat('phag_affinity_t')

        # computed values
        afumigatus.init_iron = afumigatus.kd_lip * afumigatus.conidia_vol

        afumigatus.rel_n_hyphae_int_unit_t = time_step_size / 60  # per hour
        afumigatus.rel_phag_affinity_unit_t = time_step_size / afumigatus.phag_affinity_t

        afumigatus.pr_ma_hyphae = 1 - math.exp(
            -afumigatus.rel_n_hyphae_int_unit_t / (voxel_volume * afumigatus.pr_ma_hyphae_param)
        )
        # TODO: = 1 - math.exp(-(1 / voxel_vol) * rel_n_hyphae_int_unit_t / 5.02201143330207e+9)
        #  kd ~10x neut. (ref 71)
        #  and below
        afumigatus.pr_ma_phag = 1 - math.exp(
            -afumigatus.rel_phag_affinity_unit_t
            / (voxel_volume * self.config.getfloat('pr_ma_phag_param'))
        )
        # pr_ma_phag = 1 - math.exp(-(
        #            1 / voxel_vol) * rel_phag_affinity_unit_t / 1.32489230813214e+10)
        # 30 min --> 1 - exp(-cells*t/kd) --> kd = 1.32489230813214e+10

        afumigatus.iter_to_swelling = int(
            afumigatus.time_to_swelling * (60 / time_step_size) - 2
        )  # TODO: -2?
        afumigatus.iter_to_germinate = int(
            afumigatus.time_to_germinate * (60 / time_step_size) - 2
        )  # TODO: -2?
        afumigatus.iter_to_grow = int(afumigatus.time_to_grow * 60 / time_step_size) - 1
        afumigatus.pr_aspergillus_change = -math.log(0.5) / (
            afumigatus.aspergillus_change_half_life * (60 / time_step_size)
        )

        # place cells for initial infection
        locations = list(zip(*np.where(lung_tissue == TissueType.EPITHELIUM)))
        dz_field: np.ndarray = state.grid.delta(axis=0)
        dy_field: np.ndarray = state.grid.delta(axis=1)
        dx_field: np.ndarray = state.grid.delta(axis=2)
        for vox_z, vox_y, vox_x in random.choices(
            locations, k=self.config.getint('init_infection_num')
        ):
            # the x,y,z coordinates are in the centers of the grids
            z = state.grid.z[vox_z]
            y = state.grid.y[vox_y]
            x = state.grid.x[vox_x]
            dz = dz_field[vox_z, vox_y, vox_x]
            dy = dy_field[vox_z, vox_y, vox_x]
            dx = dx_field[vox_z, vox_y, vox_x]
            afumigatus.cells.append(
                AfumigatusCellData.create_cell(
                    point=Point(
                        x=x + rg.uniform(-dx / 2, dx / 2),
                        y=y + rg.uniform(-dy / 2, dy / 2),
                        z=z + rg.uniform(-dz / 2, dz / 2),
                    ),
                    iron_pool=afumigatus.init_iron,
                )
            )

        return state

    def advance(self, state: State, previous_time: float) -> State:
        from nlisim.grid import RectangularGrid
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState, PhagocyteStatus

        afumigatus: AfumigatusState = state.afumigatus
        macrophage: MacrophageState = state.macrophage
        iron: IronState = state.iron
        grid: RectangularGrid = state.grid
        lung_tissue: np.ndarray = state.lung_tissue

        # update live cells
        for afumigatus_index in afumigatus.cells.alive():
            # get cell and voxel position
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_index]
            voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])

            # ------------ update cell

            cell_self_update(state, afumigatus, afumigatus_cell, afumigatus_index, voxel)

            # ------------ cell growth
            if (
                afumigatus_cell['state'] == AfumigatusCellState.FREE
                and lung_tissue[tuple(voxel)] != TissueType.AIR
            ):
                elongate(afumigatus_cell, afumigatus_index, afumigatus.iter_to_grow, afumigatus)
                branch(afumigatus_cell, afumigatus_index, afumigatus.pr_branch, afumigatus)

            # ------------ interactions after this point

            # interact with iron
            # TODO: this should never be reached?! Make sure that we release iron when we kill
            #  the fungal cell and release cell's iron pool back to voxel
            if afumigatus_cell['status'] in {AfumigatusCellStatus.DYING, AfumigatusCellStatus.DEAD}:
                iron.grid[voxel.z, voxel.y, voxel.x] += afumigatus_cell['iron_pool']
                afumigatus_cell['iron_pool'] = 0.0
                afumigatus_cell['dead'] = True

            # interact with macrophages, possibly internalizing the aspergillus cell
            for macrophage_index in macrophage.cells.get_cells_in_voxel(voxel):
                macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_index]

                # Only healthy macrophages can internalize
                if macrophage_cell['status'] in {
                    PhagocyteStatus.APOPTOTIC,
                    PhagocyteStatus.NECROTIC,
                    PhagocyteStatus.DEAD,
                }:
                    continue

                self.fungus_macrophage_interaction(
                    afumigatus, afumigatus_cell, afumigatus_index, macrophage, macrophage_cell
                )

            # -----------

        return state

    @staticmethod
    def fungus_macrophage_interaction(
        afumigatus: AfumigatusState,
        afumigatus_cell: AfumigatusCellData,
        afumigatus_cell_index: int,
        macrophage: 'MacrophageState',
        macrophage_cell: 'MacrophageCellData',
    ):
        from nlisim.modules.macrophage import PhagocyteStatus

        probability_of_interaction = (
            afumigatus.pr_ma_hyphae
            if afumigatus_cell['status'] == AfumigatusCellStatus.HYPHAE
            else afumigatus.pr_ma_phag
        )

        # return if they do not interact
        if rg.random() >= probability_of_interaction:
            return

        # now they interact

        internalize_aspergillus(
            macrophage_cell,
            afumigatus_cell,
            afumigatus_cell_index,
            macrophage,
            phagocytize=afumigatus_cell['status'] != AfumigatusCellStatus.HYPHAE,
        )

        # unlink the fungal cell from its tree
        if (
            afumigatus_cell['status'] == AfumigatusCellStatus.HYPHAE
            and macrophage_cell['status'] == PhagocyteStatus.ACTIVE
        ):
            afumigatus_cell['status'] = AfumigatusCellStatus.DYING
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
                if parent_cell['next_septa'] == afumigatus_cell_index:
                    parent_cell['next_septa'] = -1
                elif parent_cell['next_branch'] == afumigatus_cell_index:
                    parent_cell['next_branch'] = -1
                else:
                    raise AssertionError("The fungal tree structure must be screwed up somehow")

        # TODO: Ask about this dead code.
        # else:
        #     if self.status == Afumigatus.HYPHAE and interactable.status == Macrophage.ACTIVE:
        #         interactable.engaged = True

    def summary_stats(self, state: State) -> Dict[str, Any]:
        afumigatus: AfumigatusState = state.afumigatus
        live_fungus = afumigatus.cells.alive()

        max_index = max(map(int, AfumigatusCellStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    afumigatus.cells[afumigatus_cell_index]['status']
                    for afumigatus_cell_index in live_fungus
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        lip_active = int(
            np.sum(
                np.fromiter(
                    (
                        afumigatus.cells[afumigatus_cell_index]['boolean_network'][
                            NetworkSpecies.LIP
                        ]
                        for afumigatus_cell_index in live_fungus
                    ),
                    dtype=bool,
                )
            )
        )

        mirb_active = int(
            np.sum(
                np.fromiter(
                    (
                        afumigatus.cells[afumigatus_cell_index]['boolean_network'][
                            NetworkSpecies.MirB
                        ]
                        for afumigatus_cell_index in live_fungus
                    ),
                    dtype=bool,
                )
            )
        )

        estb_active = int(
            np.sum(
                np.fromiter(
                    (
                        afumigatus.cells[afumigatus_cell_index]['boolean_network'][
                            NetworkSpecies.EstB
                        ]
                        for afumigatus_cell_index in live_fungus
                    ),
                    dtype=bool,
                )
            )
        )

        tafc_active = int(
            np.sum(
                np.fromiter(
                    (
                        afumigatus.cells[afumigatus_cell_index]['boolean_network'][
                            NetworkSpecies.TAFC
                        ]
                        for afumigatus_cell_index in live_fungus
                    ),
                    dtype=bool,
                )
            )
        )

        return {
            'count': len(live_fungus),
            'resting conidia': int(status_counts[AfumigatusCellStatus.RESTING_CONIDIA]),
            'swelling conidia': int(status_counts[AfumigatusCellStatus.SWELLING_CONIDIA]),
            'sterile conidia': int(status_counts[AfumigatusCellStatus.STERILE_CONIDIA]),
            'germ tube': int(status_counts[AfumigatusCellStatus.GERM_TUBE]),
            'hyphae': int(status_counts[AfumigatusCellStatus.HYPHAE]),
            'LIP active': lip_active,
            'MirB active': mirb_active,
            'EstB active': estb_active,
            'TAFC active': tafc_active,
        }

    def visualization_data(self, state: State):
        return 'cells', state.afumigatus.cells


def cell_self_update(
    state: State,
    afumigatus: AfumigatusState,
    afumigatus_cell: AfumigatusCellData,
    afumigatus_index: int,
    voxel: Voxel,
) -> None:
    afumigatus_cell['activation_iteration'] += 1

    process_boolean_network(
        state=state,
        afumigatus_cell=afumigatus_cell,
        voxel=voxel,
        steps_to_eval=afumigatus.steps_to_bn_eval,
        afumigatus=afumigatus,
    )

    # resting conidia become swelling conidia after a number of iterations
    # (with some probability)
    if (
        afumigatus_cell['status'] == AfumigatusCellStatus.RESTING_CONIDIA
        and afumigatus_cell['activation_iteration'] >= afumigatus.iter_to_swelling
        and rg.random() < afumigatus.pr_aspergillus_change
    ):
        afumigatus_cell['status'] = AfumigatusCellStatus.SWELLING_CONIDIA
        afumigatus_cell['activation_iteration'] = 0

    elif (
        afumigatus_cell['status'] == AfumigatusCellStatus.SWELLING_CONIDIA
        and afumigatus_cell['activation_iteration'] >= afumigatus.iter_to_germinate
    ):
        afumigatus_cell['status'] = AfumigatusCellStatus.GERM_TUBE
        afumigatus_cell['activation_iteration'] = 0

    elif afumigatus_cell['status'] == AfumigatusCellStatus.DYING:
        # TODO: Henrique said something about the DYING state not being necessary. First glance in
        #  code suggests that this update only removes the cells from live counts
        afumigatus_cell['status'] = AfumigatusCellStatus.DEAD

    # TODO: this looks redundant/unnecessary. well, as long as we are careful about pruning the tree
    if afumigatus_cell['next_septa'] == -1:
        afumigatus_cell['growable'] = True

    # TODO: verify this, 1 turn on internalizing then free?
    if afumigatus_cell['state'] in {
        AfumigatusCellState.INTERNALIZING,
        AfumigatusCellState.RELEASING,
    }:
        afumigatus_cell['state'] = AfumigatusCellState.FREE

    # Note: called for every cell, but a no-op on non-root cells.
    diffuse_iron(afumigatus_index, afumigatus)

    # TODO: verify necessity. see above for next_septa
    if afumigatus_cell['next_branch'] == -1:
        afumigatus_cell['growable'] = True


def process_boolean_network(
    state: State,
    afumigatus: AfumigatusState,
    afumigatus_cell: AfumigatusCellData,
    voxel: Voxel,
    steps_to_eval: int,
):
    afumigatus_cell['bn_iteration'] += 1
    afumigatus_cell['bn_iteration'] %= steps_to_eval

    if afumigatus_cell['bn_iteration'] != 0:
        return

    bool_net = afumigatus_cell['boolean_network']

    temp: np.ndarray = np.zeros(shape=bool_net.shape, dtype=bool)

    temp[NetworkSpecies.hapX] = ~bool_net[NetworkSpecies.SreA]
    temp[NetworkSpecies.sreA] = ~bool_net[NetworkSpecies.HapX]
    temp[NetworkSpecies.HapX] = bool_net[NetworkSpecies.hapX] & ~bool_net[NetworkSpecies.LIP]
    temp[NetworkSpecies.SreA] = bool_net[NetworkSpecies.sreA] & bool_net[NetworkSpecies.LIP]
    temp[NetworkSpecies.RIA] = ~bool_net[NetworkSpecies.SreA]
    temp[NetworkSpecies.EstB] = ~bool_net[NetworkSpecies.SreA]
    temp[NetworkSpecies.MirB] = bool_net[NetworkSpecies.HapX] & ~bool_net[NetworkSpecies.SreA]
    temp[NetworkSpecies.SidA] = bool_net[NetworkSpecies.HapX] & ~bool_net[NetworkSpecies.SreA]
    temp[NetworkSpecies.TAFC] = bool_net[NetworkSpecies.SidA]
    temp[NetworkSpecies.ICP] = ~bool_net[NetworkSpecies.HapX] & (
        bool_net[NetworkSpecies.VAC] | bool_net[NetworkSpecies.FC1fe]
    )
    temp[NetworkSpecies.LIP] = (
        bool_net[NetworkSpecies.Fe] & bool_net[NetworkSpecies.RIA]
    ) | lip_activation(afumigatus=afumigatus, iron_pool=afumigatus_cell['iron_pool'])
    temp[NetworkSpecies.CccA] = ~bool_net[NetworkSpecies.HapX]
    temp[NetworkSpecies.FC0fe] = bool_net[NetworkSpecies.SidA]
    temp[NetworkSpecies.FC1fe] = bool_net[NetworkSpecies.LIP] & bool_net[NetworkSpecies.FC0fe]
    temp[NetworkSpecies.VAC] = bool_net[NetworkSpecies.LIP] & bool_net[NetworkSpecies.CccA]
    temp[NetworkSpecies.ROS] = (
        bool_net[NetworkSpecies.Oxygen]
        & ~(
            bool_net[NetworkSpecies.SOD2_3]
            & bool_net[NetworkSpecies.ThP]
            & bool_net[NetworkSpecies.Cat1_2]
        )
    ) | (
        bool_net[NetworkSpecies.ROS]
        & ~(
            bool_net[NetworkSpecies.SOD2_3]
            & (bool_net[NetworkSpecies.ThP] | bool_net[NetworkSpecies.Cat1_2])
        )
    )
    temp[NetworkSpecies.Yap1] = bool_net[NetworkSpecies.ROS]
    temp[NetworkSpecies.SOD2_3] = bool_net[NetworkSpecies.Yap1]
    temp[NetworkSpecies.Cat1_2] = bool_net[NetworkSpecies.Yap1] & ~bool_net[NetworkSpecies.HapX]
    temp[NetworkSpecies.ThP] = bool_net[NetworkSpecies.Yap1]
    temp[NetworkSpecies.Fe] = 0  # might change according to iron environment?
    temp[NetworkSpecies.Oxygen] = 0

    # copy temp back to bool_net
    np.copyto(dst=bool_net, src=temp)


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
    q: SimpleQueue = SimpleQueue()
    q.put(root_cell_index)
    while not q.empty():
        next_cell_index = q.get()
        tree_cells.add(next_cell_index)

        next_cell = afumigatus.cells[next_cell_index]
        total_iron += next_cell['iron_pool']

        if next_cell['next_branch'] >= 0:
            q.put(next_cell['next_branch'])
        if next_cell['next_septa'] >= 0:
            q.put(next_cell['next_septa'])

    # distribute the iron evenly
    iron_per_cell: float = total_iron / len(tree_cells)
    for tree_cell_index in tree_cells:
        afumigatus.cells[tree_cell_index]['iron_pool'] = iron_per_cell


def lip_activation(afumigatus: AfumigatusState, iron_pool: float) -> bool:
    molar_concentration = iron_pool / afumigatus.hyphae_volume
    activation = 1 - np.exp(-molar_concentration / afumigatus.kd_lip)
    return bool(rg.random() < activation)


def elongate(
    afumigatus_cell: AfumigatusCellData,
    afumigatus_cell_index: int,
    iter_to_grow: int,
    afumigatus: AfumigatusState,
):
    if (
        not afumigatus_cell['growable']
        or not afumigatus_cell['boolean_network'][NetworkSpecies.LIP]
    ):
        return

    if afumigatus_cell['status'] == AfumigatusCellStatus.HYPHAE:
        if afumigatus_cell['growth_iteration'] < iter_to_grow:
            afumigatus_cell['growth_iteration'] += 1
        else:
            afumigatus_cell['growth_iteration'] = 0
            afumigatus_cell['growable'] = False
            afumigatus_cell['branchable'] = True
            afumigatus_cell['iron_pool'] /= 2.0
            next_septa_root = afumigatus_cell['root'] + afumigatus_cell['vec']

            # create the new septa
            next_septa: CellData = AfumigatusCellData.create_cell(
                point=Point(x=next_septa_root[2], y=next_septa_root[1], z=next_septa_root[0]),
                root=next_septa_root,
                tip=next_septa_root + afumigatus_cell['vec'],
                vec=afumigatus_cell['vec'],
                iron_pool=0,
                status=AfumigatusCellStatus.HYPHAE,
                state=afumigatus_cell['state'],
                is_root=False,
            )
            next_septa_id: int = afumigatus.cells.append(next_septa)

            # link the septae together
            afumigatus_cell['next_septa'] = next_septa_id
            next_septa['previous_septa'] = afumigatus_cell_index

    elif afumigatus_cell['status'] == AfumigatusCellStatus.GERM_TUBE:
        if afumigatus_cell['growth_iteration'] < iter_to_grow:
            afumigatus_cell['growth_iteration'] += 1
        else:
            afumigatus_cell['status'] = AfumigatusCellStatus.HYPHAE
            afumigatus_cell['tip'] = afumigatus_cell['root'] + afumigatus_cell['vec']


def branch(
    afumigatus_cell: AfumigatusCellData,
    afumigatus_cell_index: int,
    pr_branch: float,
    afumigatus: AfumigatusState,
):
    if (
        not afumigatus_cell['branchable']
        or afumigatus_cell['status'] != AfumigatusCellStatus.HYPHAE
        or not afumigatus_cell['boolean_network'][NetworkSpecies.LIP]
    ):
        return

    if rg.random() < pr_branch:
        # now we branch
        branch_vector = generate_branch_direction(cell_vec=afumigatus_cell['vec'])
        root = afumigatus_cell['root']

        # create the new septa
        next_branch: CellData = AfumigatusCellData.create_cell(
            point=Point(x=root[2], y=root[1], z=root[0]),
            root=root,
            tip=root + branch_vector,
            vec=branch_vector,
            growth_iteration=-1,
            iron_pool=0,
            status=AfumigatusCellStatus.HYPHAE,
            state=afumigatus_cell['state'],
            is_root=False,
        )
        next_branch_id: int = afumigatus.cells.append(next_branch)

        # link them together
        afumigatus_cell['next_branch'] = next_branch_id
        next_branch['previous_septa'] = afumigatus_cell_index

    # only get one shot at branching
    afumigatus_cell['branchable'] = False


def generate_branch_direction(cell_vec: np.ndarray) -> np.ndarray:
    # form a random unit vector on a 45 degree cone
    theta = rg.random() * 2 * np.pi

    # create orthogonal basis adapted to cell's direction
    cell_vec_norm = np.linalg.norm(cell_vec)
    normed_cell_vec = cell_vec / cell_vec_norm

    # get first orthogonal vector
    u: np.ndarray
    epsilon = 0.01
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # if the cell vector isn't too close to e1, just use that. otherwise use e2.
    # (we can't be too close to both)
    if (
        np.linalg.norm(normed_cell_vec - e1) > epsilon
        or np.linalg.norm(normed_cell_vec + e1) > epsilon
    ):
        u = np.cross(normed_cell_vec, e1)
    else:
        u = np.cross(normed_cell_vec, e2)

    # get second orthogonal vector
    v = np.cross(normed_cell_vec, u)

    # change of coordinates matrix
    p_matrix = np.array([normed_cell_vec, u, v]).T
    branch_direction = (
        cell_vec_norm * p_matrix @ np.array([1.0, np.cos(theta), np.sin(theta)]) / np.sqrt(2)
    )

    return branch_direction
