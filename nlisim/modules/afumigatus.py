from enum import IntEnum, unique
import math
from queue import Queue
import random
from typing import Any, Dict, Tuple

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.iron import IronState
from nlisim.modules.phagocyte import interact_with_aspergillus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType


@unique
class AfumigatusCellStatus(IntEnum):
    DEAD = 0
    RESTING_CONIDIA = 1
    SWELLING_CONIDIA = 2
    GERM_TUBE = 3
    HYPHAE = 4
    STERILE_CONIDIA = 5


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
    """Generate a random point on the unit 2-sphere in R^3 using Marsaglia's method"""
    # generate vector in unit disc
    u: np.ndarray = 2 * rg.random(size=2) - 1
    while np.linalg.norm(u) > 1.0:
        u = 2 * rg.random(size=2) - 1

    norm_squared_u = float(np.dot(u, u))
    return np.array(
        [
            2 * u[0] * np.sqrt(1 - norm_squared_u),
            2 * u[1] * np.sqrt(1 - norm_squared_u),
            1 - 2 * norm_squared_u,
        ],
        dtype=np.float64,
    )


class AfumigatusCellData(CellData):
    AFUMIGATUS_FIELDS: CellFields = [
        ('iron_pool', np.float64),  # units: atto-mol
        ('state', np.uint8),
        ('status', np.uint8),
        ('is_root', bool),
        ('vec', np.float64, 3),  # unit vector, length is in afumigatus.hyphal_length
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
            'vec': kwargs.get('vec', random_sphere_point()),  # dz, dy, dx
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
    pr_ma_hyphae: float  # units: probability
    pr_ma_hyphae_param: float  # units: M
    pr_ma_phag: float  # units: probability
    pr_ma_phag_param: float  # units: M
    pr_branch: float  # units: probability
    steps_to_bn_eval: int  # units: steps
    hyphal_length: float  # units: µm
    hyphae_volume: float  # units: L
    conidia_vol: float  # units: L
    kd_lip: float  # units: aM
    init_iron: float  # units: atto-mol
    time_to_swelling: float  # units: hours
    iter_to_swelling: int  # units: steps
    time_to_germinate: float  # units: hours
    iter_to_germinate: int  # units: steps
    time_to_grow: float  # units: hours
    iter_to_grow: int  # units: steps
    pr_aspergillus_change: float
    rel_phag_affinity_unit_t: float
    phag_affinity_t: float
    aspergillus_change_half_life: float  # units: hours


class Afumigatus(ModuleModel):
    name = 'afumigatus'
    StateClass = AfumigatusState

    from nlisim.modules.macrophage import MacrophageCellData, MacrophageState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus
        voxel_volume = state.voxel_volume  # units: L
        lung_tissue = state.lung_tissue

        afumigatus.pr_ma_hyphae_param = self.config.getfloat('pr_ma_hyphae_param')
        afumigatus.pr_ma_phag_param = self.config.getfloat('pr_ma_phag_param')
        afumigatus.phag_affinity_t = self.config.getfloat('phag_affinity_t')

        afumigatus.pr_branch = self.config.getfloat('pr_branch')  # units: probability
        afumigatus.steps_to_bn_eval = self.config.getint('steps_to_bn_eval')  # units: steps

        afumigatus.conidia_vol = self.config.getfloat('conidia_vol')  # units: L
        afumigatus.hyphae_volume = self.config.getfloat('hyphae_volume')  # units: L
        afumigatus.hyphal_length = self.config.getfloat('hyphal_length')  # units: µm

        afumigatus.kd_lip = self.config.getfloat('kd_lip')  # units: aM

        afumigatus.time_to_swelling = self.config.getfloat('time_to_swelling')  # units: hours
        afumigatus.time_to_germinate = self.config.getfloat('time_to_germinate')  # units: hours
        afumigatus.time_to_grow = self.config.getfloat('time_to_grow')  # units: hours
        afumigatus.aspergillus_change_half_life = self.config.getfloat(
            'aspergillus_change_half_life'
        )  # units: hours

        # computed values
        afumigatus.init_iron = afumigatus.kd_lip * afumigatus.conidia_vol  # units: aM*L = atto-mols

        afumigatus.rel_phag_affinity_unit_t = self.time_step / afumigatus.phag_affinity_t

        afumigatus.pr_ma_hyphae = -math.expm1(
            -afumigatus.rel_phag_affinity_unit_t / (afumigatus.pr_ma_hyphae_param * voxel_volume)
        )  # exponent units:  ?/(?*L) = TODO
        afumigatus.pr_ma_phag = -math.expm1(
            -afumigatus.rel_phag_affinity_unit_t / (voxel_volume * afumigatus.pr_ma_phag_param)
        )  # exponent units:  ?/(?*L) = TODO

        afumigatus.iter_to_swelling = max(
            0, int(afumigatus.time_to_swelling * (60 / self.time_step) - 2)
        )  # units: hours * (min/hour) / (min/step) = steps TODO: -2?
        afumigatus.iter_to_germinate = max(
            0, int(afumigatus.time_to_germinate * (60 / self.time_step) - 2)
        )  # units: hours * (min/hour) / (min/step) = steps TODO: -2?
        afumigatus.iter_to_grow = max(
            0, int(afumigatus.time_to_grow * 60 / self.time_step) - 1
        )  # units: hours * (min/hour) / (min/step) = steps
        afumigatus.pr_aspergillus_change = -math.log(0.5) / (
            afumigatus.aspergillus_change_half_life * (60 / self.time_step)
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
        for afumigatus_cell_index in afumigatus.cells.alive():
            # get cell and voxel position
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]
            voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])

            # ------------ update cell

            cell_self_update(afumigatus, afumigatus_cell, afumigatus_cell_index)

            # ------------ cell growth
            if (
                afumigatus_cell['state'] == AfumigatusCellState.FREE
                and lung_tissue[tuple(voxel)] != TissueType.AIR
            ):
                elongate(
                    afumigatus_cell, afumigatus_cell_index, afumigatus.iter_to_grow, afumigatus
                )
                if afumigatus_cell['next_septa'] != -1:  # only branch if we have already elongated
                    branch(afumigatus_cell, afumigatus_cell_index, afumigatus.pr_branch, afumigatus)

            # ------------ interactions after this point

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

                Afumigatus.fungus_macrophage_interaction(
                    afumigatus=afumigatus,
                    afumigatus_cell=afumigatus_cell,
                    afumigatus_cell_index=afumigatus_cell_index,
                    macrophage=macrophage,
                    macrophage_cell=macrophage_cell,
                    macrophage_cell_index=macrophage_index,
                    iron=iron,
                    grid=grid,
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
        macrophage_cell_index: int,
        iron: IronState,
        grid: RectangularGrid,
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

        interact_with_aspergillus(
            phagocyte_cell=macrophage_cell,
            phagocyte_cell_index=macrophage_cell_index,
            phagocyte_cells=macrophage.cells,
            aspergillus_cell=afumigatus_cell,
            aspergillus_cell_index=afumigatus_cell_index,
            phagocyte=macrophage,
            phagocytize=afumigatus_cell['status'] != AfumigatusCellStatus.HYPHAE,
        )

        # unlink the fungal cell from its tree
        if (
            afumigatus_cell['status'] == AfumigatusCellStatus.HYPHAE
            and macrophage_cell['status'] == PhagocyteStatus.ACTIVE
        ):
            Afumigatus.kill_fungal_cell(
                afumigatus, afumigatus_cell, afumigatus_cell_index, iron, grid
            )

    @staticmethod
    def kill_fungal_cell(
        afumigatus: AfumigatusState,
        afumigatus_cell: AfumigatusCellData,
        afumigatus_cell_index: int,
        iron: IronState,
        grid: RectangularGrid,
    ):
        """Kill a fungal cell.

        Unlinks the cell from its fungal tree and releases its iron.
        """
        # unlink from any children
        if afumigatus_cell['next_septa'] != -1:
            next_septa = afumigatus_cell['next_septa']
            afumigatus_cell['next_septa'] = -1
            afumigatus.cells[next_septa]['is_root'] = True
            afumigatus.cells[next_septa]['previous_septa'] = -1
        if afumigatus_cell['next_branch'] != -1:
            next_branch = afumigatus_cell['next_branch']
            afumigatus_cell['next_branch'] = -1
            afumigatus.cells[next_branch]['is_root'] = True
            afumigatus.cells[next_branch]['previous_septa'] = -1

        # unlink from parent, if exists
        parent_id = afumigatus_cell['previous_septa']
        if parent_id != -1:
            afumigatus_cell['previous_septa'] = -1
            parent_cell: AfumigatusCellData = afumigatus.cells[parent_id]
            if parent_cell['next_septa'] == afumigatus_cell_index:
                parent_cell['next_septa'] = -1
            elif parent_cell['next_branch'] == afumigatus_cell_index:
                parent_cell['next_branch'] = -1
            else:
                raise AssertionError("The fungal tree structure is malformed.")

        # kill the cell off and release its iron
        voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])
        iron.grid[voxel.z, voxel.y, voxel.x] += afumigatus_cell['iron_pool']
        afumigatus_cell['iron_pool'] = 0.0
        afumigatus_cell['dead'] = True
        afumigatus_cell['status'] = AfumigatusCellStatus.DEAD

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
    afumigatus: AfumigatusState,
    afumigatus_cell: AfumigatusCellData,
    afumigatus_cell_index: int,
) -> None:
    afumigatus_cell['activation_iteration'] += 1

    process_boolean_network(
        afumigatus_cell=afumigatus_cell,
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

    # TODO: verify this, 1 turn on internalizing then free?
    if afumigatus_cell['state'] in {
        AfumigatusCellState.INTERNALIZING,
        AfumigatusCellState.RELEASING,
    }:
        afumigatus_cell['state'] = AfumigatusCellState.FREE

    # Distribute iron evenly within fungal tree.
    # Note: called for every cell, but a no-op on non-root cells.
    diffuse_iron(afumigatus_cell_index, afumigatus)


def process_boolean_network(
    afumigatus: AfumigatusState,
    afumigatus_cell: AfumigatusCellData,
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
    q: Queue = Queue()
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
        afumigatus_cell['next_septa'] != -1  # already has a next septa
        or not afumigatus_cell['boolean_network'][NetworkSpecies.LIP]
    ):
        return

    hyphal_length: float = afumigatus.hyphal_length
    if afumigatus_cell['status'] == AfumigatusCellStatus.HYPHAE:
        if afumigatus_cell['growth_iteration'] < iter_to_grow:
            afumigatus_cell['growth_iteration'] += 1
        else:
            afumigatus_cell['growth_iteration'] = 0
            afumigatus_cell['iron_pool'] /= 2.0
            next_septa_center_point = (
                afumigatus_cell['point'] + hyphal_length * afumigatus_cell['vec']
            )  # center to center is two half hyphal lengths

            # create the new septa
            next_septa: CellData = AfumigatusCellData.create_cell(
                point=Point(
                    x=next_septa_center_point[2],
                    y=next_septa_center_point[1],
                    z=next_septa_center_point[0],
                ),
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
            # center of cell moves
            afumigatus_cell['point'] += (hyphal_length / 2) * afumigatus_cell['vec']
            afumigatus.cells.update_voxel_index([afumigatus_cell_index])


def branch(
    afumigatus_cell: AfumigatusCellData,
    afumigatus_cell_index: int,
    pr_branch: float,
    afumigatus: AfumigatusState,
):
    if (
        afumigatus_cell['next_branch'] != -1  # if it already has a branch
        or afumigatus_cell['status'] != AfumigatusCellStatus.HYPHAE
        or not afumigatus_cell['boolean_network'][NetworkSpecies.LIP]
    ):
        return

    hyphal_length: float = afumigatus.hyphal_length
    if rg.random() < pr_branch:
        # now we branch
        branch_vector = generate_branch_direction(cell_vec=afumigatus_cell['vec'])
        branch_center_point = (
            afumigatus_cell['point']
            + (hyphal_length / 2) * afumigatus_cell['vec']
            + (hyphal_length / 2) * branch_vector
        )  # center of new branch is offset by rest (half) of this septa and half of the new septa

        # create the new septa
        next_branch: CellData = AfumigatusCellData.create_cell(
            point=Point(
                x=branch_center_point[2], y=branch_center_point[1], z=branch_center_point[0]
            ),
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


def generate_branch_direction(cell_vec: np.ndarray) -> np.ndarray:
    """
    Generate a direction vector for branches.

    Parameters
    ----------
    cell_vec : np.ndarray
        a unit 3-vector

    Returns
    -------
    np.ndarray
        a random unit 3-vector at a 45 degree angle to `cell_vec`, sampled from the
        uniform distribution
    """
    # norm should be approx 1, can delete for performance
    cell_vec /= np.linalg.norm(cell_vec)

    # create orthogonal basis adapted to cell's direction
    # get first orthogonal vector
    u: np.ndarray
    epsilon = 0.1
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # if the cell vector isn't too close to +/- e1, generate the orthogonal vector using the cross
    # product with e1. otherwise use e2. (we can't be too close to both)
    u = (
        np.cross(cell_vec, e1)
        if (np.linalg.norm(cell_vec - e1) > epsilon and np.linalg.norm(cell_vec + e1) > epsilon)
        else np.cross(cell_vec, e2)
    )
    u /= np.linalg.norm(u)  # unlike the other normalizations, this is non-optional

    # get second orthogonal vector, orthogonal to both the cell vec and the first orthogonal vector
    v = np.cross(cell_vec, u)
    # norm should be approx 1, can delete for performance
    v /= np.linalg.norm(v)

    # change of coordinates matrix
    p_matrix = np.array([cell_vec, u, v]).T

    # form a random unit vector on a 45 degree cone
    theta = rg.random() * 2 * np.pi
    branch_direction = p_matrix @ np.array([1.0, np.cos(theta), np.sin(theta)]) / np.sqrt(2)
    # norm should be approx 1, can delete for performance
    branch_direction /= np.linalg.norm(branch_direction)

    return branch_direction
