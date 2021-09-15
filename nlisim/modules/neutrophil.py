import math
import random
from typing import Any, Dict, Tuple

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.modules.mip2 import MIP2State
from nlisim.modules.phagocyte import (
    PhagocyteCellData,
    PhagocyteModel,
    PhagocyteModuleState,
    PhagocyteState,
    PhagocyteStatus,
    interact_with_aspergillus,
)
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType, activation_function, choose_voxel_by_prob

MAX_CONIDIA = (
    50  # note: this the max that we can set the max to. i.e. not an actual model parameter
)


class NeutrophilCellData(PhagocyteCellData):
    NEUTROPHIL_FIELDS: CellFields = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('iron_pool', np.float64),  # units: atto-mol
        ('tnfa', bool),
        ('status_iteration', np.uint),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + NEUTROPHIL_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'status': kwargs.get('status', PhagocyteStatus.RESTING),
            'state': kwargs.get('state', PhagocyteState.FREE),
            'iron_pool': kwargs.get('iron_pool', 0.0),
            'tnfa': kwargs.get('tnfa', False),
            'status_iteration': kwargs.get('status_iteration', 0),
        }

        # ensure that these come in the correct order
        return PhagocyteCellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in NeutrophilCellData.NEUTROPHIL_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData


def cell_list_factory(self: 'NeutrophilState') -> NeutrophilCellList:
    return NeutrophilCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class NeutrophilState(PhagocyteModuleState):
    cells: NeutrophilCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    half_life: float  # units: hours
    apoptosis_probability: float  # units: probability
    time_to_change_state: float  # units: hours
    iter_to_change_state: int  # units: steps
    pr_n_hyphae: float  # units: probability
    pr_n_hyphae_param: float
    pr_n_phagocyte: float  # units: probability
    pr_n_phagocyte_param: float
    recruitment_rate: float
    rec_bias: float
    max_neutrophils: float  # TODO: 0.5?
    n_frac: float
    drift_bias: float
    n_move_rate_act: float  # units: µm
    n_move_rate_rest: float  # units: µm
    init_num_neutrophils: int  # units: count


class Neutrophil(PhagocyteModel):
    name = 'neutrophil'
    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        voxel_volume = state.voxel_volume
        lung_tissue = state.lung_tissue

        neutrophil.init_num_neutrophils = self.config.getint('init_num_neutrophils')  # units: count

        neutrophil.time_to_change_state = self.config.getfloat(
            'time_to_change_state'
        )  # units: hours
        neutrophil.max_conidia = self.config.getint(
            'max_conidia'
        )  # (from phagocyte model) units: count

        neutrophil.recruitment_rate = self.config.getfloat('recruitment_rate')
        neutrophil.rec_bias = self.config.getfloat('rec_bias')
        neutrophil.max_neutrophils = self.config.getfloat('max_neutrophils')  # units: count
        neutrophil.n_frac = self.config.getfloat('n_frac')

        neutrophil.drift_bias = self.config.getfloat('drift_bias')
        neutrophil.n_move_rate_act = self.config.getfloat('n_move_rate_act')
        neutrophil.n_move_rate_rest = self.config.getfloat('n_move_rate_rest')

        neutrophil.pr_n_hyphae_param = self.config.getfloat('pr_n_hyphae_param')
        neutrophil.pr_n_phagocyte_param = self.config.getfloat('pr_n_phagocyte_param')

        neutrophil.half_life = self.config.getfloat('half_life')  # units: hours

        # computed values
        neutrophil.apoptosis_probability = -math.log(0.5) / (
            neutrophil.half_life * (60 / self.time_step)  # units: hours*(min/hour)/(min/step)=steps
        )  # units: probability
        neutrophil.iter_to_change_state = int(
            neutrophil.time_to_change_state * 60 / self.time_step
        )  # units: hours * (min/hour) / (min/step) = steps
        neutrophil.pr_n_hyphae = -math.expm1(
            -self.time_step / 60 / (voxel_volume * neutrophil.pr_n_hyphae_param)
        )  # units: probability
        neutrophil.pr_n_phagocyte = -math.expm1(
            -self.time_step / 60 / (voxel_volume * neutrophil.pr_n_phagocyte_param)
        )  # units: probability

        # place initial neutrophils
        locations = list(zip(*np.where(lung_tissue != TissueType.AIR)))
        dz_field: np.ndarray = state.grid.delta(axis=0)
        dy_field: np.ndarray = state.grid.delta(axis=1)
        dx_field: np.ndarray = state.grid.delta(axis=2)
        for vox_z, vox_y, vox_x in random.choices(locations, k=neutrophil.init_num_neutrophils):
            # the x,y,z coordinates are in the centers of the grids
            z = state.grid.z[vox_z]
            y = state.grid.y[vox_y]
            x = state.grid.x[vox_x]
            dz = dz_field[vox_z, vox_y, vox_x]
            dy = dy_field[vox_z, vox_y, vox_x]
            dx = dx_field[vox_z, vox_y, vox_x]

            self.create_neutrophil(
                state=state,
                x=x + rg.uniform(-dx / 2, dx / 2),
                y=y + rg.uniform(-dy / 2, dy / 2),
                z=z + rg.uniform(-dz / 2, dz / 2),
            )

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            Afumigatus,
            AfumigatusCellData,
            AfumigatusCellStatus,
            AfumigatusState,
        )
        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState

        neutrophil: NeutrophilState = state.neutrophil
        macrophage: MacrophageState = state.macrophage
        afumigatus: AfumigatusState = state.afumigatus
        iron: IronState = state.iron
        grid: RectangularGrid = state.grid
        voxel_volume: float = state.voxel_volume
        space_volume: float = state.space_volume

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])

            self.update_status(state, neutrophil_cell)

            # ---------- interactions

            # dead and dying cells release iron
            if neutrophil_cell['status'] in {
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.DEAD,
            }:
                iron.grid[tuple(neutrophil_cell_voxel)] += neutrophil_cell['iron_pool']
                neutrophil_cell['iron_pool'] = 0
                neutrophil_cell['dead'] = True

            # interact with fungus
            if neutrophil_cell['state'] == PhagocyteState.FREE and neutrophil_cell[
                'status'
            ] not in {
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.DEAD,
            }:
                # get fungal cells in this voxel
                local_aspergillus = afumigatus.cells.get_cells_in_voxel(neutrophil_cell_voxel)
                for aspergillus_cell_index in local_aspergillus:
                    aspergillus_cell: AfumigatusCellData = afumigatus.cells[aspergillus_cell_index]
                    if aspergillus_cell['dead']:
                        continue

                    if aspergillus_cell['status'] in {
                        AfumigatusCellStatus.HYPHAE,
                        AfumigatusCellStatus.GERM_TUBE,
                    }:
                        # possibly kill the fungal cell, extracellularly
                        if rg.uniform() < neutrophil.pr_n_hyphae:
                            interact_with_aspergillus(
                                phagocyte_cell=neutrophil_cell,
                                aspergillus_cell=aspergillus_cell,
                                aspergillus_cell_index=aspergillus_cell_index,
                                phagocyte=neutrophil,
                            )
                            Afumigatus.kill_fungal_cell(
                                afumigatus=afumigatus,
                                afumigatus_cell=aspergillus_cell,
                                afumigatus_cell_index=aspergillus_cell_index,
                                iron=iron,
                                grid=grid,
                            )
                        else:
                            neutrophil_cell['state'] = PhagocyteState.INTERACTING

                    elif aspergillus_cell['status'] == AfumigatusCellStatus.SWELLING_CONIDIA:
                        if rg.uniform() < neutrophil.pr_n_phagocyte:
                            interact_with_aspergillus(
                                phagocyte_cell=neutrophil_cell,
                                aspergillus_cell=aspergillus_cell,
                                aspergillus_cell_index=aspergillus_cell_index,
                                phagocyte=neutrophil,
                            )

            # interact with macrophages:
            # if we are apoptotic, give our iron and phagosome to a nearby
            # present macrophage (if empty)
            if neutrophil_cell['status'] == PhagocyteStatus.APOPTOTIC:
                local_macrophages = macrophage.cells.get_cells_in_voxel(neutrophil_cell_voxel)
                for macrophage_index in local_macrophages:
                    macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_index]
                    macrophage_num_cells_in_phagosome = np.sum(macrophage_cell['phagosome'] >= 0)
                    # TODO: Henrique, why only if empty?
                    if macrophage_num_cells_in_phagosome == 0:
                        macrophage_cell['phagosome'] = neutrophil_cell['phagosome']
                        macrophage_cell['iron_pool'] += neutrophil_cell['iron_pool']
                        neutrophil_cell['iron_pool'] = 0.0
                        neutrophil_cell['status'] = PhagocyteStatus.DEAD
                        macrophage_cell['status'] = PhagocyteStatus.INACTIVE

            # Movement
            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                max_move_step = neutrophil.n_move_rate_act * self.time_step
            else:
                max_move_step = neutrophil.n_move_rate_rest * self.time_step
            move_step: int = rg.poisson(max_move_step)
            # move the cell 1 µm, move_step number of times
            for _ in range(move_step):
                self.single_step_move(
                    state, neutrophil_cell, neutrophil_cell_index, neutrophil.cells
                )
            # TODO: understand the meaning of the parameter here: moving randomly n steps is
            #  different than moving n steps in a random direction. Which is it?

        # Recruitment
        self.recruit_neutrophils(state, space_volume, voxel_volume)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        neutrophil: NeutrophilState = state.neutrophil
        live_neutrophils = neutrophil.cells.alive()

        max_index = max(map(int, PhagocyteStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    neutrophil.cells[neutrophil_cell_index]['status']
                    for neutrophil_cell_index in live_neutrophils
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        tnfa_active = int(
            np.sum(
                np.fromiter(
                    (
                        neutrophil.cells[neutrophil_cell_index]['tnfa']
                        for neutrophil_cell_index in live_neutrophils
                    ),
                    dtype=bool,
                )
            )
        )

        return {
            'count': len(neutrophil.cells.alive()),
            'inactive': int(status_counts[PhagocyteStatus.INACTIVE]),
            'inactivating': int(status_counts[PhagocyteStatus.INACTIVATING]),
            'resting': int(status_counts[PhagocyteStatus.RESTING]),
            'activating': int(status_counts[PhagocyteStatus.ACTIVATING]),
            'active': int(status_counts[PhagocyteStatus.ACTIVE]),
            'apoptotic': int(status_counts[PhagocyteStatus.APOPTOTIC]),
            'necrotic': int(status_counts[PhagocyteStatus.NECROTIC]),
            'interacting': int(status_counts[PhagocyteStatus.INTERACTING]),
            'TNFa active': tnfa_active,
        }

    def visualization_data(self, state: State):
        return 'cells', state.neutrophil.cells

    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, voxel: Voxel
    ) -> Point:
        """
        Calculate a 1µm movement of a neutrophil

        Parameters
        ----------
        state : State
            global simulation state
        cell : NeutrophilCellData
            a neutrophil cell
        voxel : Voxel
            current voxel position of the neutrophil

        Returns
        -------
        Point
            the new position of the neutrophil
        """
        # neutrophils are attracted by MIP2

        neutrophil: NeutrophilState = state.neutrophil
        mip2: MIP2State = state.mip2
        grid: RectangularGrid = state.grid
        lung_tissue: np.ndarray = state.lung_tissue
        voxel_volume: float = state.voxel_volume

        # neutrophil has a non-zero probability of moving into non-air voxels
        nearby_voxels: Tuple[Voxel, ...] = tuple(grid.get_adjacent_voxels(voxel))
        weights = np.array(
            [
                0.0
                if lung_tissue[tuple(vxl)] == TissueType.AIR
                else activation_function(
                    x=mip2.grid[tuple(vxl)],
                    k_d=mip2.k_d,
                    h=self.time_step / 60,  # units: (min/step) / (min/hour)
                    volume=voxel_volume,
                    b=1,
                )
                + neutrophil.drift_bias
                for vxl in nearby_voxels
            ],
            dtype=np.float64,
        )

        voxel_movement_direction: Voxel = choose_voxel_by_prob(
            voxels=nearby_voxels, default_value=voxel, weights=weights
        )

        # get normalized direction vector
        dp_dt: np.ndarray = grid.get_voxel_center(voxel_movement_direction) - grid.get_voxel_center(
            voxel
        )
        norm = np.linalg.norm(dp_dt)
        if norm > 0.0:
            dp_dt /= norm

        return cell['point'] + dp_dt

    def update_status(self, state: State, neutrophil_cell: NeutrophilCellData) -> None:
        """
        Update the status of the cell, progressing between states after a certain number of ticks.

        Parameters
        ----------
        state : State
            global simulation state
        neutrophil_cell : NeutrophilCellData

        Returns
        -------
        nothing
        """
        neutrophil: NeutrophilState = state.neutrophil

        if neutrophil_cell['status'] in {PhagocyteStatus.NECROTIC, PhagocyteStatus.APOPTOTIC}:
            self.release_phagosome(state, neutrophil_cell)
            # releases iron & dies later

        elif rg.uniform() < neutrophil.apoptosis_probability:
            neutrophil_cell['status'] = PhagocyteStatus.APOPTOTIC

        elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
            if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                neutrophil_cell['status_iteration'] = 0
                neutrophil_cell['tnfa'] = False
                neutrophil_cell['status'] = PhagocyteStatus.RESTING
                neutrophil_cell['state'] = PhagocyteState.FREE
            else:
                neutrophil_cell['status_iteration'] += 1

        elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVATING:
            if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                neutrophil_cell['status_iteration'] = 0
                neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
            else:
                neutrophil_cell['status_iteration'] += 1

    def recruit_neutrophils(self, state: State, space_volume: float, voxel_volume: float) -> None:
        """
        Recruit neutrophils based on MIP2 activation

        Parameters
        ----------
        state : State
            global simulation state
        space_volume : float
        voxel_volume : float

        Returns
        -------
        nothing
        """
        from nlisim.modules.mip2 import MIP2State
        from nlisim.util import TissueType, activation_function

        neutrophil: NeutrophilState = state.neutrophil
        mip2: MIP2State = state.mip2
        lung_tissue = state.lung_tissue

        # 1. compute number of neutrophils to recruit
        num_live_neutrophils = len(neutrophil.cells.alive())
        avg = (
            neutrophil.recruitment_rate
            * neutrophil.n_frac
            * np.sum(mip2.grid)
            * (1 - num_live_neutrophils / neutrophil.max_neutrophils)
            / (mip2.k_d * space_volume)
        )
        number_to_recruit = np.random.poisson(avg) if avg > 0 else 0
        if number_to_recruit <= 0:
            return
        # 2. get voxels for new macrophages, based on activation
        activation_voxels = tuple(
            zip(
                *np.where(
                    np.logical_and(
                        activation_function(
                            x=mip2.grid,
                            k_d=mip2.k_d,
                            h=self.time_step / 60,  # units: (min/step) / (min/hour)
                            volume=voxel_volume,
                            b=neutrophil.rec_bias,
                        )
                        < rg.uniform(size=mip2.grid.shape),
                        lung_tissue != TissueType.AIR,
                    )
                )
            )
        )

        dz_field: np.ndarray = state.grid.delta(axis=0)
        dy_field: np.ndarray = state.grid.delta(axis=1)
        dx_field: np.ndarray = state.grid.delta(axis=2)
        for coordinates in rg.choice(activation_voxels, size=number_to_recruit, replace=True):
            vox_z, vox_y, vox_x = coordinates
            # the x,y,z coordinates are in the centers of the grids
            z = state.grid.z[vox_z]
            y = state.grid.y[vox_y]
            x = state.grid.x[vox_x]
            dz = dz_field[vox_z, vox_y, vox_x]
            dy = dy_field[vox_z, vox_y, vox_x]
            dx = dx_field[vox_z, vox_y, vox_x]
            self.create_neutrophil(
                state=state,
                x=x + rg.uniform(-dx / 2, dx / 2),
                y=y + rg.uniform(-dy / 2, dy / 2),
                z=z + rg.uniform(-dz / 2, dz / 2),
            )

    @staticmethod
    def create_neutrophil(state: State, x: float, y: float, z: float, **kwargs) -> None:
        """
        Create a new neutrophil cell

        Parameters
        ----------
        state : State
            global simulation state
        x : float
        y : float
        z : float
            coordinates of created neutrophil
        kwargs
            parameters for neutrophil, will give

        Returns
        -------
        nothing
        """
        neutrophil: NeutrophilState = state.neutrophil

        # use default value of iron pool if not present
        iron_pool = kwargs.get('iron_pool', 0.0)
        kwargs.pop('iron_pool', None)

        neutrophil.cells.append(
            NeutrophilCellData.create_cell(
                point=Point(x=x, y=y, z=z), iron_pool=iron_pool, **kwargs
            )
        )
