import math
import random
from typing import Any, Dict, Tuple

import attr
from attr import attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.modules.phagocyte import (
    PhagocyteCellData,
    PhagocyteModel,
    PhagocyteModuleState,
    PhagocyteState,
    PhagocyteStatus,
)
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import choose_voxel_by_prob


class MacrophageCellData(PhagocyteCellData):
    MACROPHAGE_FIELDS: CellFields = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('fpn', bool),
        ('fpn_iteration', np.int64),
        ('tf', bool),  # TODO: descriptive name, transferrin?
        ('tnfa', bool),
        ('iron_pool', np.float64),
        ('status_iteration', np.uint64),
        ('velocity', np.float64, 3),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + MACROPHAGE_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'status': kwargs.get('status', PhagocyteStatus.RESTING),
            'state': kwargs.get('state', PhagocyteState.FREE),
            'fpn': kwargs.get('fpn', True),
            'fpn_iteration': kwargs.get('fpn_iteration', 0),
            'tf': kwargs.get('tf', False),
            'tnfa': kwargs.get('tnfa', False),
            'iron_pool': kwargs.get('iron_pool', 0.0),
            'status_iteration': kwargs.get('status_iteration', 0),
            'velocity': kwargs.get('root', np.zeros(3, dtype=np.float64)),
        }

        # ensure that these come in the correct order
        return PhagocyteCellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in MacrophageCellData.MACROPHAGE_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData


def cell_list_factory(self: 'MacrophageState') -> MacrophageCellList:
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(PhagocyteModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    time_to_rest: float  # units: min
    iter_to_rest: int  # units: steps
    time_to_change_state: float  # units: hours
    iter_to_change_state: int  # units: steps
    ma_internal_iron: float  # units: atto-mols
    prob_death_per_timestep: float  # units: probability * step^-1
    max_ma: int  # units: count
    min_ma: int  # units: count
    init_num_macrophages: int  # units: count
    recruitment_rate: float
    rec_bias: float
    drift_bias: float
    ma_move_rate_act: float  # µm/min
    ma_move_rate_rest: float  # µm/min
    half_life: float  # units: hours
    # UNUSED:
    # kd_ma_iron: float
    # ma_vol: float  # units: pL


class Macrophage(PhagocyteModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        from nlisim.util import TissueType

        macrophage: MacrophageState = state.macrophage
        lung_tissue = state.lung_tissue
        time_step_size: float = self.time_step

        macrophage.max_conidia = self.config.getint(
            'max_conidia'
        )  # (from phagocyte model) units: count
        macrophage.time_to_rest = self.config.getint('time_to_rest')  # units: min
        macrophage.time_to_change_state = self.config.getint('time_to_change_state')  # units: hours
        macrophage.ma_internal_iron = self.config.getfloat('ma_internal_iron')  # units: atto-mols

        macrophage.max_ma = self.config.getint('max_ma')  # units: count
        macrophage.min_ma = self.config.getint('min_ma')  # units: count
        macrophage.init_num_macrophages = self.config.getint('init_num_macrophages')  # units: count

        macrophage.recruitment_rate = self.config.getfloat('recruitment_rate')
        macrophage.rec_bias = self.config.getfloat('rec_bias')
        macrophage.drift_bias = self.config.getfloat('drift_bias')

        macrophage.ma_move_rate_act = self.config.getfloat('ma_move_rate_act')  # µm/min
        macrophage.ma_move_rate_rest = self.config.getfloat('ma_move_rate_rest')  # µm/min

        macrophage.half_life = self.config.getfloat('ma_half_life')  # units: hours

        # UNUSED:
        # macrophage.kd_ma_iron = self.config.getfloat('kd_ma_iron')
        # macrophage.ma_vol = self.config.getfloat('ma_vol')

        # computed values
        macrophage.iter_to_rest = int(
            macrophage.time_to_rest / self.time_step
        )  # units: min / (min/step) = steps
        macrophage.iter_to_change_state = int(
            macrophage.time_to_change_state * (60 / time_step_size)
        )  # units: hours * (min/hour) / (min/step) = step

        macrophage.prob_death_per_timestep = -math.log(0.5) / (
            macrophage.half_life * (60 / time_step_size)
        )  # units: 1/(  hours * (min/hour) / (min/step)  ) = 1/step

        # initialize cells, placing them randomly
        locations = list(zip(*np.where(lung_tissue != TissueType.AIR)))
        dz_field: np.ndarray = state.grid.delta(axis=0)
        dy_field: np.ndarray = state.grid.delta(axis=1)
        dx_field: np.ndarray = state.grid.delta(axis=2)
        for vox_z, vox_y, vox_x in random.choices(locations, k=macrophage.init_num_macrophages):
            # the x,y,z coordinates are in the centers of the grids
            z = state.grid.z[vox_z]
            y = state.grid.y[vox_y]
            x = state.grid.x[vox_x]
            dz = dz_field[vox_z, vox_y, vox_x]
            dy = dy_field[vox_z, vox_y, vox_x]
            dx = dx_field[vox_z, vox_y, vox_x]
            self.create_macrophage(
                state=state,
                x=x + rg.uniform(-dx / 2, dx / 2),
                y=y + rg.uniform(-dy / 2, dy / 2),
                z=z + rg.uniform(-dz / 2, dz / 2),
                iron_pool=macrophage.ma_internal_iron,
            )

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        macrophage: MacrophageState = state.macrophage

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell = macrophage.cells[macrophage_cell_index]

            num_cells_in_phagosome = np.sum(macrophage_cell['phagosome'] >= 0)

            self.update_status(state, macrophage_cell, num_cells_in_phagosome)

            if (
                num_cells_in_phagosome == 0
                and rg.uniform() < macrophage.prob_death_per_timestep
                and len(macrophage.cells.alive()) > macrophage.min_ma
            ):
                macrophage_cell['status'] = PhagocyteStatus.DEAD
                macrophage_cell['dead'] = True

            if not macrophage_cell['fpn']:
                if macrophage_cell['fpn_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['fpn_iteration'] = 0
                    macrophage_cell['fpn'] = True
                else:
                    macrophage_cell['fpn_iteration'] += 1

            # Movement
            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                max_move_step = (
                    macrophage.ma_move_rate_act * self.time_step
                )  # (µm/min) * (min/step) = µm * step
            else:
                max_move_step = (
                    macrophage.ma_move_rate_rest * self.time_step
                )  # (µm/min) * (min/step) = µm * step
            move_step: int = rg.poisson(max_move_step)
            # move the cell 1 µm, move_step number of times
            for _ in range(move_step):
                self.single_step_move(
                    state, macrophage_cell, macrophage_cell_index, macrophage.cells
                )

        # Recruitment
        self.recruit_macrophages(state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        macrophage: MacrophageState = state.macrophage
        live_macrophages = macrophage.cells.alive()

        max_index = max(map(int, PhagocyteStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    macrophage.cells[macrophage_cell_index]['status']
                    for macrophage_cell_index in live_macrophages
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        tnfa_active = int(
            np.sum(
                np.fromiter(
                    (
                        macrophage.cells[macrophage_cell_index]['tnfa']
                        for macrophage_cell_index in live_macrophages
                    ),
                    dtype=bool,
                )
            )
        )

        return {
            'count': len(live_macrophages),
            'inactive': int(status_counts[PhagocyteStatus.INACTIVE]),
            'inactivating': int(status_counts[PhagocyteStatus.INACTIVATING]),
            'resting': int(status_counts[PhagocyteStatus.RESTING]),
            'activating': int(status_counts[PhagocyteStatus.ACTIVATING]),
            'active': int(status_counts[PhagocyteStatus.ACTIVE]),
            'apoptotic': int(status_counts[PhagocyteStatus.APOPTOTIC]),
            'necrotic': int(status_counts[PhagocyteStatus.NECROTIC]),
            'anergic': int(status_counts[PhagocyteStatus.ANERGIC]),
            'interacting': int(status_counts[PhagocyteStatus.INTERACTING]),
            'TNFa active': tnfa_active,
        }

    def visualization_data(self, state: State):
        return 'cells', state.macrophage.cells

    def recruit_macrophages(self, state: State) -> None:
        """
        Recruit macrophages based on MIP1b activation

        Parameters
        ----------
        state : State
            global simulation state

        Returns
        -------
        nothing
        """
        from nlisim.modules.mip1b import MIP1BState
        from nlisim.util import TissueType, activation_function

        macrophage: MacrophageState = state.macrophage
        mip1b: MIP1BState = state.mip1b
        voxel_volume: float = state.voxel_volume
        space_volume: float = state.space_volume
        lung_tissue = state.lung_tissue

        # 1. compute number of macrophages to recruit
        num_live_macrophages = len(macrophage.cells.alive())
        avg = (
            macrophage.recruitment_rate
            * np.sum(mip1b.grid)
            * (1 - num_live_macrophages / macrophage.max_ma)
            / (mip1b.k_d * space_volume)
        )
        number_to_recruit = max(
            np.random.poisson(avg) if avg > 0 else 0, macrophage.min_ma - num_live_macrophages
        )
        # 2. get voxels for new macrophages, based on activation
        if number_to_recruit > 0:
            activation_voxels = zip(
                *np.where(
                    np.logical_and(
                        activation_function(
                            x=mip1b.grid,
                            k_d=mip1b.k_d,
                            h=self.time_step / 60,
                            volume=voxel_volume,
                            b=macrophage.rec_bias,
                        )
                        < rg.uniform(size=mip1b.grid.shape),
                        lung_tissue != TissueType.AIR,
                    )
                )
            )
            dz_field: np.ndarray = state.grid.delta(axis=0)
            dy_field: np.ndarray = state.grid.delta(axis=1)
            dx_field: np.ndarray = state.grid.delta(axis=2)
            for coordinates in rg.choice(
                tuple(activation_voxels), size=number_to_recruit, replace=True
            ):
                vox_z, vox_y, vox_x = coordinates
                # the x,y,z coordinates are in the centers of the grids
                z = state.grid.z[vox_z]
                y = state.grid.y[vox_y]
                x = state.grid.x[vox_x]
                dz = dz_field[vox_z, vox_y, vox_x]
                dy = dy_field[vox_z, vox_y, vox_x]
                dx = dx_field[vox_z, vox_y, vox_x]
                self.create_macrophage(
                    state=state,
                    x=x + rg.uniform(-dx / 2, dx / 2),
                    y=y + rg.uniform(-dy / 2, dy / 2),
                    z=z + rg.uniform(-dz / 2, dz / 2),
                )

    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, voxel: Voxel
    ) -> Point:
        """
        Calculate a 1µm movement of a macrophage

        Parameters
        ----------
        state : State
            global simulation state
        cell : MacrophageCellData
            a macrophage cell
        voxel : Voxel
            current voxel position of the macrophage

        Returns
        -------
        Point
            the new position of the macrophage
        """
        # macrophages are attracted by MIP1b
        from nlisim.modules.mip1b import MIP1BState
        from nlisim.util import TissueType, activation_function

        macrophage: MacrophageState = state.macrophage
        mip1b: MIP1BState = state.mip1b
        grid: RectangularGrid = state.grid
        lung_tissue: np.ndarray = state.lung_tissue
        voxel_volume: float = state.voxel_volume

        # compute chemokine influence on velocity, with some randomness.
        # macrophage has a non-zero probability of moving into non-air voxels.
        # if not any of these, stay in place. This could happen if e.g. you are
        # somehow stranded in air.
        nearby_voxels: Tuple[Voxel, ...] = tuple(grid.get_adjacent_voxels(voxel, corners=True))
        weights = np.array(
            [
                0.0
                if lung_tissue[tuple(vxl)] == TissueType.AIR
                else activation_function(
                    x=mip1b.grid[tuple(vxl)],
                    k_d=mip1b.k_d,
                    h=self.time_step / 60,  # units: (min/step) / (min/hour)
                    volume=voxel_volume,
                    b=1,
                )
                + macrophage.drift_bias
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

        # average and re-normalize with existing velocity
        dp_dt += cell['velocity']
        norm = np.linalg.norm(dp_dt)
        if norm > 0.0:
            dp_dt /= norm

        # we need to determine if this movement will put us into an air voxel. This can happen
        # when pushed there by momentum. If that happens, we stay in place and zero out the
        # momentum. Otherwise, velocity is updated to dp/dt and movement is as expected.
        new_position = cell['point'] + dp_dt
        new_voxel: Voxel = grid.get_voxel(new_position)
        if state.lung_tissue[tuple(new_voxel)] == TissueType.AIR:
            cell['velocity'][:] = np.zeros(3, dtype=np.float64)
            return cell['point']
        else:
            cell['velocity'][:] = dp_dt
            return new_position

    @staticmethod
    def create_macrophage(*, state: State, x: float, y: float, z: float, **kwargs) -> None:
        """
        Create a new macrophage cell

        Parameters
        ----------
        state : State
            global simulation state
        x : float
        y : float
        z : float
            coordinates of created macrophage
        kwargs
            parameters for macrophage, will give

        Returns
        -------
        nothing
        """
        macrophage: MacrophageState = state.macrophage

        # use default value of iron pool if not present
        iron_pool = kwargs.get('iron_pool', macrophage.ma_internal_iron)
        kwargs.pop('iron_pool', None)

        macrophage.cells.append(
            MacrophageCellData.create_cell(
                point=Point(x=x, y=y, z=z),
                iron_pool=iron_pool,
                **kwargs,
            )
        )

    def update_status(
        self, state: State, macrophage_cell: MacrophageCellData, num_cells_in_phagosome
    ) -> None:
        """
        Update the status of the cell, progressing between states after a certain number of ticks.

        Parameters
        ----------
        state : State
            global simulation state
        macrophage_cell : MacrophageCellData
        num_cells_in_phagosome

        Returns
        -------
        nothing
        """
        macrophage: MacrophageState = state.macrophage

        if macrophage_cell['status'] == PhagocyteStatus.NECROTIC:
            # TODO: what about APOPTOTIC?
            self.release_phagosome(state, macrophage_cell)

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
