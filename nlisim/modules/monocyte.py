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


class MonocyteCellData(PhagocyteCellData):
    MONOCYTE_FIELDS: CellFields = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('fpn', bool),
        ('fpn_iteration', np.int64),
        ('tf', bool),  # TODO: descriptive name, transferrin?
        ('tnfa', bool),
        ('iron_pool', np.float64),
        ('status_iteration', np.uint64),
        ('velocity', np.float64, 3),
        ('Cd14', np.uint8),
        ('Cd16', np.uint8),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + MONOCYTE_FIELDS, align=True
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
            [initializer[key] for key, *_ in MonocyteCellData.MONOCYTE_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class MonocyteCellList(CellList):
    CellDataClass = MonocyteCellData


def cell_list_factory(self: 'MonocyteState') -> MonocyteCellData:
    return MonocyteCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MonocyteState(PhagocyteModuleState):
    cells: MonocyteCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    time_to_rest: float  # units: min
    iter_to_rest: int  # units: steps
    time_to_change_state: float  # units: hours
    iter_to_change_state: int  # units: steps
    ma_internal_iron: float  # units: atto-mols
    prob_death_per_timestep: float  # units: probability * step^-1
    max_ma: int  # units: count
    min_ma: int  # units: count
    init_num_monocytes: int  # units: count
    recruitment_rate: float
    rec_bias: float
    drift_bias: float
    ma_move_rate_act: float  # µm/min
    ma_move_rate_rest: float  # µm/min
    half_life: float  # units: hours
    # UNUSED:
    # kd_ma_iron: float
    # ma_vol: float  # units: pL


class Monocyte(PhagocyteModel):
    name = 'monocyte'
    StateClass = MonocyteState

    def initialize(self, state: State):
        from nlisim.util import TissueType

        monocyte: MonocyteState = state.monocyte
        lung_tissue = state.lung_tissue
        time_step_size: float = self.time_step

        monocyte.max_conidia = self.config.getint(
            'max_conidia'
        )  # (from phagocyte model) units: count
        monocyte.time_to_rest = self.config.getint('time_to_rest')  # units: min
        monocyte.time_to_change_state = self.config.getint('time_to_change_state')  # units: hours
        monocyte.ma_internal_iron = self.config.getfloat('ma_internal_iron')  # units: atto-mols

        monocyte.max_ma = self.config.getint('max_ma')  # units: count
        monocyte.min_ma = self.config.getint('min_ma')  # units: count
        monocyte.init_num_monocytes = self.config.getint('init_num_monocytes')  # units: count

        monocyte.recruitment_rate = self.config.getfloat('recruitment_rate')
        monocyte.rec_bias = self.config.getfloat('rec_bias')
        monocyte.drift_bias = self.config.getfloat('drift_bias')

        monocyte.ma_move_rate_act = self.config.getfloat('ma_move_rate_act')  # µm/min
        monocyte.ma_move_rate_rest = self.config.getfloat('ma_move_rate_rest')  # µm/min

        monocyte.half_life = self.config.getfloat('ma_half_life')  # units: hours

        # UNUSED:

        # computed values
        monocyte.iter_to_rest = int(
            monocyte.time_to_rest / self.time_step
        )  # units: min / (min/step) = steps
        monocyte.iter_to_change_state = int(
            monocyte.time_to_change_state * (60 / time_step_size)
        )  # units: hours * (min/hour) / (min/step) = step

        monocyte.prob_death_per_timestep = -math.log(0.5) / (
            monocyte.half_life * (60 / time_step_size)
        )  # units: 1/(  hours * (min/hour) / (min/step)  ) = 1/step

        # initialize cells, placing them randomly
        locations = list(zip(*np.where(lung_tissue != TissueType.AIR)))
        dz_field: np.ndarray = state.grid.delta(axis=0)
        dy_field: np.ndarray = state.grid.delta(axis=1)
        dx_field: np.ndarray = state.grid.delta(axis=2)
        for vox_z, vox_y, vox_x in random.choices(locations, k=monocyte.init_num_monocytes):
            # the x,y,z coordinates are in the centers of the grids
            z = state.grid.z[vox_z]
            y = state.grid.y[vox_y]
            x = state.grid.x[vox_x]
            dz = dz_field[vox_z, vox_y, vox_x]
            dy = dy_field[vox_z, vox_y, vox_x]
            dx = dx_field[vox_z, vox_y, vox_x]
            self.create_monocyte(
                state=state,
                x=x + rg.uniform(-dx / 2, dx / 2),
                y=y + rg.uniform(-dy / 2, dy / 2),
                z=z + rg.uniform(-dz / 2, dz / 2),
                iron_pool=monocyte.ma_internal_iron,
            )

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        monocyte: MonocyteState = state.monocyte

        for monocyte_cell_index in monocyte.cells.alive():
            monocyte_cell = monocyte.cells[monocyte_cell_index]

            num_cells_in_phagosome = np.sum(monocyte_cell['phagosome'] >= 0)

            self.update_status(state, monocyte_cell, num_cells_in_phagosome)

            if (
                num_cells_in_phagosome == 0
                and rg.uniform() < monocyte.prob_death_per_timestep
                and len(monocyte.cells.alive()) > monocyte.min_ma
            ):
                monocyte_cell['status'] = PhagocyteStatus.DEAD
                monocyte_cell['dead'] = True

            if not monocyte_cell['fpn']:
                if monocyte_cell['fpn_iteration'] >= monocyte.iter_to_change_state:
                    monocyte_cell['fpn_iteration'] = 0
                    monocyte_cell['fpn'] = True
                else:
                    monocyte_cell['fpn_iteration'] += 1

            # Movement
            if monocyte_cell['status'] == PhagocyteStatus.ACTIVE:
                max_move_step = (
                    monocyte.ma_move_rate_act * self.time_step
                )  # (µm/min) * (min/step) = µm * step
            else:
                max_move_step = (
                    monocyte.ma_move_rate_rest * self.time_step
                )  # (µm/min) * (min/step) = µm * step
            move_step: int = rg.poisson(max_move_step)
            # move the cell 1 µm, move_step number of times
            for _ in range(move_step):
                self.single_step_move(
                    state, monocyte_cell, monocyte_cell_index, monocyte.cells
                )

        # Recruitment
        self.recruit_monocytes(state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        monocyte: MonocyteState = state.monocyte
        live_monocytes = monocyte.cells.alive()

        max_index = max(map(int, PhagocyteStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    monocyte.cells[monocyte_cell_index]['status']
                    for monocyte_cell_index in live_monocytes
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        tnfa_active = int(
            np.sum(
                np.fromiter(
                    (
                        monocyte.cells[monocyte_cell_index]['tnfa']
                        for monocyte_cell_index in live_monocytes
                    ),
                    dtype=bool,
                )
            )
        )

        return {
            'count': len(live_monocytes),
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
        return 'cells', state.monocyte.cells

    def recruit_monocytes(self, state: State) -> None:
        """
        Recruit monocytes based on MCP1 activation

        Parameters
        ----------
        state : State
            global simulation state

        Returns
        -------
        nothing
        """
        from nlisim.modules.mcp1 import MCP1State
        from nlisim.util import TissueType, activation_function

        monocyte: MonocyteState = state.monocyte
        mcp1: MCP1State = state.mcp1
        voxel_volume: float = state.voxel_volume
        space_volume: float = state.space_volume
        lung_tissue = state.lung_tissue

        # 1. compute number of monocytes to recruit
        num_live_monocytes = len(monocyte.cells.alive())
        avg = (
            monocyte.recruitment_rate
            * np.sum(mcp1.grid)
            * (1 - num_live_monocytes / monocyte.max_ma)
            / (mcp1.k_d * space_volume)
        )
        number_to_recruit = max(
            np.random.poisson(avg) if avg > 0 else 0, monocyte.min_ma - num_live_monocytes
        )
        # 2. get voxels for new monocytes, based on activation
        if number_to_recruit > 0:
            activation_voxels = zip(
                *np.where(
                    np.logical_and(
                        activation_function(
                            x=mcp1.grid,
                            k_d=mcp1.k_d,
                            h=self.time_step / 60,
                            volume=voxel_volume,
                            b=monocyte.rec_bias,
                        )
                        < rg.uniform(size=mcp1.grid.shape),
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
                self.create_monocyte(
                    state=state,
                    x=x + rg.uniform(-dx / 2, dx / 2),
                    y=y + rg.uniform(-dy / 2, dy / 2),
                    z=z + rg.uniform(-dz / 2, dz / 2),
                )

    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, voxel: Voxel
    ) -> Point:
        """
        Calculate a 1µm movement of a monocyte

        Parameters
        ----------
        state : State
            global simulation state
        cell : MonocyteCellData
            a monocyte cell
        voxel : Voxel
            current voxel position of the monocyte

        Returns
        -------
        Point
            the new position of the monocyte
        """
        # monocytees are attracted by MCP1
        from nlisim.modules.mcp1 import MCP1State
        from nlisim.util import TissueType, activation_function

        monocyte: MonocyteState = state.monocyte
        mcp1: MCP1State = state.mcp1
        grid: RectangularGrid = state.grid
        lung_tissue: np.ndarray = state.lung_tissue
        voxel_volume: float = state.voxel_volume

        # compute chemokine influence on velocity, with some randomness.
        # monocyte has a non-zero probability of moving into non-air voxels.
        # if not any of these, stay in place. This could happen if e.g. you are
        # somehow stranded in air.
        nearby_voxels: Tuple[Voxel, ...] = tuple(grid.get_adjacent_voxels(voxel, corners=True))
        weights = np.array(
            [
                0.0
                if lung_tissue[tuple(vxl)] == TissueType.AIR
                else activation_function(
                    x=mcp1.grid[tuple(vxl)],
                    k_d=mcp1.k_d,
                    h=self.time_step / 60,  # units: (min/step) / (min/hour)
                    volume=voxel_volume,
                    b=1,
                )
                + monocyte.drift_bias
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
    def monocyte(*, state: State, x: float, y: float, z: float, **kwargs) -> None:
        """
        Create a new monocyte cell

        Parameters
        ----------
        state : State
            global simulation state
        x : float
        y : float
        z : float
            coordinates of created monocyte
        kwargs
            parameters for monocyte, will give

        Returns
        -------
        nothing
        """
        monocyte: MonocyteState = state.monocyte

        # use default value of iron pool if not present
        iron_pool = kwargs.get('iron_pool', monocyte.ma_internal_iron)
        kwargs.pop('iron_pool', None)

        monocyte.cells.append(
            MonocyteCellData.create_cell(
                point=Point(x=x, y=y, z=z),
                iron_pool=iron_pool,
                **kwargs,
            )
        )

    def update_status(
        self, state: State, monocyte_cell: MonocyteCellData, num_cells_in_phagosome
    ) -> None:
        """
        Update the status of the cell, progressing between states after a certain number of ticks.

        Parameters
        ----------
        state : State
            global simulation state
        monocyte_cell : MonocyteCellData
        num_cells_in_phagosome

        Returns
        -------
        nothing
        """
        monocyte: MonocyteState = state.monocyte

        if monocyte_cell['status'] == PhagocyteStatus.NECROTIC:
            # TODO: what about APOPTOTIC?
            self.release_phagosome(state, monocyte_cell)

        elif num_cells_in_phagosome > monocyte.max_conidia:
            # TODO: how do we get here?
            monocyte_cell['status'] = PhagocyteStatus.NECROTIC

        elif monocyte_cell['status'] == PhagocyteStatus.ACTIVE:
            if monocyte_cell['status_iteration'] >= monocytee.iter_to_rest:
                monocyte_cell['status_iteration'] = 0
                monocyte_cell['tnfa'] = False
                monocyte_cell['status'] = PhagocyteStatus.RESTING
            else:
                monocyte_cell['status_iteration'] += 1

        elif monocyte_cell['status'] == PhagocyteStatus.INACTIVE:
            if monocyte_cell['status_iteration'] >= monocyte.iter_to_change_state:
                monocyte_cell['status_iteration'] = 0
                monocyte_cell['status'] = PhagocyteStatus.RESTING
            else:
                monocyte_cell['status_iteration'] += 1

        elif monocyte_cell['status'] == PhagocyteStatus.ACTIVATING:
            if monocyte_cell['status_iteration'] >= monocyte.iter_to_change_state:
                monocyte_cell['status_iteration'] = 0
                monocyte_cell['status'] = PhagocyteStatus.ACTIVE
            else:
                monocyte_cell['status_iteration'] += 1

        elif monocyte_cell['status'] == PhagocyteStatus.INACTIVATING:
            if monocyte_cell['status_iteration'] >= monocyte.iter_to_change_state:
                monocyte_cell['status_iteration'] = 0
                monocyte_cell['status'] = PhagocyteStatus.INACTIVE
            else:
                monocyte_cell['status_iteration'] += 1

        elif monocyte_cell['status'] == PhagocyteStatus.ANERGIC:
            if monocyte_cell['status_iteration'] >= monocyte.iter_to_change_state:
                monocyte_cell['status_iteration'] = 0
                monocyte_cell['status'] = PhagocyteStatus.RESTING
            else:
                monocyte_cell['status_iteration'] += 1
