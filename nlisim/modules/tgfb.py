import math
from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel, MoleculesState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, turnover_rate


def molecule_grid_factory(self: 'TGFBState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class TGFBState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    k_d: float


class TGFB(MoleculeModel):
    """TGFB"""

    name = 'tgfb'
    StateClass = TGFBState

    def initialize(self, state: State) -> State:
        tgfb: TGFBState = state.tgfb

        # config file values
        tgfb.half_life = self.config.getfloat('half_life')
        tgfb.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        tgfb.k_d = self.config.getfloat('k_d')

        # computed values
        tgfb.half_life_multiplier = 1 + math.log(0.5) / (tgfb.half_life / self.time_step)
        # time unit conversions
        tgfb.macrophage_secretion_rate_unit_t = tgfb.macrophage_secretion_rate * 60 * self.time_step

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteStatus

        tgfb: TGFBState = state.tgfb
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        voxel_volume: float = state.voxel_volume
        grid: RectangularGrid = state.grid

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            if macrophage_cell['status'] == PhagocyteStatus.INACTIVE:
                tgfb.grid[tuple(macrophage_cell_voxel)] += tgfb.macrophage_secretion_rate_unit_t
                if (
                    activation_function(
                        x=tgfb.grid[tuple(macrophage_cell_voxel)],
                        kd=tgfb.k_d,
                        h=self.time_step / 60,
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    macrophage_cell['status_iteration'] = 0

            elif macrophage_cell['status'] not in {
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.DEAD,
            }:
                if (
                    activation_function(
                        x=tgfb.grid[tuple(macrophage_cell_voxel)],
                        kd=tgfb.k_d,
                        h=self.time_step / 60,
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    macrophage_cell['status'] = PhagocyteStatus.INACTIVATING
                    macrophage_cell[
                        'status_iteration'
                    ] = 0  # Previously, was no reset of the status iteration

        # Degrade TGFB
        tgfb.grid *= tgfb.half_life_multiplier
        tgfb.grid *= turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of TGFB
        self.diffuse(tgfb.grid, state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tgfb: TGFBState = state.tgfb
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(tgfb.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        tgfb: TGFBState = state.tgfb
        return 'molecule', tgfb.grid
