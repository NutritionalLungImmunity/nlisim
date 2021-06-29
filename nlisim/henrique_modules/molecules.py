from functools import reduce
from itertools import product
import math
from operator import mul

from attr import attrs
import numpy as np

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attrs(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    turnover_rate: float
    cyt_bind_t: float
    rel_cyt_bind_unit_t: float
    turnover_rate: float
    diffusion_constant_timestep: float
    implicit_euler_matrix: np.ndarray


class Molecules(ModuleModel):
    name = 'molecules'
    StateClass = MoleculesState

    # noinspection SpellCheckingInspection
    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules

        molecules.cyt_bind_t = self.config.getfloat('cyt_bind_t')
        molecules.turnover_rate = self.config.getfloat('turnover_rate')

        # TODO: move these to individual molecules? otherwise changes in the time step will be off
        # Computed values
        molecules.rel_cyt_bind_unit_t = self.time_step / molecules.cyt_bind_t
        # TODO: original comments as below. Is the param 0.2?
        #  i.e. ...math.log(1+0.2)... Yes, 20% per hour
        # 0.2 # 10.1124/jpet.118.250134 (approx) 0.2/h CHANGE!!!!
        molecules.turnover_rate = 1 - math.log(1.2) / int(30 / 2.0)  # TODO: hard coded the 2.0 ...
        # TODO: is this a 2 hour constant? i.e. 4*30 min
        molecules.diffusion_constant_timestep = (
            self.config.getfloat('diffusion_constant') * self.time_step / (4 * 30)
        )

        # construct the laplacian
        # TODO: this is so wasteful of memory and should be replaced. Crank-Nicholson?
        grid_cardinality = reduce(mul, state.grid.shape)
        tissue = state.geometry.lung_tissue
        from nlisim.util import TissueType

        laplacian = np.zeros(shape=(grid_cardinality, grid_cardinality), dtype=float)
        for z, y, x in product(*map(range, state.grid.shape)):
            # ignore air voxels
            if tissue[z, y, x] == TissueType.AIR.value:
                continue
            # collect connections to non-air voxels (toric boundary)
            base_idx = np.ravel_multi_index((z, y, x), state.grid.shape)
            if tissue[z - 1, y, x] != TissueType.AIR.value:
                offset_idx = np.ravel_multi_index(
                    ((z - 1) % state.grid.shape[0], y, x), state.grid.shape
                )
                laplacian[offset_idx, base_idx] += 1.0
                laplacian[base_idx, base_idx] -= 1.0
            if tissue[z + 1, y, x] != TissueType.AIR.value:
                offset_idx = np.ravel_multi_index(
                    ((z + 1) % state.grid.shape[0], y, x), state.grid.shape
                )
                laplacian[offset_idx, base_idx] += 1.0
                laplacian[base_idx, base_idx] -= 1.0

            if tissue[z, y - 1, x] != TissueType.AIR.value:
                offset_idx = np.ravel_multi_index(
                    (z, (y - 1) % state.grid.shape[1], x), state.grid.shape
                )
                laplacian[offset_idx, base_idx] += 1.0
                laplacian[base_idx, base_idx] -= 1.0
            if tissue[z, y + 1, x] != TissueType.AIR.value:
                offset_idx = np.ravel_multi_index(
                    (z, (y + 1) % state.grid.shape[1], x), state.grid.shape
                )
                laplacian[offset_idx, base_idx] += 1.0
                laplacian[base_idx, base_idx] -= 1.0

            if tissue[z, y, x - 1] != TissueType.AIR.value:
                offset_idx = np.ravel_multi_index(
                    (z, y, (x - 1) % state.grid.shape[2]), state.grid.shape
                )
                laplacian[offset_idx, base_idx] += 1.0
                laplacian[base_idx, base_idx] -= 1.0
            if tissue[z, y, x + 1] != TissueType.AIR.value:
                offset_idx = np.ravel_multi_index(
                    (z, y, (x + 1) % state.grid.shape[2]), state.grid.shape
                )
                laplacian[offset_idx, base_idx] += 1.0
                laplacian[base_idx, base_idx] -= 1.0
        molecules.implicit_euler_matrix = (
            np.eye(grid_cardinality) - molecules.diffusion_constant_timestep * laplacian
        )

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        return state


class MoleculeModel(ModuleModel):
    @staticmethod
    def diffuse(grid: np.ndarray, state: State):
        # implicit euler
        shape = grid.shape
        np.copyto(
            grid,
            np.linalg.solve(state.molecules.implicit_euler_matrix, grid.reshape(-1)).reshape(shape),
        )
