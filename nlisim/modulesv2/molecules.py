import math

from attr import attrs
import numpy as np
import scipy.ndimage

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attrs(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    turnover_rate: float
    cyt_bind_t: float
    rel_cyt_bind_unit_t: float
    turnover_rate: float
    diffusion_constant_timestep: float


class Molecules(ModuleModel):
    name = 'molecules'
    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules

        molecules.cyt_bind_t = self.config.getfloat('cyt_bind_t')
        molecules.turnover_rate = self.config.getfloat('turnover_rate')

        # TODO: move these to individual molecules? otherwise changes in the time step will be off
        # Computed values
        molecules.rel_cyt_bind_unit_t = self.time_step / molecules.cyt_bind_t
        # TODO: original comments as below. Is the param 0.2? i.e. ...math.log(1+0.2)... Yes, 20% per hour
        # 0.2 # 10.1124/jpet.118.250134 (approx) 0.2/h CHANGE!!!!
        molecules.turnover_rate = 1 - math.log(1.2) / int(30 / 2.0)  # TODO: hard coded the 2.0 ...
        # TODO: is this a 2 hour constant? i.e. 4*30 min
        molecules.diffusion_constant_timestep = self.config.getfloat('diffusion_constant') * \
                                                self.time_step / (4 * 30)

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        return state


class MoleculeModel(ModuleModel):

    @staticmethod
    def diffuse(grid: np.ndarray, diffusion_constant: float):
        # TODO: verify
        grid += diffusion_constant * scipy.ndimage.laplace(grid)

    @staticmethod
    def michaelian_kinetics(*,
                            substrate: np.ndarray,
                            enzyme: np.ndarray,
                            km: float,
                            h: float,
                            k_cat: float = 1.0,
                            voxel_volume: float) -> np.ndarray:
        # Note: was originally h*k_cat*enzyme*substrate/(substrate+km), but with
        # enzyme /= voxel_volume and substrate /= voxel_volume.
        # This is algebraically equivalent and reduces the number of operations.
        return h * k_cat * enzyme * substrate / (substrate + km * voxel_volume)
