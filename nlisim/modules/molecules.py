from attr import attrs
import numpy as np
from scipy.sparse import csr_matrix

from nlisim.diffusion import periodic_discrete_laplacian
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attrs(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    turnover_rate: float  # units: proportion
    cyt_bind_t: float  # units: min
    rel_cyt_bind_unit_t: float
    diffusion_constant: float  # units: µm^2 * min^-1
    laplacian: csr_matrix  # units: µm^-2


class Molecules(ModuleModel):
    name = 'molecules'
    StateClass = MoleculesState

    # noinspection SpellCheckingInspection
    def initialize(self, state: State):
        from nlisim.util import TissueType

        molecules: MoleculesState = state.molecules

        molecules.cyt_bind_t = self.config.getfloat('cyt_bind_t')  # units: min
        # molecules.turnover_rate = self.config.getfloat('turnover_rate') # units: hours^-1
        molecules.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2 * min^-1

        # NOTE: there is an implicit assumption, below, that all molecules are on the
        #  same time step and that time step is 2 minutes. This is true on Sept 13, 2021.

        # Computed values
        molecules.rel_cyt_bind_unit_t = (
            self.time_step / molecules.cyt_bind_t
        )  # units: (min/step) / (min) = 1/step
        molecules.turnover_rate = 1 - np.log(1.2) / int(
            30 / 2
        )  # TODO: still hardcoding the 2, move to individual molecules?

        # construct the laplacian
        molecules.laplacian = periodic_discrete_laplacian(
            grid=state.grid, mask=state.lung_tissue != TissueType.AIR
        )  # units: µm^-2

        return state
