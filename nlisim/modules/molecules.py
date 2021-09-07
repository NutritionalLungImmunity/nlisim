from attr import attrs
from scipy.sparse import csr_matrix

from nlisim.diffusion import periodic_discrete_laplacian
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attrs(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    turnover_rate: float  # units: hours^-1
    cyt_bind_t: float
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

        molecules.cyt_bind_t = self.config.getfloat('cyt_bind_t')
        molecules.turnover_rate = self.config.getfloat('turnover_rate')

        # NOTE: there is an implicit assumption, below, that all molecules are on the
        #  same time step. This is true on Aug 24, 2021.

        # Computed values
        molecules.rel_cyt_bind_unit_t = self.time_step / molecules.cyt_bind_t
        # TODO: original comments as below. Is the param 0.2?
        #  i.e. ...math.log(1+0.2)... Yes, 20% per hour
        # 0.2 # 10.1124/jpet.118.250134 (approx) 0.2/h CHANGE!!!!
        # molecules.turnover_rate = 1 - math.log(1.2) / int(30 / 2.0)  # TODO: hard coded the 2.0 ...
        molecules.turnover_rate = (1 - 0.2) ** (
            2 / 60
        )  # TODO: still hardcoding the 2, move to individual molecules?
        molecules.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2 * min^-1

        # construct the laplacian
        molecules.laplacian = periodic_discrete_laplacian(
            grid=state.grid, mask=state.lung_tissue != TissueType.AIR
        )  # units: µm^-2

        # Note: Henrique's code did things a little differently here:
        #  1) three 1D laplacians instead of one 3D laplacian
        #  2) four implicit steps instead of one

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        return state
