import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'HemolysinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class HemolysinState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    hemolysin_qtty: float


class Hemolysin(MoleculeModel):
    """Hemolysin"""

    name = 'hemolysin'
    StateClass = HemolysinState

    def initialize(self, state: State) -> State:
        hemolysin: HemolysinState = state.hemolysin
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        hemolysin.hemolysin_qtty = self.config.getfloat('hemolysin_qtty')
        # constant from setting rate of secretion rate to 1

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        hemolysin: HemolysinState = state.hemolysin
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Afumigatus:
        #     if interactable.status == Afumigatus.HYPHAE:
        #         self.inc(Constants.HEMOLYSIN_QTTY)
        #     return True

        # Degrade Hemolysin
        hemolysin.grid *= self.turnover_rate(x_mol=hemolysin.grid,
                                             x_system_mol=0.0,
                                             turnover_rate=molecules.turnover_rate,
                                             rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
