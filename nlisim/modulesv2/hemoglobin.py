import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'HemoglobinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class HemoglobinState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    uptake_rate: float
    ma_heme_import_rate: float


class Hemoglobin(MoleculeModel):
    """Hemoglobin"""

    name = 'hemoglobin'
    StateClass = HemoglobinState

    def initialize(self, state: State) -> State:
        hemoglobin: HemoglobinState = state.hemoglobin
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        hemoglobin.uptake_rate = self.config.getfloat('uptake_rate')
        hemoglobin.ma_heme_import_rate = self.config.getfloat('ma_heme_import_rate')

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        hemoglobin: HemoglobinState = state.hemoglobin
        molecules: MoleculesState = state.molecules

        # TODO: Commented by Henrique, verify
        # elif itype is Macrophage:
        #     # v = Constants.MA_HEME_IMPORT_RATE * self.values[0]
        #     # self.dec(v)
        #     # interactable.inc_iron_pool(4*v)
        #     # return True
        #     return False

        # TODO: move to cell
        # elif itype is Afumigatus:
        #     if (
        #             interactable.status == Afumigatus.HYPHAE or interactable.status == Afumigatus.GERM_TUBE):  # and interactable.boolean_network[Afumigatus.LIP] == 1:
        #         v = Constants.HEMOGLOBIN_UPTAKE_RATE * self.values[0]
        #         self.decrease(v)
        #         interactable.inc_iron_pool(4 * v)
        #     return True

        # Degrade Hemoglobin
        hemoglobin.grid *= self.turnover_rate(x_mol=hemoglobin.grid,
                                              x_system_mol=0.0,
                                              turnover_rate=molecules.turnover_rate,
                                              rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
