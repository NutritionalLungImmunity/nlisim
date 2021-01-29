import attr
from attr import attrib, attrs
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.afumigatus import AfumigatusState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.state import State


def molecule_grid_factory(self: 'HemoglobinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HemoglobinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
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
        afumigatus: AfumigatusState = state.afumigatus

        # interact with afumigatus


        # TODO: move to cell
        # elif itype is Afumigatus:
        #     if (
        #             interactable.status == Afumigatus.HYPHAE or interactable.status == Afumigatus.GERM_TUBE):
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
