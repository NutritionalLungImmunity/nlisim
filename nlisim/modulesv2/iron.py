import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'IronState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IronState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))

class Iron(MoleculeModel):
    """Iron"""

    name = 'iron'
    StateClass = IronState

    def initialize(self, state: State) -> State:
        iron: IronState = state.iron
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values

        # computed values

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Macrophage:
        #     if interactable.status == Macrophage.NECROTIC or interactable.status == Macrophage.APOPTOTIC or interactable.status == Macrophage.DEAD:
        #         self.inc(interactable.iron_pool, "Iron")
        #         interactable.inc_iron_pool(-interactable.iron_pool)
        #     return True

        # Degrade Iron
        # *no operation* (turnover done by liver, if at all)

        return state
