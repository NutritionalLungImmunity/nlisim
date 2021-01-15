import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.iron import IronState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.transferrin import TransferrinState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'TAFCState') -> np.ndarray:
    # note the expansion to another axis to account for 0, 1, or 2 bound Fe's.
    return np.zeros(shape=self.global_state.grid.shape,
                    dtype=[('TAFC', np.float),
                           ('TAFCBI', np.float)])


@attr.s(kw_only=True, repr=False)
class TAFCState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    k_m_tf_tafc: float
    threshold: float


class TAFC(MoleculeModel):
    """TAFC: (T)ri(A)cetyl(F)usarinine C"""

    name = 'tafc'
    StateClass = TAFCState

    def initialize(self, state: State) -> State:
        tafc: TAFCState = state.tafc
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        tafc.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')

        # computed values
        tafc.threshold = tafc.k_m_tf_tafc * voxel_volume / 1.0e6

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        tafc: TAFCState = state.tafc
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to TAFC
        dfe2dt = self.michaelian_kinetics(substrate=transferrin.grid["TfFe2"],
                                          enzyme=tafc.grid["TAFC"],
                                          km=tafc.k_m_tf_tafc,
                                          h=state.simulation.time_step_size / 60,
                                          voxel_volume=voxel_volume)
        dfedt = self.michaelian_kinetics(substrate=transferrin.grid["TfFe"],
                                         enzyme=tafc.grid["TAFC"],
                                         km=tafc.k_m_tf_tafc,
                                         h=state.simulation.time_step_size / 60,
                                         voxel_volume=voxel_volume)

        # - enforce bounds from TAFC quantity
        rel = tafc.grid['TAFC'] / (dfe2dt + dfedt)
        np.nan_to_num(rel, nan=0.0)
        np.maximum(rel, 1.0, out=rel)
        dfe2dt = dfe2dt * rel
        dfedt = dfedt * rel

        # transferrin+2Fe loses an iron, becomes transferrin+Fe
        transferrin.grid['TfFe2'] -= dfe2dt
        transferrin.grid['TfFe'] += dfe2dt

        # transferrin+Fe loses an iron, becomes transferrin
        transferrin.grid['TfFe'] -= dfedt
        transferrin.grid['Tf'] += dfedt

        # iron from transferrin becomes bound to TAFC (TAFC->TAFCBI)
        tafc.grid['TAFC'] -= dfe2dt + dfedt
        tafc.grid['TAFCBI'] += dfe2dt + dfedt

        # TODO: move to cell
        # elif itype is Afumigatus:
        #     if interactable.state == Afumigatus.FREE and interactable.status != Afumigatus.DYING and interactable.status != Afumigatus.DEAD:
        #         if interactable.boolean_network[Afumigatus.MirB] == 1 and interactable.boolean_network[
        #             Afumigatus.EstB] == 1:
        #             qtty = self.get("TAFCBI") * Constants.TAFC_UP
        #             qtty = qtty if qtty < self.get("TAFCBI") else self.get("TAFCBI")
        #
        #             self.decrease(qtty, "TAFCBI")
        #             interactable.inc_iron_pool(qtty)
        #         if interactable.boolean_network[Afumigatus.TAFC] == 1 and \
        #                 (interactable.status == Afumigatus.SWELLING_CONIDIA or \
        #                  interactable.status == Afumigatus.HYPHAE or
        #                  interactable.status == Afumigatus.GERM_TUBE):  # SECRETE TAFC
        #             self.inc(Constants.TAFC_QTTY, "TAFC")
        #     return True


        # interaction with iron, all available iron is bound to TAFC
        potential_reactive_quantity = np.minimum(iron.grid, tafc.grid['TAFC'])
        tafc.grid['TAFC'] -= potential_reactive_quantity
        tafc.grid['TAFCBI'] += potential_reactive_quantity
        iron.grid -= potential_reactive_quantity

        # Degrade TAFC
        tafc.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float),
                                        x_system_mol=0.0,
                                        turnover_rate=molecules.turnover_rate,
                                        rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
