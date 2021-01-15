import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.iron import IronState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.transferrin import TransferrinState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'LactoferrinState') -> np.ndarray:
    # note the expansion to another axis to account for 0, 1, or 2 bound Fe's.
    return np.zeros(shape=self.global_state.grid.shape,
                    dtype=[('Lactoferrin', np.float),
                           ('LactoferrinFe', np.float),
                           ('LactoferrinFe2', np.float)])


@attr.s(kw_only=True, repr=False)
class LactoferrinState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    k_m_tf_lac: float
    p1: float
    p2: float
    p3: float


class Lactoferrin(MoleculeModel):
    """Lactoferrin"""

    name = 'lactoferrin'
    StateClass = LactoferrinState

    def initialize(self, state: State) -> State:
        lactoferrin: LactoferrinState = state.lactoferrin
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        lactoferrin.k_m_tf_lac = self.config.getfloat('k_m_tf_lac')
        lactoferrin.p1 = self.config.getfloat('p1')
        lactoferrin.p2 = self.config.getfloat('p2')
        lactoferrin.p3 = self.config.getfloat('p3')

        # computed values
        lactoferrin.threshold = lactoferrin.k_m_tf_lac * voxel_volume / 1.0e6

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        lactoferrin: LactoferrinState = state.lactoferrin
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # TODO: move to cell
        # elif itype is Macrophage:  # ADD UPTAKE
        #     qttyFe2 = self.get("LactoferrinFe2") * Constants.MA_IRON_IMPORT_RATE * Constants.REL_IRON_IMP_EXP_UNIT_T
        #     qttyFe = self.get("LactoferrinFe") * Constants.MA_IRON_IMPORT_RATE * Constants.REL_IRON_IMP_EXP_UNIT_T
        #
        #     qttyFe2 = qttyFe2 if qttyFe2 < self.get("LactoferrinFe2") else self.get("LactoferrinFe2")
        #     qttyFe = qttyFe if qttyFe < self.get("LactoferrinFe") else self.get("LactoferrinFe")
        #
        #     self.decrease(qttyFe2, "LactoferrinFe2")
        #     self.decrease(qttyFe, "LactoferrinFe")
        #     interactable.inc_iron_pool(2 * qttyFe2 + qttyFe)
        #     return True

        # TODO: move to cell
        # elif itype is Neutrophil:
        #     if interactable.status == Neutrophil.ACTIVE and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.LAC_QTTY, "Lactoferrin")
        #     return True

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to lactoferrin
        dfe2dt = self.michaelian_kinetics(substrate=transferrin.grid['TfFe2'],
                                          enzyme=lactoferrin.grid["Lactoferrin"],
                                          km=lactoferrin.k_m_tf_lac,
                                          h=state.simulation.time_step_size / 60,
                                          voxel_volume=voxel_volume)
        dfedt = self.michaelian_kinetics(substrate=transferrin.grid['TfFe'],
                                         enzyme=lactoferrin.grid['Lactoferrin'],
                                         km=lactoferrin.k_m_tf_lac,
                                         h=state.simulation.time_step_size / 60,
                                         voxel_volume=voxel_volume)
        # - enforce bounds from lactoferrin quantity
        mask = (dfe2dt + dfedt) > lactoferrin.grid['Lactoferrin']
        rel = lactoferrin.grid['Lactoferrin'] / (dfe2dt + dfedt)
        dfe2dt[mask] = (dfe2dt * rel)[mask]
        dfedt[mask] = (dfedt * rel)[mask]

        # - calculate iron transfer from transferrin+[1,2]Fe to lactoferrin+Fe
        dfe2dt_fe = self.michaelian_kinetics(substrate=transferrin.grid['TfFe2'],
                                             enzyme=lactoferrin.grid['LactoferrinFe'],
                                             km=lactoferrin.k_m_tf_lac,
                                             h=state.simulation.time_step_size / 60,
                                             voxel_volume=voxel_volume)
        dfedt_fe = self.michaelian_kinetics(substrate=transferrin.grid['TfFe'],
                                            enzyme=lactoferrin.grid['LactoferrinFe'],
                                            km=lactoferrin.k_m_tf_lac,
                                            h=state.simulation.time_step_size / 60,
                                            voxel_volume=voxel_volume)
        # - enforce bounds from lactoferrin+Fe quantity
        mask = (dfe2dt_fe + dfedt_fe) > lactoferrin.grid['LactoferrinFe']
        rel = lactoferrin.grid['LactoferrinFe'] / (dfe2dt_fe + dfedt_fe)
        dfe2dt_fe[mask] = (dfe2dt_fe * rel)[mask]
        dfedt_fe[mask] = (dfedt_fe * rel)[mask]

        # transferrin+2Fe loses an iron, becomes transferrin+Fe
        transferrin.grid['TfFe2'] -= dfe2dt + dfe2dt_fe
        transferrin.grid['TfFe'] += dfe2dt + dfe2dt_fe

        # transferrin+Fe loses an iron, becomes transferrin
        transferrin.grid['TfFe'] -= dfedt + dfedt_fe
        transferrin.grid['Tf'] += dfedt + dfedt_fe

        # lactoferrin gains an iron, becomes lactoferrin+Fe
        lactoferrin.grid['Lactoferrin'] -= dfe2dt + dfedt
        lactoferrin.grid['LactoferrinFe'] += dfe2dt + dfedt

        # lactoferrin+Fe gains an iron, becomes lactoferrin+2Fe
        lactoferrin.grid['LactoferrinFe'] -= dfe2dt_fe + dfedt_fe
        lactoferrin.grid['LactoferrinFe2'] += dfe2dt_fe + dfedt_fe

        # interaction with iron
        lactoferrin_fe_capacity = 2 * lactoferrin.grid["Lactoferrin"] + lactoferrin.grid["LactoferrinFe"]
        potential_reactive_quantity = np.minimum(iron.grid, lactoferrin_fe_capacity)
        rel_TfFe = self.iron_tf_reaction(potential_reactive_quantity,
                                         lactoferrin.grid["Lactoferrin"],
                                         lactoferrin.grid["LactoferrinFe"],
                                         p1=lactoferrin.p1,
                                         p2=lactoferrin.p2,
                                         p3=lactoferrin.p3)
        tffe_qtty = rel_TfFe * potential_reactive_quantity
        tffe2_qtty = (potential_reactive_quantity - tffe_qtty) / 2
        lactoferrin.grid['Lactoferrin'] -= tffe_qtty + tffe2_qtty
        lactoferrin.grid['LactoferrinFe'] += tffe_qtty
        lactoferrin.grid['LactoferrinFe2'] += tffe2_qtty
        iron.grid -= potential_reactive_quantity

        # Degrade Lactoferrin
        lactoferrin.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float),
                                               x_system_mol=0.0,
                                               turnover_rate=molecules.turnover_rate,
                                               rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state

    # TODO: duplicated with code in transferrin
    @staticmethod
    def iron_tf_reaction(iron: np.ndarray,
                         Tf: np.ndarray,
                         TfFe: np.ndarray,
                         p1: float,
                         p2: float,
                         p3: float) -> np.ndarray:
        total_binding_site = 2 * (Tf + TfFe)  # That is right 2*(Tf + TfFe)!
        total_iron = iron + TfFe  # it does not count TfFe2

        with np.seterr(divide='ignore'):
            rel_total_iron = total_iron / total_binding_site
            np.nan_to_num(rel_total_iron, nan=0.0, posinf=0.0, neginf=0.0)
            rel_total_iron = np.maximum(np.minimum(rel_total_iron, 1.0), 0.0)

        # rel_TfFe = p1 * rel_total_iron * rel_total_iron * rel_total_iron + \
        #            p2 * rel_total_iron * rel_total_iron + \
        #            p3 * rel_total_iron
        # this reduces the number of operations slightly:
        rel_TfFe = ((p1 * rel_total_iron + p2) * rel_total_iron + p3) * rel_total_iron

        np.maximum(0.0, rel_TfFe, out=rel_TfFe)  # one root of the polynomial is at ~0.99897 and goes neg after
        # rel_TfFe = np.minimum(1.0, rel_TfFe) <- not currently needed, future-proof?
        rel_TfFe[total_iron == 0] = 0.0
        rel_TfFe[total_binding_site == 0] = 0.0
        return rel_TfFe
