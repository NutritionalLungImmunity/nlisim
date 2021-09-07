from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import EPSILON, michaelian_kinetics, turnover_rate


def molecule_grid_factory(self: 'TAFCState') -> np.ndarray:
    # note the expansion to another axis to account for 0 or 1 bound Fe's.
    return np.zeros(
        shape=self.global_state.grid.shape, dtype=[('TAFC', np.float64), ('TAFCBI', np.float64)]
    )


@attr.s(kw_only=True, repr=False)
class TAFCState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    k_m_tf_tafc: float  # units: aM
    tafcbi_uptake_rate: float  # units: L * cell^-1 * h^-1
    tafcbi_uptake_rate_unit_t: float  # units: proportion * cell^-1 * step^-1
    afumigatus_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    afumigatus_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1


class TAFC(ModuleModel):
    # noinspection SpellCheckingInspection
    """TAFC: (T)ri(A)cetyl(F)usarinine C"""

    name = 'tafc'
    StateClass = TAFCState

    def initialize(self, state: State) -> State:
        tafc: TAFCState = state.tafc
        voxel_volume: float = state.voxel_volume

        # config file values
        tafc.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')  # units: aM
        tafc.afumigatus_secretion_rate = self.config.getfloat(
            'afumigatus_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        tafc.tafcbi_uptake_rate = self.config.getfloat(
            'tafcbi_uptake_rate'
        )  # units: L * cell^-1 * h^-1

        # computed values
        tafc.afumigatus_secretion_rate_unit_t = tafc.afumigatus_secretion_rate * (
            self.time_step / 60
        )  # units: (atto-mol * cell^-1 * h^-1) * (min/step) / (min/hour)
        tafc.tafcbi_uptake_rate_unit_t = (
            tafc.tafcbi_uptake_rate
            / voxel_volume
            * (self.time_step / 60)
            # units: (L * cell^-1 * h^-1) / L  * (min/step) / (min/hour)
            # = proportion * cell^-1 * step^-1
        )
        # TODO: this has to be unnecessary:
        tafc.tafcbi_uptake_rate_unit_t = min(1.0, tafc.tafcbi_uptake_rate_unit_t)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            AfumigatusCellData,
            AfumigatusCellState,
            AfumigatusCellStatus,
            AfumigatusState,
            NetworkSpecies,
        )
        from nlisim.modules.iron import IronState
        from nlisim.modules.transferrin import TransferrinState

        tafc: TAFCState = state.tafc
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        grid: RectangularGrid = state.grid
        voxel_volume: float = state.voxel_volume  # units: L

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to TAFC
        dfe2_dt = michaelian_kinetics(
            substrate=transferrin.grid["TfFe2"],
            enzyme=tafc.grid["TAFC"],
            k_m=tafc.k_m_tf_tafc,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,  # default
            voxel_volume=voxel_volume,
        )
        dfe_dt = michaelian_kinetics(
            substrate=transferrin.grid["TfFe"],
            enzyme=tafc.grid["TAFC"],
            k_m=tafc.k_m_tf_tafc,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,  # default
            voxel_volume=voxel_volume,
        )

        # - enforce bounds from TAFC quantity
        total_change = dfe2_dt + dfe_dt
        rel = tafc.grid['TAFC'] / (total_change + EPSILON)
        # enforce bounds and zero out problem divides
        rel[total_change == 0] = 0.0
        rel[:] = np.maximum(np.minimum(rel, 1.0), 0.0)

        dfe2_dt = dfe2_dt * rel
        dfe_dt = dfe_dt * rel

        # transferrin+2Fe loses an iron, becomes transferrin+Fe
        transferrin.grid['TfFe2'] -= dfe2_dt
        transferrin.grid['TfFe'] += dfe2_dt

        # transferrin+Fe loses an iron, becomes transferrin
        transferrin.grid['TfFe'] -= dfe_dt
        transferrin.grid['Tf'] += dfe_dt

        # iron from transferrin becomes bound to TAFC (TAFC->TAFCBI)
        tafc.grid['TAFC'] -= dfe2_dt + dfe_dt
        tafc.grid['TAFCBI'] += dfe2_dt + dfe_dt

        # interaction with iron, all available iron is bound to TAFC
        potential_reactive_quantity = np.minimum(iron.grid, tafc.grid['TAFC'])
        tafc.grid['TAFC'] -= potential_reactive_quantity
        tafc.grid['TAFCBI'] += potential_reactive_quantity
        iron.grid -= potential_reactive_quantity

        # interaction with fungus
        for afumigatus_cell_index in afumigatus.cells.alive():
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]

            if (
                afumigatus_cell['state'] != AfumigatusCellState.FREE
                or afumigatus_cell['status'] == AfumigatusCellStatus.DYING
            ):
                continue

            afumigatus_cell_voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])
            afumigatus_bool_net: np.ndarray = afumigatus_cell['boolean_network']

            # uptake iron from TAFCBI
            if afumigatus_bool_net[NetworkSpecies.MirB] & afumigatus_bool_net[NetworkSpecies.EstB]:
                quantity = (
                    tafc.grid['TAFCBI'][tuple(afumigatus_cell_voxel)]
                    * tafc.tafcbi_uptake_rate_unit_t
                )
                tafc.grid['TAFCBI'][tuple(afumigatus_cell_voxel)] -= quantity
                afumigatus_cell['iron_pool'] += quantity

            # secrete TAFC
            if afumigatus_bool_net[NetworkSpecies.TAFC] and afumigatus_cell['status'] in {
                AfumigatusCellStatus.SWELLING_CONIDIA,
                AfumigatusCellStatus.HYPHAE,
                AfumigatusCellStatus.GERM_TUBE,
            }:
                tafc.grid['TAFC'][
                    tuple(afumigatus_cell_voxel)
                ] += tafc.afumigatus_secretion_rate_unit_t

        # Degrade TAFC
        trnvr_rt = turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        tafc.grid['TAFC'] *= trnvr_rt
        tafc.grid['TAFCBI'] *= trnvr_rt

        # Diffusion of TAFC
        for component in {'TAFC', 'TAFCBI'}:
            tafc.grid[component][:] = apply_diffusion(
                variable=tafc.grid[component],
                laplacian=molecules.laplacian,
                diffusivity=molecules.diffusion_constant,
                dt=self.time_step,
            )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tafc: TAFCState = state.tafc
        voxel_volume = state.voxel_volume

        concentration_no_fe = np.mean(tafc.grid['TAFC']) / voxel_volume
        concentration_fe = np.mean(tafc.grid['TAFCBI']) / voxel_volume

        concentration = concentration_no_fe + concentration_fe

        return {
            'concentration (nM)': float(concentration / 1e9),
            'concentration TAFC (nM)': float(concentration_no_fe / 1e9),
            'concentration TAFCBI (nM)': float(concentration_fe / 1e9),
        }

    def visualization_data(self, state: State):
        tafc: TAFCState = state.tafc
        return 'molecule', tafc.grid
