from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel, MoleculesState
from nlisim.state import State
from nlisim.util import EPSILON, michaelian_kinetics, turnover_rate


def molecule_grid_factory(self: 'TAFCState') -> np.ndarray:
    # note the expansion to another axis to account for 0, 1, or 2 bound Fe's.
    return np.zeros(
        shape=self.global_state.grid.shape, dtype=[('TAFC', np.float64), ('TAFCBI', np.float64)]
    )


@attr.s(kw_only=True, repr=False)
class TAFCState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    k_m_tf_tafc: float
    tafc_up: float
    threshold: float
    tafc_qtty: float


class TAFC(MoleculeModel):
    # noinspection SpellCheckingInspection
    """TAFC: (T)ri(A)cetyl(F)usarinine C"""

    name = 'tafc'
    StateClass = TAFCState

    def initialize(self, state: State) -> State:
        tafc: TAFCState = state.tafc
        voxel_volume: float = state.voxel_volume

        # config file values
        tafc.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')

        # computed values
        tafc.tafc_qtty = self.config.getfloat('tafc_qtty') * 15  # TODO: unit_t
        tafc.tafc_up = self.config.getfloat('tafc_up') / voxel_volume / 15
        tafc.threshold = tafc.k_m_tf_tafc * voxel_volume / 1.0e6

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
        voxel_volume: float = state.voxel_volume

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to TAFC
        dfe2dt = michaelian_kinetics(
            substrate=transferrin.grid["TfFe2"],
            enzyme=tafc.grid["TAFC"],
            km=tafc.k_m_tf_tafc,
            h=self.time_step / 60,
            k_cat=1.0,  # default
            voxel_volume=voxel_volume,
        )
        dfedt = michaelian_kinetics(
            substrate=transferrin.grid["TfFe"],
            enzyme=tafc.grid["TAFC"],
            km=tafc.k_m_tf_tafc,
            h=self.time_step / 60,
            k_cat=1.0,  # default
            voxel_volume=voxel_volume,
        )

        # - enforce bounds from TAFC quantity
        total_change = dfe2dt + dfedt
        rel = tafc.grid['TAFC'] / (total_change + EPSILON)
        # enforce bounds and zero out problem divides
        rel[total_change == 0] = 0.0
        np.minimum(rel, 1.0, out=rel)
        np.maximum(rel, 0.0, out=rel)

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
                qtty = tafc.grid['TAFCBI'][tuple(afumigatus_cell_voxel)] * tafc.tafc_up
                # TODO: can't be bigger, unless tafc.tafc_up > 1. Am I missing something?
                # qtty = qtty if qtty < self.get("TAFCBI", x, y, z) else self.get("TAFCBI", x, y, z)
                tafc.grid['TAFCBI'][tuple(afumigatus_cell_voxel)] -= qtty
                afumigatus_cell['iron_pool'] += qtty

            # secrete TAFC
            if afumigatus_bool_net[NetworkSpecies.TAFC] and afumigatus_cell['status'] in {
                AfumigatusCellStatus.SWELLING_CONIDIA,
                AfumigatusCellStatus.HYPHAE,
                AfumigatusCellStatus.GERM_TUBE,
            }:
                tafc.grid['TAFC'][tuple(afumigatus_cell_voxel)] += tafc.tafc_qtty

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
        self.diffuse(tafc.grid['TAFC'], state)
        self.diffuse(tafc.grid['TAFCBI'], state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tafc: TAFCState = state.tafc
        voxel_volume = state.voxel_volume

        concentration_no_fe = np.mean(tafc.grid['TAFC']) / voxel_volume
        concentration_fe = np.mean(tafc.grid['TAFCBI']) / voxel_volume

        concentration = concentration_no_fe + concentration_fe

        return {
            'concentration any': float(concentration),
            'concentration TAFC': float(concentration_no_fe),
            'concentration TAFCBI': float(concentration_fe),
        }

    def visualization_data(self, state: State):
        tafc: TAFCState = state.tafc
        return 'molecule', tafc.grid
