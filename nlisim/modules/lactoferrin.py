from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State
from nlisim.util import EPSILON, iron_tf_reaction, michaelian_kinetics, turnover_rate


def molecule_grid_factory(self: 'LactoferrinState') -> np.ndarray:
    return np.zeros(
        shape=self.global_state.grid.shape,
        dtype=[
            ('Lactoferrin', np.float64),
            ('LactoferrinFe', np.float64),
            ('LactoferrinFe2', np.float64),
        ],
    )


@attr.s(kw_only=True, repr=False)
class LactoferrinState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    k_m_tf_lac: float  # units: aM
    p1: float  # units: none
    p2: float  # units: none
    p3: float  # units: none
    ma_iron_import_rate: float
    iron_imp_exp_t: float
    rel_iron_imp_exp_unit_t: float
    neutrophil_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    threshold: float


class Lactoferrin(ModuleModel):
    """Lactoferrin"""

    name = 'lactoferrin'
    StateClass = LactoferrinState

    def initialize(self, state: State) -> State:
        lactoferrin: LactoferrinState = state.lactoferrin
        voxel_volume: float = state.voxel_volume  # units: L

        # config file values
        lactoferrin.k_m_tf_lac = self.config.getfloat('k_m_tf_lac')  # units: aM
        lactoferrin.p1 = self.config.getfloat('p1')
        lactoferrin.p2 = self.config.getfloat('p2')
        lactoferrin.p3 = self.config.getfloat('p3')
        lactoferrin.iron_imp_exp_t = self.config.getfloat('iron_imp_exp_t')

        # computed values
        lactoferrin.ma_iron_import_rate = self.config.getfloat('ma_iron_import_rate') / voxel_volume
        lactoferrin.threshold = lactoferrin.k_m_tf_lac * voxel_volume / 1.0e6
        lactoferrin.rel_iron_imp_exp_unit_t = self.time_step / lactoferrin.iron_imp_exp_t
        lactoferrin.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.molecules import MoleculesState
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteState, PhagocyteStatus
        from nlisim.modules.transferrin import TransferrinState

        lactoferrin: LactoferrinState = state.lactoferrin
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid
        voxel_volume = state.voxel_volume

        # macrophages uptake iron from lactoferrin
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            uptake_quantity = np.minimum(
                lactoferrin.ma_iron_import_rate * lactoferrin.rel_iron_imp_exp_unit_t, 1.0
            )

            qtty_fe2 = (
                lactoferrin.grid['LactoferrinFe2'][tuple(macrophage_cell_voxel)] * uptake_quantity
            )

            qtty_fe = (
                lactoferrin.grid['LactoferrinFe'][tuple(macrophage_cell_voxel)] * uptake_quantity
            )

            lactoferrin.grid['LactoferrinFe2'][tuple(macrophage_cell_voxel)] -= qtty_fe2
            lactoferrin.grid['LactoferrinFe'][tuple(macrophage_cell_voxel)] -= qtty_fe
            macrophage_cell['iron_pool'] += 2 * qtty_fe2 + qtty_fe

        # active and interacting neutrophils secrete lactoferrin
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]

            if (
                neutrophil_cell['status'] != PhagocyteStatus.ACTIVE
                or neutrophil_cell['state'] != PhagocyteState.INTERACTING
            ):
                continue

            neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])
            lactoferrin.grid['Lactoferrin'][
                tuple(neutrophil_cell_voxel)
            ] += lactoferrin.neutrophil_secretion_rate

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to lactoferrin
        dfe2_dt = michaelian_kinetics(
            substrate=transferrin.grid['TfFe2'],
            enzyme=lactoferrin.grid["Lactoferrin"],
            k_m=lactoferrin.k_m_tf_lac,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
            voxel_volume=voxel_volume,
        )
        dfe_dt = michaelian_kinetics(
            substrate=transferrin.grid['TfFe'],
            enzyme=lactoferrin.grid['Lactoferrin'],
            k_m=lactoferrin.k_m_tf_lac,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
            voxel_volume=voxel_volume,
        )
        # - enforce bounds from lactoferrin quantity
        dfex_dt = dfe2_dt + dfe_dt
        mask = dfex_dt > lactoferrin.grid['Lactoferrin']

        rel = lactoferrin.grid['Lactoferrin'] / (dfex_dt + EPSILON)
        # enforce bounds
        rel[dfex_dt == 0] = 0.0
        rel[:] = np.maximum(np.minimum(rel, 1.0), 0.0)

        dfe2_dt[mask] = (dfe2_dt * rel)[mask]
        dfe_dt[mask] = (dfe_dt * rel)[mask]

        # - calculate iron transfer from transferrin+[1,2]Fe to lactoferrin+Fe
        dfe2dt_fe = michaelian_kinetics(
            substrate=transferrin.grid['TfFe2'],
            enzyme=lactoferrin.grid['LactoferrinFe'],
            k_m=lactoferrin.k_m_tf_lac,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
            voxel_volume=voxel_volume,
        )
        dfedt_fe = michaelian_kinetics(
            substrate=transferrin.grid['TfFe'],
            enzyme=lactoferrin.grid['LactoferrinFe'],
            k_m=lactoferrin.k_m_tf_lac,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
            voxel_volume=voxel_volume,
        )
        # - enforce bounds from lactoferrin+Fe quantity
        dfexdt_fe = dfe2dt_fe + dfedt_fe
        mask = dfexdt_fe > lactoferrin.grid['LactoferrinFe']

        rel = lactoferrin.grid['LactoferrinFe'] / (dfe2dt_fe + dfedt_fe + EPSILON)
        # enforce bounds
        rel[dfexdt_fe == 0] = 0.0
        np.minimum(rel, 1.0, out=rel)
        np.maximum(rel, 0.0, out=rel)

        dfe2dt_fe[mask] = (dfe2dt_fe * rel)[mask]
        dfedt_fe[mask] = (dfedt_fe * rel)[mask]

        # transferrin+2Fe loses an iron, becomes transferrin+Fe
        transferrin.grid['TfFe2'] -= dfe2_dt + dfe2dt_fe
        transferrin.grid['TfFe'] += dfe2_dt + dfe2dt_fe

        # transferrin+Fe loses an iron, becomes transferrin
        transferrin.grid['TfFe'] -= dfe_dt + dfedt_fe
        transferrin.grid['Tf'] += dfe_dt + dfedt_fe

        # lactoferrin gains an iron, becomes lactoferrin+Fe
        lactoferrin.grid['Lactoferrin'] -= dfe2_dt + dfe_dt
        lactoferrin.grid['LactoferrinFe'] += dfe2_dt + dfe_dt

        # lactoferrin+Fe gains an iron, becomes lactoferrin+2Fe
        lactoferrin.grid['LactoferrinFe'] -= dfe2dt_fe + dfedt_fe
        lactoferrin.grid['LactoferrinFe2'] += dfe2dt_fe + dfedt_fe

        # interaction with iron
        lactoferrin_fe_capacity = (
            2 * lactoferrin.grid["Lactoferrin"] + lactoferrin.grid["LactoferrinFe"]
        )
        potential_reactive_quantity = np.minimum(iron.grid, lactoferrin_fe_capacity)
        rel_tf_fe = iron_tf_reaction(
            iron=potential_reactive_quantity,
            tf=lactoferrin.grid["Lactoferrin"],
            tf_fe=lactoferrin.grid["LactoferrinFe"],
            p1=lactoferrin.p1,
            p2=lactoferrin.p2,
            p3=lactoferrin.p3,
        )
        tffe_qtty = rel_tf_fe * potential_reactive_quantity
        tffe2_qtty = (potential_reactive_quantity - tffe_qtty) / 2
        lactoferrin.grid['Lactoferrin'] -= tffe_qtty + tffe2_qtty
        lactoferrin.grid['LactoferrinFe'] += tffe_qtty
        lactoferrin.grid['LactoferrinFe2'] += tffe2_qtty
        iron.grid -= potential_reactive_quantity

        # Degrade Lactoferrin
        trnvr_rt = turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        lactoferrin.grid['Lactoferrin'] *= trnvr_rt
        lactoferrin.grid['LactoferrinFe'] *= trnvr_rt
        lactoferrin.grid['LactoferrinFe2'] *= trnvr_rt

        # Diffusion of lactoferrin
        for component in {'Lactoferrin', 'LactoferrinFe', 'LactoferrinFe2'}:
            lactoferrin.grid[component][:] = apply_diffusion(
                variable=lactoferrin.grid[component],
                laplacian=molecules.laplacian,
                diffusivity=molecules.diffusion_constant,
                dt=self.time_step,
            )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        lactoferrin: LactoferrinState = state.lactoferrin
        voxel_volume = state.voxel_volume

        concentration_0fe = np.mean(lactoferrin.grid['Lactoferrin']) / voxel_volume / 1e9
        concentration_1fe = np.mean(lactoferrin.grid['LactoferrinFe']) / voxel_volume / 1e9
        concentration_2fe = np.mean(lactoferrin.grid['LactoferrinFe2']) / voxel_volume / 1e9

        concentration = concentration_0fe + concentration_1fe + concentration_2fe

        return {
            'concentration (nM)': float(concentration),
            '+0Fe concentration (nM)': float(concentration_0fe),
            '+1Fe concentration (nM)': float(concentration_1fe),
            '+2Fe concentration (nM)': float(concentration_2fe),
        }

    def visualization_data(self, state: State):
        lactoferrin: LactoferrinState = state.lactoferrin

        return 'molecule', lactoferrin.grid
