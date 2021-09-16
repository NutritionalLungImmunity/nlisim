from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State
from nlisim.util import iron_tf_reaction


def molecule_grid_factory(self: 'TransferrinState') -> np.ndarray:
    return np.zeros(
        shape=self.global_state.grid.shape,
        dtype=[('Tf', np.float64), ('TfFe', np.float64), ('TfFe2', np.float64)],
    )


@attrs(kw_only=True, repr=False)
class TransferrinState(ModuleState):
    grid: np.ndarray = attrib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    k_m_tf_tafc: float  # units: aM
    p1: float
    p2: float
    p3: float
    threshold_log_hep: float  # units: log(atto-mols)
    default_apotf_rel_concentration: float  # units: proportion
    default_tffe_rel_concentration: float  # units: proportion
    default_tffe2_rel_concentration: float  # units: proportion
    default_tf_concentration: float  # units: aM
    default_apotf_concentration: float  # units: aM
    default_tffe_concentration: float  # units: aM
    default_tffe2_concentration: float  # units: aM
    tf_intercept: float  # units: aM
    tf_slope: float  # units: aM
    ma_iron_import_rate: float  # units: proportion * cell^-1 * step^-1
    ma_iron_import_rate_vol: float  # units: L * cell^-1 * h^-1
    ma_iron_export_rate: float  # units: proportion * cell^-1 * step^-1
    ma_iron_export_rate_vol: float  # units: L * cell^-1 * h^-1


class Transferrin(ModuleModel):
    """Transferrin"""

    name = 'transferrin'
    StateClass = TransferrinState

    def initialize(self, state: State) -> State:
        transferrin: TransferrinState = state.transferrin
        voxel_volume = state.voxel_volume

        # config file values
        transferrin.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')  # units: aM
        transferrin.p1 = self.config.getfloat('p1')
        transferrin.p2 = self.config.getfloat('p2')
        transferrin.p3 = self.config.getfloat('p3')
        transferrin.threshold_log_hep = self.config.getfloat('threshold_log_hep')

        transferrin.tf_intercept = self.config.getfloat('tf_intercept')
        transferrin.tf_slope = self.config.getfloat('tf_slope')

        transferrin.default_apotf_rel_concentration = self.config.getfloat(
            'default_apotf_rel_concentration'
        )  # units: proportion
        transferrin.default_tffe_rel_concentration = self.config.getfloat(
            'default_tffe_rel_concentration'
        )  # units: proportion
        transferrin.default_tffe2_rel_concentration = self.config.getfloat(
            'default_tffe2_rel_concentration'
        )  # units: proportion

        transferrin.ma_iron_import_rate_vol = self.config.getfloat('ma_iron_import_rate_vol')
        transferrin.ma_iron_export_rate_vol = self.config.getfloat('ma_iron_export_rate_vol')

        # computed values
        transferrin.default_tf_concentration = (
            transferrin.tf_intercept + transferrin.tf_slope * transferrin.threshold_log_hep
        )  # based on y--log(x) plot. units: aM * L = aM
        transferrin.default_apotf_concentration = (
            transferrin.default_apotf_rel_concentration * transferrin.default_tf_concentration
        )  # units: aM
        transferrin.default_tffe_concentration = (
            transferrin.default_tffe_rel_concentration * transferrin.default_tf_concentration
        )  # units: aM
        transferrin.default_tffe2_concentration = (
            transferrin.default_tffe2_rel_concentration * transferrin.default_tf_concentration
        )  # units: aM

        transferrin.ma_iron_import_rate = (
            transferrin.ma_iron_import_rate_vol / voxel_volume / (self.time_step / 60)
        )  # units: proportion * cell^-1 * step^-1
        transferrin.ma_iron_export_rate = (
            transferrin.ma_iron_export_rate_vol / voxel_volume / (self.time_step / 60)
        )  # units: proportion * cell^-1 * step^-1

        # initialize the molecular field
        transferrin.grid['Tf'] = transferrin.default_apotf_concentration * voxel_volume
        transferrin.grid['TfFe'] = transferrin.default_tffe_concentration * voxel_volume
        transferrin.grid['TfFe2'] = transferrin.default_tffe2_concentration * voxel_volume

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.molecules import MoleculesState
        from nlisim.modules.phagocyte import PhagocyteStatus

        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        macrophage: MacrophageState = state.macrophage
        molecules: MoleculesState = state.molecules
        grid: RectangularGrid = state.grid

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            uptake_proportion = np.minimum(transferrin.ma_iron_import_rate, 1.0)
            qtty_fe2 = transferrin.grid['TfFe2'][tuple(macrophage_cell_voxel)] * uptake_proportion
            qtty_fe = transferrin.grid['TfFe'][tuple(macrophage_cell_voxel)] * uptake_proportion

            # macrophage uptakes iron, leaves transferrin+0Fe behind
            transferrin.grid['TfFe2'][tuple(macrophage_cell_voxel)] -= qtty_fe2
            transferrin.grid['TfFe'][tuple(macrophage_cell_voxel)] -= qtty_fe
            transferrin.grid['Tf'][tuple(macrophage_cell_voxel)] += qtty_fe + qtty_fe2
            macrophage_cell['iron_pool'] += 2 * qtty_fe2 + qtty_fe

            if macrophage_cell['fpn'] and macrophage_cell['status'] not in {
                PhagocyteStatus.ACTIVE,
                PhagocyteStatus.ACTIVATING,
            }:
                # amount of iron to export is bounded by the amount of iron in the cell as well
                # as the amount which can be accepted by transferrin TODO: ask why not 2*Tf+TfFe?
                qtty: np.float64 = min(
                    macrophage_cell['iron_pool'],
                    2 * transferrin.grid['Tf'][tuple(macrophage_cell_voxel)],
                    macrophage_cell['iron_pool']
                    * transferrin.grid['Tf'][tuple(macrophage_cell_voxel)]
                    * transferrin.ma_iron_export_rate,
                )
                rel_tf_fe = iron_tf_reaction(
                    iron=qtty,
                    tf=transferrin.grid['Tf'][tuple(macrophage_cell_voxel)],
                    tf_fe=transferrin.grid['TfFe'][tuple(macrophage_cell_voxel)],
                    p1=transferrin.p1,
                    p2=transferrin.p2,
                    p3=transferrin.p3,
                )
                tffe_qtty = rel_tf_fe * qtty
                tffe2_qtty = (qtty - tffe_qtty) / 2

                transferrin.grid['Tf'][tuple(macrophage_cell_voxel)] -= tffe_qtty + tffe2_qtty
                transferrin.grid['TfFe'][tuple(macrophage_cell_voxel)] += tffe_qtty
                transferrin.grid['TfFe2'][tuple(macrophage_cell_voxel)] += tffe2_qtty
                macrophage_cell['iron_pool'] -= qtty

        # interaction with iron: transferrin -> transferrin+[1,2]Fe
        transferrin_fe_capacity = 2 * transferrin.grid['Tf'] + transferrin.grid['TfFe']
        potential_reactive_quantity = np.minimum(iron.grid, transferrin_fe_capacity)
        rel_tf_fe = iron_tf_reaction(
            iron=potential_reactive_quantity,
            tf=transferrin.grid["Tf"],
            tf_fe=transferrin.grid["TfFe"],
            p1=transferrin.p1,
            p2=transferrin.p2,
            p3=transferrin.p3,
        )
        tffe_qtty = rel_tf_fe * potential_reactive_quantity
        tffe2_qtty = (potential_reactive_quantity - tffe_qtty) / 2
        transferrin.grid['Tf'] -= tffe_qtty + tffe2_qtty
        transferrin.grid['TfFe'] += tffe_qtty
        transferrin.grid['TfFe2'] += tffe2_qtty
        iron.grid -= potential_reactive_quantity
        # Note: asked Henrique why there is no transferrin+Fe -> transferrin+2Fe reaction
        # answer was that this should already be accounted for

        # Degrade transferrin: done in liver

        # Diffusion of transferrin
        for component in {'Tf', 'TfFe', 'TfFe2'}:
            transferrin.grid[component][:] = apply_diffusion(
                variable=transferrin.grid[component],
                laplacian=molecules.laplacian,
                diffusivity=molecules.diffusion_constant,
                dt=self.time_step,
            )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        transferrin: TransferrinState = state.transferrin
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        concentration_0fe = np.mean(transferrin.grid['Tf'][mask]) / voxel_volume / 1e9
        concentration_1fe = np.mean(transferrin.grid['TfFe'][mask]) / voxel_volume / 1e9
        concentration_2fe = np.mean(transferrin.grid['TfFe2'][mask]) / voxel_volume / 1e9

        concentration = concentration_0fe + concentration_1fe + concentration_2fe

        return {
            'concentration (nM)': float(concentration),
            '+0Fe concentration (nM)': float(concentration_0fe),
            '+1Fe concentration (nM)': float(concentration_1fe),
            '+2Fe concentration (nM)': float(concentration_2fe),
        }

    def visualization_data(self, state: State):
        transferrin: TransferrinState = state.transferrin
        return 'molecule', transferrin.grid
