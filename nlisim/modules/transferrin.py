import math
from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel
from nlisim.state import State
from nlisim.util import iron_tf_reaction


def molecule_grid_factory(self: 'TransferrinState') -> np.ndarray:
    return np.zeros(
        shape=self.global_state.grid.shape,
        dtype=[('Tf', np.float64), ('TfFe', np.float64), ('TfFe2', np.float64)],
    )


@attrs(kw_only=True, repr=False)
class TransferrinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    k_m_tf_tafc: float
    threshold: float
    p1: float
    p2: float
    p3: float
    threshold_log_hep: float
    threshold_hep: float
    default_apotf_rel_concentration: float
    default_tffe_rel_concentration: float
    default_tffe2_rel_concentration: float
    default_tf_concentration: float
    default_apotf_concentration: float
    default_tffe_concentration: float
    default_tffe2_concentration: float
    tf_intercept: float
    tf_slope: float
    iron_imp_exp_t: float
    ma_iron_import_rate: float
    ma_iron_export_rate: float
    rel_iron_imp_exp_unit_t: float


class Transferrin(MoleculeModel):
    """Transferrin"""

    name = 'transferrin'
    StateClass = TransferrinState

    def initialize(self, state: State) -> State:
        transferrin: TransferrinState = state.transferrin
        voxel_volume = state.voxel_volume

        # config file values
        transferrin.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')
        transferrin.p1 = self.config.getfloat('p1')
        transferrin.p2 = self.config.getfloat('p2')
        transferrin.p3 = self.config.getfloat('p3')
        transferrin.threshold_log_hep = self.config.getfloat('threshold_log_hep')

        transferrin.tf_intercept = self.config.getfloat('tf_intercept')
        transferrin.tf_slope = self.config.getfloat('tf_slope')

        transferrin.default_apotf_rel_concentration = self.config.getfloat(
            'default_apotf_rel_concentration'
        )
        transferrin.default_tffe_rel_concentration = self.config.getfloat(
            'default_tffe_rel_concentration'
        )
        transferrin.default_tffe2_rel_concentration = self.config.getfloat(
            'default_tffe2_rel_concentration'
        )

        transferrin.iron_imp_exp_t = self.config.getfloat('iron_imp_exp_t')

        # computed values
        transferrin.threshold = transferrin.k_m_tf_tafc * voxel_volume / 1.0e6
        transferrin.threshold_hep = math.pow(10, transferrin.threshold_log_hep)
        transferrin.default_tf_concentration = (
            transferrin.tf_intercept + transferrin.tf_slope * transferrin.threshold_log_hep
        ) * voxel_volume
        transferrin.default_apotf_concentration = (
            transferrin.default_apotf_rel_concentration * transferrin.default_tf_concentration
        )
        transferrin.default_tffe_concentration = (
            transferrin.default_tffe_rel_concentration * transferrin.default_tf_concentration
        )
        transferrin.default_tffe2_concentration = (
            transferrin.default_tffe2_rel_concentration * transferrin.default_tf_concentration
        )

        transferrin.rel_iron_imp_exp_unit_t = self.time_step / transferrin.iron_imp_exp_t
        # TODO: I just commented out the voxel_volume code in the config file. Putting it here.
        #  Adjust comments in config?
        transferrin.ma_iron_import_rate = self.config.getfloat('ma_iron_import_rate') / voxel_volume
        transferrin.ma_iron_export_rate = self.config.getfloat('ma_iron_export_rate') / voxel_volume

        # initialize the molecular field
        transferrin.grid['Tf'] = transferrin.default_apotf_concentration
        transferrin.grid['TfFe'] = transferrin.default_tffe_concentration
        transferrin.grid['TfFe2'] = transferrin.default_tffe2_concentration

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteStatus

        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        # molecules: MoleculesState = state.molecules

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            # TODO: what is going on with these min's? hard to believe that these constants will > 1
            qtty_fe2 = transferrin.grid['TfFe2'][tuple(macrophage_cell_voxel)] * min(
                1.0, transferrin.ma_iron_import_rate * transferrin.rel_iron_imp_exp_unit_t
            )

            qtty_fe = transferrin.grid['TfFe'][tuple(macrophage_cell_voxel)] * min(
                1.0, transferrin.ma_iron_import_rate * transferrin.rel_iron_imp_exp_unit_t
            )

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
                    * transferrin.ma_iron_export_rate
                    * transferrin.rel_iron_imp_exp_unit_t,
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
        self.diffuse(transferrin.grid['Tf'], state)
        self.diffuse(transferrin.grid['TfFe'], state)
        self.diffuse(transferrin.grid['TfFe2'], state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        transferrin: TransferrinState = state.transferrin
        voxel_volume = state.voxel_volume

        concentration_0fe = np.mean(transferrin.grid['Tf']) / voxel_volume
        concentration_1fe = np.mean(transferrin.grid['TfFe']) / voxel_volume
        concentration_2fe = np.mean(transferrin.grid['TfFe2']) / voxel_volume

        concentration = concentration_0fe + concentration_1fe + concentration_2fe

        return {
            'concentration': float(concentration),
            '+0Fe concentration': float(concentration_0fe),
            '+1Fe concentration': float(concentration_1fe),
            '+2Fe concentration': float(concentration_2fe),
        }

    def visualization_data(self, state: State):
        transferrin: TransferrinState = state.transferrin
        return 'molecule', transferrin.grid
