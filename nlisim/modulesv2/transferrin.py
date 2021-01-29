import math

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.iron import IronState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.state import State


def molecule_grid_factory(self: 'TransferrinState') -> np.ndarray:
    # note the expansion to another axis to account for 0, 1, or 2 bound Fe's.
    return np.zeros(shape=self.global_state.grid.shape,
                    dtype=[('Tf', np.float),
                           ('TfFe', np.float),
                           ('TfFe2', np.float)])


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


class Transferrin(MoleculeModel):
    """Transferrin"""

    name = 'transferrin'
    StateClass = TransferrinState

    def initialize(self, state: State) -> State:
        transferrin: TransferrinState = state.transferrin
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        transferrin.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')
        transferrin.p1 = self.config.getfloat('p1')
        transferrin.p2 = self.config.getfloat('p2')
        transferrin.p3 = self.config.getfloat('p3')
        transferrin.threshold_log_hep = self.config.getfloat('threshold_log_hep')

        transferrin.tf_intercept = self.config.getfloat('tf_intercept')
        transferrin.tf_slope = self.config.getfloat('tf_slope')

        transferrin.default_apotf_rel_concentration = self.config.getfloat('default_apotf_rel_concentration')
        transferrin.default_tffe_rel_concentration = self.config.getfloat('default_tffe_rel_concentration')
        transferrin.default_tffe2_rel_concentration = self.config.getfloat('default_tffe2_rel_concentration')

        # computed values
        transferrin.threshold = transferrin.k_m_tf_tafc * voxel_volume / 1.0e6
        transferrin.threshold_hep = math.pow(10, transferrin.threshold_log_hep)
        transferrin.default_tf_concentration = (transferrin.tf_intercept +
                                                transferrin.tf_slope * transferrin.threshold_log_hep) * voxel_volume
        transferrin.default_apotf_concentration = transferrin.default_apotf_rel_concentration \
                                                  * transferrin.default_tf_concentration
        transferrin.default_tffe_concentration = transferrin.default_tffe_rel_concentration \
                                                 * transferrin.default_tf_concentration
        transferrin.default_tffe2_concentration = transferrin.default_tffe2_rel_concentration \
                                                  * transferrin.default_tf_concentration

        # initialize the molecular field
        transferrin.grid['Tf'] = transferrin.default_apotf_concentration
        transferrin.grid['TfFe'] = transferrin.default_tffe_concentration
        transferrin.grid['TfFe2'] = transferrin.default_tffe2_concentration

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # TODO: move to cell
        # elif itype is Macrophage:
        #     qttyFe2 = self.get("TfFe2") * Constants.MA_IRON_IMPORT_RATE * Constants.REL_IRON_IMP_EXP_UNIT_T
        #     qttyFe = self.get("TfFe") * Constants.MA_IRON_IMPORT_RATE * Constants.REL_IRON_IMP_EXP_UNIT_T
        #
        #     qttyFe2 = qttyFe2 if qttyFe2 < self.get("TfFe2") else self.get("TfFe2")
        #     qttyFe = qttyFe if qttyFe < self.get("TfFe") else self.get("TfFe")
        #
        #     self.decrease(qttyFe2, "TfFe2")
        #     self.decrease(qttyFe, "TfFe")
        #     self.inc(qttyFe2 + qttyFe, "Tf")
        #     interactable.inc_iron_pool(2 * qttyFe2 + qttyFe)
        #     if interactable.fpn and interactable.status != Macrophage.ACTIVE and interactable.status != Macrophage.ACTIVATING:
        #         qtty = interactable.iron_pool * self.get(
        #             "Tf") * Constants.MA_IRON_EXPORT_RATE * Constants.REL_IRON_IMP_EXP_UNIT_T
        #         # qtty = qttyFe + 2 * qttyFe2
        #         qtty = qtty if qtty <= 2 * self.get("Tf") else 2 * self.get("Tf")
        #         rel_TfFe = Util.iron_tf_reaction(qtty, self.get("Tf"), self.get("TfFe"))
        #         tffe_qtty = rel_TfFe * qtty
        #         tffe2_qtty = (qtty - tffe_qtty) / 2
        #         self.decrease(tffe_qtty + tffe2_qtty, "Tf")
        #         self.inc(tffe_qtty, "TfFe")
        #         self.inc(tffe2_qtty, "TfFe2")
        #         interactable.inc_iron_pool(-qtty)
        #     return True

        # interaction with iron: transferrin -> transferrin+[1,2]Fe
        transferrin_fe_capacity = 2 * transferrin.grid['Tf'] + transferrin.grid['TfFe']
        potential_reactive_quantity = np.minimum(iron.grid, transferrin_fe_capacity)
        rel_TfFe = self.iron_tf_reaction(potential_reactive_quantity,
                                         transferrin.grid["Tf"],
                                         transferrin.grid["TfFe"],
                                         p1=transferrin.p1,
                                         p2=transferrin.p2,
                                         p3=transferrin.p3)
        tffe_qtty = rel_TfFe * potential_reactive_quantity
        tffe2_qtty = (potential_reactive_quantity - tffe_qtty) / 2
        transferrin.grid['Tf'] -= tffe_qtty + tffe2_qtty
        transferrin.grid['TfFe'] += tffe_qtty
        transferrin.grid['TfFe2'] += tffe2_qtty
        iron.grid -= potential_reactive_quantity
        # Note: asked Henrique why there is no transferrin+Fe -> transferrin+2Fe reaction
        # answer was that this should already be accounted for

        # Degrade transferrin: done in liver

        return state

    # TODO: duplicated with code in lactoferrin
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

        rel_TfFe = np.maximum(0.0, rel_TfFe)  # one root of the polynomial is at ~0.99897 and goes neg after
        # TODO: rel_TfFe = np.minimum(1.0, rel_TfFe) <- not currently needed, future-proof?
        rel_TfFe[total_iron == 0] = 0.0
        rel_TfFe[total_binding_site == 0] = 0.0
        return rel_TfFe
