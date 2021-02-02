import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.hepcidin import HepcidinState
from nlisim.modulesv2.macrophage import MacrophageState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.transferrin import TransferrinState
from nlisim.state import State
from nlisim.util import activation_function

@attrs(kw_only=True, repr=False)
class LiverState(ModuleState):
    pass


class Liver(MoleculeModel):
    """Liver"""

    name = 'liver'
    StateClass = LiverState

    def initialize(self, state: State) -> State:
        hepcidin: LiverState = state.liver
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        transferrin : TransferrinState = state.transferrin
        hepcidin: HepcidinState = state.hepcidin
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        geometry: GeometryState = state.geometry

        # interact with transferrin


        #     elif itype is Transferrin:
        #         if self._log_hepcidin is None or self._log_hepcidin < Constants.THRESHOLD_LOG_HEP:
        #             tf = Constants.TF_INTERCEPT + Constants.TF_SLOPE * Constants.THRESHOLD_LOG_HEP
        #         else:
        #             tf = Constants.TF_INTERCEPT + Constants.TF_SLOPE * self._log_hepcidin
        #         tf = tf  # *0.25142602860942986
        #
        #         rate_tf = Util.turnover_rate(reactable.get("Tf"),
        #                                      tf * Constants.DEFAULT_APOTF_REL_CONCENTRATION * Constants.VOXEL_VOL) - 1
        #         rate_tffe = Util.turnover_rate(reactable.get("TfFe"),
        #                                        tf * Constants.DEFAULT_TFFE_REL_CONCENTRATION * Constants.VOXEL_VOL) - 1
        #         rate_tffe2 = Util.turnover_rate(reactable.get("TfFe2"),
        #                                         tf * Constants.DEFAULT_TFFE2_REL_CONCENTRATION * Constants.VOXEL_VOL) - 1
        #
        #         reactable.pinc(rate_tf, "Tf")
        #         reactable.pinc(rate_tffe, "TfFe")
        #         reactable.pinc(rate_tffe2, "TfFe2")
        #
        #         return True

        return state


#
#     elif itype is IL6:
#         global_il6_concentration = reactable.total_molecule[0] / (2 * Constants.SPACE_VOL)  # div 2 : serum
#         if global_il6_concentration > Constants.IL6_THRESHOLD:
#             self._log_hepcidin = Constants.HEP_INTERCEPT + Constants.HEP_SLOPE * math.log(global_il6_concentration,
#                                                                                           10)
#         else:
#             self._log_hepcidin = None
#         return True
#
#     elif itype is Hepcidin:
#         if self._log_hepcidin is None or self._log_hepcidin > Constants.THRESHOLD_LOG_HEP:
#             rate = Util.turnover_rate(reactable.get("Hepcidin"),
#                                       Constants.THRESHOLD_HEP * Constants.VOXEL_VOL) - 1
#         else:
#             rate = Util.turnover_rate(reactable.get("Hepcidin"),
#                                       math.pow(10, self._log_hepcidin) * Constants.VOXEL_VOL) - 1
#         reactable.pinc(rate)
#         return True
#
#     else:
#         return reactable.react(self)
