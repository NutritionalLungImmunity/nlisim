import logging
import math

from attr import attrs

from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover


@attrs(kw_only=True, repr=False)
class LiverState(ModuleState):
    hep_slope: float
    hep_intercept: float
    il6_threshold: float  # units: atto-M
    threshold_log_hep: float
    threshold_hep: float


class Liver(ModuleModel):
    """Liver"""

    name = 'liver'
    StateClass = LiverState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        liver: LiverState = state.liver

        # config file values
        liver.hep_slope = self.config.getfloat('hep_slope')  # units: aM
        liver.hep_intercept = self.config.getfloat('hep_intercept')  # units: aM
        liver.il6_threshold = self.config.getfloat('il6_threshold')  # units: aM
        liver.threshold_log_hep = self.config.getfloat('threshold_log_hep')

        # computed values
        liver.threshold_hep = math.pow(10, liver.threshold_log_hep)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.hepcidin import HepcidinState
        from nlisim.modules.il6 import IL6State
        from nlisim.modules.transferrin import TransferrinState

        liver: LiverState = state.liver
        transferrin: TransferrinState = state.transferrin
        il6: IL6State = state.il6
        hepcidin: HepcidinState = state.hepcidin
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        # interact with IL6
        global_il6_concentration = mesh.integrate_point_function(il6.field) / (
            2 * mesh.total_volume
        )  # div 2: serum, units: aM
        if global_il6_concentration > liver.il6_threshold:
            log_hepcidin = liver.hep_intercept + liver.hep_slope * (
                math.log(global_il6_concentration, 10)
            )
        else:
            log_hepcidin = float('-inf')

        # interact with transferrin
        tf = transferrin.tf_intercept + transferrin.tf_slope * max(
            transferrin.threshold_log_hep, log_hepcidin
        )  # units: aM
        turnover(
            field=transferrin.field['Tf'],
            system_concentration=tf * transferrin.default_apotf_rel_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        turnover(
            field=transferrin.field['TfFe'],
            system_concentration=tf * transferrin.default_tffe_rel_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        turnover(
            field=transferrin.field['TfFe2'],
            system_concentration=tf * transferrin.default_tffe2_rel_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # interact with hepcidin
        system_concentration = (
            liver.threshold_hep
            if log_hepcidin == float('-inf') or log_hepcidin > liver.threshold_log_hep
            else math.pow(10.0, log_hepcidin)
        )
        turnover(
            field=hepcidin.field,
            system_concentration=system_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        return state
