import math

from attr import attrs
import numpy as np

from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


@attrs(kw_only=True, repr=False)
class LiverState(ModuleState):
    hep_slope: float
    hep_intercept: float
    log_hepcidin: float
    il6_threshold: float
    threshold_log_hep: float
    threshold_hep: float


class Liver(ModuleModel):
    """Liver"""

    name = 'liver'
    StateClass = LiverState

    def initialize(self, state: State) -> State:
        liver: LiverState = state.liver

        # config file values
        liver.hep_slope = self.config.getfloat('hep_slope')
        liver.hep_intercept = self.config.getfloat('hep_intercept')
        liver.il6_threshold = self.config.getfloat('il6_threshold')
        liver.threshold_log_hep = self.config.getfloat('threshold_log_hep')

        # default values
        liver.log_hepcidin = float('-inf')  # TODO: verify valid default value

        # computed values (none)
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
        voxel_volume: float = state.voxel_volume
        space_volume: float = state.space_volume

        # interact with transferrin
        tf = transferrin.tf_intercept + transferrin.tf_slope * max(
            transferrin.threshold_log_hep, liver.log_hepcidin
        )
        rate_tf = turnover_rate(
            x=transferrin.grid['Tf'],
            x_system=tf * transferrin.default_apotf_rel_concentration * voxel_volume,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        rate_tf_fe = turnover_rate(
            x=transferrin.grid['TfFe'],
            x_system=tf * transferrin.default_tffe_rel_concentration * voxel_volume,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        rate_tf_fe2 = turnover_rate(
            x=transferrin.grid['TfFe2'],
            x_system=tf * transferrin.default_tffe2_rel_concentration * voxel_volume,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        transferrin.grid['Tf'] *= rate_tf
        transferrin.grid['TfFe'] *= rate_tf_fe
        transferrin.grid['TfFe2'] *= rate_tf_fe2

        # interact with IL6
        global_il6_concentration = np.sum(il6.grid) / (2 * space_volume)  # div 2 : serum
        if global_il6_concentration > liver.il6_threshold:
            liver.log_hepcidin = liver.hep_intercept + liver.hep_slope * math.log(
                global_il6_concentration, 10
            )
        else:
            liver.log_hepcidin = float('-inf')

        # interact with hepcidin
        system_concentration = (
            liver.threshold_hep
            if liver.log_hepcidin == float('-inf') or liver.log_hepcidin > liver.threshold_log_hep
            else math.pow(10.0, liver.log_hepcidin)
        )
        hepcidin.grid *= turnover_rate(
            x=hepcidin.grid,
            x_system=system_concentration * voxel_volume,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        return state
