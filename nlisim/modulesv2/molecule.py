import math

import numpy as np

from nlisim.module import ModuleModel


class MoleculeModel(ModuleModel):

    @staticmethod
    def michaelianKinetics(*,
                           substrate1: np.ndarray,
                           substrate2: np.ndarray,
                           km: float,
                           h: float,
                           k_cat: float = 1.0,
                           voxel_volume: float) -> np.ndarray:

        substrate1 = substrate1 / voxel_volume  # transform into M
        substrate2 = substrate2 / voxel_volume
        enzyme = np.minimum(substrate1, substrate2)
        substrate = np.maximum(substrate1, substrate2)

        # TODO: comment below attached to return, verify this is ok
        # (*voxel_volume) transform back into mol
        return h * k_cat * enzyme * substrate * voxel_volume / (substrate + km)

    @staticmethod
    def turnover_rate(*,
                      x_mol: np.ndarray,
                      x_system_mol: float,
                      turnover_rate: float,
                      rel_cyt_bind_unit_t: float):
        # TODO: ask about this
        # if x_mol == 0 and x_system_mol == 0:
        #     return 0

        # NOTE: in formula, voxel_volume cancels. So I cancelled it.
        y = ((x_mol - x_system_mol) * math.exp(-turnover_rate * rel_cyt_bind_unit_t) + x_system_mol)

        with np.errstate(divide='ignore'):
            retval = y / x_mol
        # zero out problem divides
        retval[x_mol == 0] = 0
        # alternate method: np.nan_to_num(retval, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # enforce bounds
        np.minimum(retval, 1.0, out=retval)
        np.maximum(retval, 0.0, out=retval)
        return retval
