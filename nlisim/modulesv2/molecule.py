import math

import numpy as np

from nlisim.module import ModuleModel


class MoleculeModel(ModuleModel):

    @staticmethod
    def michaelian_kinetics(*,
                            substrate: np.ndarray,
                            enzyme: np.ndarray,
                            km: float,
                            h: float,
                            k_cat: float = 1.0,
                            voxel_volume: float) -> np.ndarray:
        substrate = substrate / voxel_volume  # transform into M
        enzyme = enzyme / voxel_volume

        # enzyme = np.minimum(substrate1, substrate2)
        # substrate = np.maximum(substrate1, substrate2)

        # TODO: replace with h * k_cat * enzyme * substrate / (substrate + km * voxel_volume)
        # by multiplying by voxel_volume/voxel_volume and removing the conversions to M
        return h * k_cat * enzyme * substrate * voxel_volume / (substrate + km)

    @staticmethod
    def turnover_rate(*,
                      x_mol: np.ndarray,
                      x_system_mol: float,
                      turnover_rate: float,
                      rel_cyt_bind_unit_t: float):
        # NOTE: in formula, voxel_volume cancels. So I cancelled it.
        y = ((x_mol - x_system_mol) * math.exp(-turnover_rate * rel_cyt_bind_unit_t) + x_system_mol)

        with np.errstate(divide='ignore'):
            retval = y / x_mol
        # zero out problem divides
        retval[x_mol == 0] = 0.0
        # alternate method: np.nan_to_num(retval, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # enforce bounds
        np.minimum(retval, 1.0, out=retval)
        np.maximum(retval, 0.0, out=retval)
        return retval
