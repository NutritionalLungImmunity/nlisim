import math

import numpy as np
import scipy.ndimage

from nlisim.module import ModuleModel


class MoleculeModel(ModuleModel):

    @staticmethod
    def diffuse(grid: np.ndarray, diffusion_constant: float):
        # TODO: verify
        grid += diffusion_constant * scipy.ndimage.laplace(grid)

    @staticmethod
    def michaelian_kinetics(*,
                            substrate: np.ndarray,
                            enzyme: np.ndarray,
                            km: float,
                            h: float,
                            k_cat: float = 1.0,
                            voxel_volume: float) -> np.ndarray:
        substrate /= voxel_volume  # transform into M
        enzyme /= voxel_volume

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
            result = y / x_mol
        # zero out problem divides
        result[x_mol == 0] = 0.0

        # enforce bounds
        np.minimum(result, 1.0, out=result)
        np.maximum(result, 0.0, out=result)
        return result
