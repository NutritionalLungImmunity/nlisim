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
