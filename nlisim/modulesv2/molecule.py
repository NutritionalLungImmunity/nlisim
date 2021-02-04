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
        # Note: was originally h*k_cat*enzyme*substrate/(substrate+km), but with
        # enzyme /= voxel_volume and substrate /= voxel_volume.
        # This is algebraically equivalent and reduces the number of operations.
        return h * k_cat * enzyme * substrate / (substrate + km * voxel_volume)
