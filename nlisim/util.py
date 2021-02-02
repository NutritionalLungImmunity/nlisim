import math

import numpy as np


def activation_function(*, x, kd, h, volume, b=1):
    x = x / volume  # CONVERT MOL TO MOLAR
    return h * (1 - b * math.exp(-(x / kd)))


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
