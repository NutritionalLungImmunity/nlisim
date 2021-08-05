from enum import IntEnum
import math
import sys
from typing import Optional, Union

import numpy as np

# ϵ is used in a divide by zero fix: 1/x -> 1/(x+ϵ)
# this value is pretty close to the smallest 64bit float with the property that its
# reciprocal is finite.
EPSILON = 5.57e-309


def activation_function(*, x, kd, h, volume, b=1):
    # x -> x / volume CONVERT MOL TO MOLAR
    return h * (1 - b * np.exp(-(x / volume) / kd))


def turnover_rate(
    *, x: np.ndarray, x_system: float, base_turnover_rate: float, rel_cyt_bind_unit_t: float
):
    if x_system == 0.0:
        return np.full(shape=x.shape, fill_value=np.exp(-base_turnover_rate * rel_cyt_bind_unit_t))
    # NOTE: in formula, voxel_volume cancels. So I cancelled it.
    y = (x - x_system) * math.exp(-base_turnover_rate * rel_cyt_bind_unit_t) + x_system

    result = y / (x + EPSILON)

    # enforce bounds and zero out problem divides
    result[x == 0].fill(0.0)
    np.minimum(result, 1.0, out=result)
    np.maximum(result, 0.0, out=result)

    return result


def iron_tf_reaction(
    *,
    iron: Union[float, np.float64, np.ndarray],
    tf: np.ndarray,
    tf_fe: np.ndarray,
    p1: float,
    p2: float,
    p3: float,
) -> np.ndarray:
    # easier to deal with (1,) array
    if np.isscalar(iron) or type(iron) == float:
        iron = np.array([iron])

    total_binding_site: np.ndarray = 2 * (tf + tf_fe)  # That is right 2*(Tf + TfFe)!
    total_iron: np.ndarray = iron + tf_fe  # it does not count TfFe2

    rel_total_iron: np.ndarray = total_iron / (total_binding_site + EPSILON)
    # enforce bounds and zero out problem divides
    rel_total_iron[total_binding_site == 0] = 0.0
    np.minimum(rel_total_iron, 1.0, out=rel_total_iron)
    np.maximum(rel_total_iron, 0.0, out=rel_total_iron)

    rel_tf_fe: np.ndarray = ((p1 * rel_total_iron + p2) * rel_total_iron + p3) * rel_total_iron

    rel_tf_fe = np.maximum(
        0.0, rel_tf_fe
    )  # one root of the polynomial is at ~0.99897 and goes neg after
    # rel_TfFe = np.minimum(1.0, rel_TfFe) <- not currently needed, future-proof it?

    rel_tf_fe[total_iron == 0] = 0.0
    rel_tf_fe[total_binding_site == 0] = 0.0

    return rel_tf_fe


def michaelian_kinetics(
    *,
    substrate: np.ndarray,
    enzyme: np.ndarray,
    km: float,
    h: float,
    k_cat: float = 1.0,
    voxel_volume: float,
) -> np.ndarray:
    # Note: was originally h*k_cat*enzyme*substrate/(substrate+km), but with
    # enzyme /= voxel_volume and substrate /= voxel_volume.
    # This is algebraically equivalent and reduces the number of operations.
    return h * k_cat * enzyme * substrate / (substrate + km * voxel_volume)


class TissueType(IntEnum):
    AIR = 0
    BLOOD = 1
    OTHER = 2
    EPITHELIUM = 3
    SURFACTANT = 4  # unused 1/28/2021
    PORE = 5  # unused 1/28/2021

    @classmethod
    def validate(cls, value: np.ndarray):
        return np.logical_and(value >= 0, value <= 5).all() and np.issubclass_(
            value.dtype.type, np.integer
        )


def nan_filter(value: Union[np.ndarray, float]) -> Optional[float]:
    value = float(value)  # for numpy scalars
    if not math.isinf(value) and not math.isnan(value):
        return value
    else:
        print(f"Got an {value}", file=sys.stderr)
        return None
