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


def iron_tf_reaction(iron: np.ndarray,
                     tf: np.ndarray,
                     tf_fe: np.ndarray,
                     p1: float,
                     p2: float,
                     p3: float) -> np.ndarray:
    total_binding_site = 2 * (tf + tf_fe)  # That is right 2*(Tf + TfFe)!
    total_iron = iron + tf_fe  # it does not count TfFe2

    with np.seterr(divide='ignore'):
        rel_total_iron = total_iron / total_binding_site
        np.nan_to_num(rel_total_iron, nan=0.0, posinf=0.0, neginf=0.0)
        rel_total_iron = np.maximum(np.minimum(rel_total_iron, 1.0), 0.0)

    # rel_TfFe = p1 * rel_total_iron * rel_total_iron * rel_total_iron + \
    #            p2 * rel_total_iron * rel_total_iron + \
    #            p3 * rel_total_iron
    # this reduces the number of operations slightly:
    rel_tf_fe = ((p1 * rel_total_iron + p2) * rel_total_iron + p3) * rel_total_iron

    np.maximum(0.0, rel_tf_fe, out=rel_tf_fe)  # one root of the polynomial is at ~0.99897 and goes neg after
    # rel_TfFe = np.minimum(1.0, rel_TfFe) <- not currently needed, future-proof?
    rel_tf_fe[total_iron == 0] = 0.0
    rel_tf_fe[total_binding_site == 0] = 0.0
    return rel_tf_fe
