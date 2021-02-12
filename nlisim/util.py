import math

import numpy as np


def activation_function(*,
                        x,
                        kd,
                        h,
                        volume,
                        b=1):
    x = x / volume  # CONVERT MOL TO MOLAR
    return h * (1 - b * np.exp(-x / kd))


def turnover_rate(*,
                  x_mol: np.ndarray,
                  x_system_mol: float,
                  base_turnover_rate: float,
                  rel_cyt_bind_unit_t: float):
    # NOTE: in formula, voxel_volume cancels. So I cancelled it.
    y = ((x_mol - x_system_mol) * math.exp(-base_turnover_rate * rel_cyt_bind_unit_t) + x_system_mol)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = y / x_mol
        # zero out problem divides
        if np.isscalar(result) and x_mol == 0.0:
            result = 0.0
        else:
            result[x_mol == 0].fill(0.0)

    # enforce bounds
    result = np.maximum(np.minimum(result, 1.0), 0.0)
    return result


def iron_tf_reaction(*,
                     iron: np.ndarray,
                     tf: np.ndarray,
                     tf_fe: np.ndarray,
                     p1: float,
                     p2: float,
                     p3: float) -> np.ndarray:
    total_binding_site = 2 * (tf + tf_fe)  # That is right 2*(Tf + TfFe)!
    total_iron = iron + tf_fe  # it does not count TfFe2

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_total_iron = total_iron / total_binding_site
        np.nan_to_num(rel_total_iron, nan=0.0, posinf=0.0, neginf=0.0)
        rel_total_iron = np.maximum(np.minimum(rel_total_iron, 1.0), 0.0)

    rel_tf_fe = ((p1 * rel_total_iron + p2) * rel_total_iron + p3) * rel_total_iron

    rel_tf_fe = np.maximum(0.0, rel_tf_fe)  # one root of the polynomial is at ~0.99897 and goes neg after
    # rel_TfFe = np.minimum(1.0, rel_TfFe) <- not currently needed, future-proof it?

    if np.isscalar(rel_tf_fe):
        if total_iron == 0.0 or total_binding_site == 0.0:
            rel_tf_fe = 0.0
    else:
        rel_tf_fe[total_iron == 0] = 0.0
        rel_tf_fe[total_binding_site == 0] = 0.0

    return rel_tf_fe


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
