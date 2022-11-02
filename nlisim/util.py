from enum import IntEnum
import logging
from typing import Union

import numpy as np

# ϵ is used in a divide by zero fix: 1/x -> 1/(x+ϵ)
EPSILON = 1e-50

logger: logging.Logger = logging.getLogger('nlisim')


def activation_function(*, x, k_d, h, volume, b=1) -> Union[float, np.ndarray]:
    # units:
    # x: atto-mol
    # k_d: aM
    # volume: L
    return h * (1 - b * np.exp(-x / k_d / volume))


def turnover(
    *,
    field: np.ndarray,
    system_concentration: float,
    base_turnover_rate: float,
    rel_cyt_bind_unit_t: float,
) -> None:
    # Note: field and system_concentration should be both either in M or mols, not a mixture
    # of the two.
    assert system_concentration >= 0.0
    assert base_turnover_rate >= 0.0
    assert rel_cyt_bind_unit_t >= 0.0
    if system_concentration == 0.0:
        field *= np.exp(-base_turnover_rate * rel_cyt_bind_unit_t)
    else:
        field[:] = system_concentration + (field - system_concentration) * np.exp(
            -base_turnover_rate * rel_cyt_bind_unit_t
        )


def iron_tf_reaction(
    *,
    iron: Union[float, np.float64, np.ndarray],
    tf: np.ndarray,
    tf_fe: np.ndarray,
    p1: float,
    p2: float,
    p3: float,
) -> np.ndarray:
    # Note: It doesn't matter what the units of iron, tf, and tf_fe are as long as they are the same

    # easier to deal with (1,) array
    if np.isscalar(iron) or type(iron) == float:
        iron = np.array([iron])

    # That is right, 2*(Tf + TfFe)!
    total_binding_site: np.ndarray = 2 * (tf + tf_fe)
    total_iron: np.ndarray = iron + tf_fe  # it does not count TfFe2

    rel_total_iron: np.ndarray = np.divide(  # safe division, defaults to zero when dividing by zero
        total_iron,
        total_binding_site,
        out=np.zeros_like(total_iron),  # source of defaults
        where=total_binding_site != 0.0,
    )
    np.clip(rel_total_iron, 0.0, 1.0, out=rel_total_iron)  # fix any remaining problem divides

    rel_tf_fe: np.ndarray = np.maximum(
        0.0, ((p1 * rel_total_iron + p2) * rel_total_iron + p3) * rel_total_iron
    )  # maximum used as one root of the polynomial is at ~0.99897 and goes neg after

    rel_tf_fe[total_iron == 0] = 0.0
    rel_tf_fe[total_binding_site == 0] = 0.0

    return rel_tf_fe


def michaelian_kinetics(
    *,
    substrate: np.ndarray,  # units: atto-mol
    enzyme: np.ndarray,  # units: atto-mol
    k_m: float,  # units: atto-M
    h: float,  # units: sec/step
    k_cat: float = 1.0,  # units: 1/sec
    volume: Union[float, np.ndarray],  # units: L
) -> np.ndarray:
    """
    Compute Michaelis–Menten kinetics.

    units:
    substrate : atto-mol
    enzyme : atto-mol
    k_m : aM
    h: sec/step
    k_cat: 1/sec
    volume: L

    result: atto-mol/step
    """
    # Note: was originally defined by converting to molarity, but can be redefined in terms
    # of mols. This is algebraically equivalent and reduces the number of operations.
    return h * k_cat * enzyme * substrate / (substrate + k_m * volume)


def michaelian_kinetics_molarity(
    *,
    substrate: np.ndarray,  # units: atto-M
    enzyme: np.ndarray,  # units: atto-M
    k_m: float,  # units: atto-M
    h: float,  # units: sec/step
    k_cat: float = 1.0,  # units: 1/step
) -> np.ndarray:
    """
    Compute Michaelis–Menten kinetics.

    result: atto-M/step
    """
    return h * k_cat * enzyme * substrate / (substrate + k_m)


class GridTissueType(IntEnum):
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
