from enum import IntEnum
from typing import Tuple, Union

# noinspection PyPackageRequirements
import numpy as np

from nlisim.coordinates import Point, Voxel
from nlisim.grid import TetrahedralMesh
from nlisim.random import rg

# ϵ is used in a divide by zero fix: 1/x -> 1/(x+ϵ)
EPSILON = 1e-50


def activation_function(*, x, k_d, h, volume, b=1) -> Union[float, np.ndarray]:
    # units:
    # x: atto-mol
    # k_d: aM
    # volume: L
    return h * (1 - b * np.exp(-x / k_d / volume))


def turnover_rate(
    *,
    x: Union[float, np.ndarray],
    x_system: Union[float, np.ndarray],
    base_turnover_rate: float,
    rel_cyt_bind_unit_t: float,
):
    # Note: x and x_system should be both either in M or mols, not a mixture of the two.
    if x_system == 0.0:
        if isinstance(x, float):
            return np.exp(-base_turnover_rate * rel_cyt_bind_unit_t)
        else:
            return np.full(
                shape=x.shape, fill_value=np.exp(-base_turnover_rate * rel_cyt_bind_unit_t)
            )
    # NOTE: in formula, voxel_volume cancels. So I cancelled it.
    y = (x - x_system) * np.exp(-base_turnover_rate * rel_cyt_bind_unit_t) + x_system

    result = np.true_divide(y, x, where=(x != 0.0))

    # enforce bounds and zero out problem divides
    result[x == 0] = 0.0
    np.clip(result, 0.0, 1.0, out=result)

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
    substrate: np.ndarray,
    enzyme: np.ndarray,
    k_m: float,
    h: float,
    k_cat: float = 1.0,
    volume: Union[float, np.ndarray],
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
    substrate: np.ndarray,
    enzyme: np.ndarray,
    k_m: float,
    h: float,
    k_cat: float = 1.0,
) -> np.ndarray:
    """
    Compute Michaelis–Menten kinetics.

    units:
    substrate : atto-M
    enzyme : atto-M
    k_m : atto-M
    h: sec/step
    k_cat: 1/sec

    result: atto-M/step
    """
    # Note: was originally defined by converting to molarity, but can be redefined in terms
    # of mols. This is algebraically equivalent and reduces the number of operations.
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


def choose_voxel_by_prob(
    voxels: Tuple[Voxel, ...], default_value: Voxel, weights: np.ndarray
) -> Voxel:
    """
    Choose a voxels using a non-normalized probability distribution.

    If weights are all zero, the default value is chosen.

    Parameters
    ----------
    voxels
        an tuple of voxels
    default_value
        default return value for when weights are uniformly zero
    weights
        an array of non-negative (unchecked) unnormalized probabilities/weights for the voxels

    Returns
    -------
    a Voxel, from voxels, chosen by the probability distribution, or the default
    """
    normalization_constant = np.sum(weights)
    if normalization_constant <= 0:
        # e.g. if all neighbors are air
        return default_value

    # prepend a zero to detect `failure by zero' in the argmax below
    normalized_weights = np.concatenate((np.array([0.0]), weights / normalization_constant))

    # sample from distribution given by normalized weights
    random_voxel_idx: int = int(np.argmax(np.cumsum(normalized_weights) - rg.uniform() > 0.0) - 1)
    if random_voxel_idx < 0:
        # the only way the 0th could be chosen is by argmax failing
        return default_value
    else:
        return voxels[random_voxel_idx]


def secrete_in_element(
    *,
    mesh: TetrahedralMesh,
    point_field: np.ndarray,
    element_index: Union[int, np.ndarray],
    point: Union[Point, np.ndarray],
    amount: Union[float, np.ndarray],  # units: atto-mol
) -> None:
    proportions = mesh.tetrahedral_proportions(element_index, point)
    points = mesh.element_point_indices[element_index]
    # new pt concentration = (old pt amount + new amount) / pt dual volume
    #    = (old conc * pt dual volume + new amount) / pt dual volume
    #    = old conc + (new amount / pt dual volume)
    np.add.at(
        point_field, points, proportions * amount / mesh.point_dual_volumes[points]
    )  # units: prop * atto-mol / L = atto-M


def uptake_in_element(
    *,
    mesh: TetrahedralMesh,
    point_field: np.ndarray,  # units: atto-M
    element_index: Union[int, np.ndarray],
    point: Union[Point, np.ndarray],
    amount: Union[float, np.ndarray],  # units: atto-mol
) -> None:
    points = mesh.element_point_indices[element_index]
    # TODO: justify
    point_field_proportions = mesh.tetrahedral_proportions(element_index=element_index, point=point)

    # new pt concentration = (old pt amount + new amount) / pt dual volume
    #    = (old conc * pt dual volume + new amount) / pt dual volume
    #    = old conc + (new amount / pt dual volume)
    print(f"{point_field_proportions=}")
    print(f"{amount=}")
    print(f"{mesh.point_dual_volumes[points]=}")
    assert np.all(0.0 <= point_field_proportions) and np.all(
        point_field_proportions <= 1.0
    ), f"{point_field_proportions=}"
    np.subtract.at(
        point_field, points, point_field_proportions * amount / mesh.point_dual_volumes[points]
    )  # units: prop * atto-mol / L = atto-M


def sample_point_from_simplex(num_points: int = 1, dimension: int = 3) -> np.ndarray:
    """
    Generate a uniformly distributed random point from a simplex in probability coordinates.

    Parameters
    ----------
    num_points: int
        The number of sample points to return
    dimension: int
        The dimension of the simplex. e.g. a triangle has dimension 2 and tetrahedron has
        dimension 3. Default value is 3.

    Returns
    -------
    A shape (dimension+1,) or (dimension+1,num_points) np.ndarray of floats between 0.0 and 1.0
     which sum to 1.0.

    """
    if num_points == 1:
        return np.diff(np.sort(np.random.random(dimension)), prepend=0.0, append=1.0)
    else:
        return np.diff(
            np.sort(np.random.random((dimension, num_points)), axis=0),
            prepend=0.0,
            append=1.0,
            axis=0,
        )


def tetrahedral_gradient(*, field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of a (linear) function defined at the points of a tetrahedron

    Parameters
    ----------
    points : a shape=(4,3) np.ndarray of points of a tetrahedron
    field : a shape=(4,) np.ndarray of point values of a function at the points of the tetrahedron

    Returns
    -------
    the gradient of the function as an (3,) np.ndarray
    """
    base_point = points[0, :]
    basis_vectors = points[1:, :] - base_point
    dfield = np.linalg.solve(basis_vectors, field[1:] - field[0])
    return dfield
