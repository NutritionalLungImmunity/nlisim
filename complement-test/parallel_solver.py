from typing import Callable, Optional, Tuple

from numba import jit
import numpy as np
from numpy.linalg import LinAlgError


# @jit(nopython=True)
def newton_raphson(
    *,
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t_npk: float,
    dt: float,
    bdf_const: float,
    target_const: float,
    y_guess: np.ndarray,
    ys: np.ndarray,
    idx: int,
    dim: int,
):
    ys[idx] = y_guess  # store initial guess
    for _ in range(10):  # bounded number of attempts to improve guess
        try:
            correction = np.linalg.solve(
                np.identity(dim) - bdf_const * dt * jac(t_npk, ys[idx]),
                target_const - ys[idx] + bdf_const * dt * f(t_npk, ys[idx]),
            )
            ys[idx] += correction
        except LinAlgError:
            break

        print(correction)
        if np.allclose(correction, 0.0):
            break


# @jit(nopython=True)
def implicit_euler(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t_np1: float,
    dt: float,
    y0: np.ndarray,
    ys: np.ndarray,
    idx: int,
):
    newton_raphson(
        f=f,
        jac=jac,
        t_npk=t_np1,
        dt=dt,
        bdf_const=1.0,
        target_const=ys[idx - 1],
        y_guess=y0 + dt * f(t_np1 - dt, y0),  # initial guess from forward Euler method
        ys=ys,
        idx=idx,
        dim=y0.shape[0],
    )


# @jit(nopython=True)
def bdf2(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t_np1: float,
    dt: float,
    y0: np.ndarray,
    ys: np.ndarray,
    idx: int,
):
    # solve
    # y_{n+2}-(4/3)*y_{n+1}+(1/3)*y_{n}=(2/3)*dt*f(t_{n+2},y_{n+2})
    # using Newton-Raphson
    newton_raphson(
        f=f,
        jac=jac,
        t_npk=t_np1,
        dt=dt,
        bdf_const=2.0 / 3.0,
        target_const=(4.0 * ys[idx - 1] - ys[idx - 2]) / 3.0,
        y_guess=y0 + dt * f(t_np1 - dt, y0),  # initial guess from forward Euler method
        ys=ys,
        idx=idx,
        dim=y0.shape[0],
    )


# @jit(nopython=True)
def bdf3(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    t_np1: float,
    dt: float,
    y0: np.ndarray,
    ys: np.ndarray,
    idx: int,
):
    # solve
    # y_{n+3}-(18/11)*y_{n+2}+(9/11)*y_{n+1}-(2/11)*y_{n}=(6/11)*dt*f(t_{n+2},y_{n+2})
    # using Newton-Raphson
    newton_raphson(
        f=f,
        jac=jac,
        t_npk=t_np1,
        dt=dt,
        bdf_const=6.0 / 11.0,
        target_const=(18.0 * ys[idx - 1] - 9.0 * ys[idx - 2] + 2.0 * ys[idx - 3]) / 11.0,
        y_guess=y0 + dt * f(t_np1 - dt, y0),  # initial guess from forward Euler method
        ys=ys,
        idx=idx,
        dim=y0.shape[0],
    )


# @jit(nopython=True)
def implicit_euler_solver(
    *,
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # https://en.wikipedia.org/wiki/Backward_Euler_method
    if dt is None:
        dt = (t_span[1] - t_span[0]) / 16

    # number of datapoints, equiv to 1 + total number of integration steps to take
    num_datapoints = 1 + int(np.ceil((t_span[1] - t_span[0]) / dt))
    assert num_datapoints >= 1

    # initialize the ts, ys arrays, size one more than number of integrations to include IVs
    ts = np.zeros(num_datapoints, dtype=np.float64)
    ts[0] = t_span[0]
    ys = np.zeros((num_datapoints,) + y0.shape, dtype=np.float64)
    ys[0] = y0

    for prev_idx in range(num_datapoints - 2):
        ################################################################################
        # implicit Euler step
        ################################################################################
        next_idx = prev_idx + 1
        ts[next_idx] = ts[prev_idx] + dt
        implicit_euler(f, jac, ts[next_idx], dt, y0, ys, idx=next_idx)

    # there is only one step left, but it might be irregularly spaced
    implicit_euler(
        f, jac, t_span[1], t_span[1] - ts[num_datapoints - 2], y0, ys, idx=num_datapoints - 1
    )
    ts[num_datapoints - 1] = t_span[1]

    return ts, ys


# @jit(nopython=True)
def bdf_solver(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # https://en.wikipedia.org/wiki/Backward_differentiation_formula
    if dt is None:
        dt = (t_span[1] - t_span[0]) / 16

    # number of datapoints left to compute, equiv to total number of integration steps left to take
    remaining_datapoints = int(np.ceil((t_span[1] - t_span[0]) / dt))
    assert remaining_datapoints >= 1

    # initialize the ts, ys arrays, size one more than number of integrations to include IVs
    ts = np.zeros(remaining_datapoints + 1, dtype=np.float64)
    ts[0] = t_span[0]
    ys = np.zeros((remaining_datapoints + 1,) + y0.shape, dtype=np.float64)
    ys[0] = y0

    ################################################################################
    # begin with an implicit Euler step
    ################################################################################
    # compute t_{n} and t_{n+1}
    t_n = ts[0]
    t_np1 = (
        t_n + dt if remaining_datapoints > 1 else t_span[1]
    )  # make sure that we nail the final t
    ts[1] = t_np1
    # solve y_{n+1} = y_n + dt*f(t_{n+1},y_{n+1}) using Newton-Raphson
    implicit_euler(f, jac, t_np1, ts[1] - ts[0], y0, ys, idx=1)
    remaining_datapoints -= 1

    # if there is only one step left, and it is irregular, finish with implicit Euler.
    if remaining_datapoints == 1 and ts[1] + dt > t_span[1]:
        implicit_euler(f, jac, t_span[1], t_span[1] - ts[1], y0, ys, idx=2)
        remaining_datapoints -= 1

    # if this is the end, return results
    if remaining_datapoints == 0:
        return ts, ys

    ################################################################################
    # second step comes from an order 2 BDF method
    ################################################################################
    prev_idx = 1
    next_idx = 2
    ts[next_idx] = ts[prev_idx] + dt
    bdf2(f, jac, t_np1, dt, y0, ys, idx=next_idx)
    remaining_datapoints -= 1

    # if there is only one step left, and it is irregular, finish with implicit Euler.
    if remaining_datapoints == 1 and ts[next_idx] + dt > t_span[1]:
        implicit_euler(f, jac, t_span[1], t_span[1] - ts[next_idx], y0, ys, idx=next_idx + 1)
        remaining_datapoints -= 1

    # if this is the end, return results
    if remaining_datapoints == 0:
        return ts, ys

    ################################################################################
    # third and higher steps come from the order 3 BDF method
    ################################################################################
    while remaining_datapoints > 0:
        prev_idx += 1
        next_idx += 1
        ts[next_idx] = ts[prev_idx] + dt
        bdf3(f, jac, t_np1, dt, y0, ys, idx=next_idx)
        remaining_datapoints -= 1

        # if there is only one step left, and it is irregular, finish with implicit Euler.
        if remaining_datapoints == 1 and ts[next_idx] + dt > t_span[1]:
            implicit_euler(f, jac, t_span[1], t_span[1] - ts[next_idx], y0, ys, idx=next_idx + 1)
            remaining_datapoints -= 1

    return ts, ys
