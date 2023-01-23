from typing import Callable, List, Tuple, Union

import numba
from numba import njit
import numpy as np
from numpy.linalg import LinAlgError

# y' = f(t, y, y_delay)
# y, y' are (n,) arrays
# y_delay is an (n,m) array with m=(number of delays)
#
# then the jacobian of f is
# J_f(t, y, y_delay)
# with values that are (n,n) arrays
# (delay values are constants, so they act more like parameters here)


@njit(inline='always')
def index(ts: np.ndarray, bound: float) -> int:
    for idx, t in np.ndenumerate(ts):
        if t >= bound:
            return idx[0]
    return -1


def make_delay_solver(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    delays: Union[List[float], np.ndarray],
    implicit_euler: bool = False,
    jit_compile: bool = False,
) -> Callable[[np.ndarray, Tuple[float, float], float], Tuple[np.ndarray, np.ndarray]]:
    """
    Create a vectorizable delay differential equation (DDE) solver for stiff first order DDEs.

    Parameters
    ----------
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
        The function f in y'=f(t,y,y_delay) where y is an n-vector and y_delay is an (d,n) array of
        d delayed y values.
    jac: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
        The Jacobian of f, taking values as (n,n) matrices. Parameters are identical to those of f.
    delays: Union[List[float], np.ndarray]
        A list of the (d,) delays used in y_delay.
    implicit_euler: bool
        If True, use the implicit Euler scheme to do integration. Otherwise, the solver uses BDF
        formulas orders 1 through 3. The BDF solver is a higher order solver and may have a higher
        accuracy, but the implicit Euler scheme is simpler.
    jit_compile: bool
        Use Numba's jit compilation to optimize the solver. This imparts a one-time startup cost.

    Returns
    -------
    solver: Callable[[np.ndarray, Tuple[float, float], float], Tuple[np.ndarray, np.ndarray]]
        A function with three parameters, y0, t_span, and an optional dt. Here, y0 is an
        (m,n) np.ndarray of initial conditions (where n is the dimension of the DDE and m is the
        number of initial conditions), t_span is a Tuple[float, float] of the time range to
        integrate over and dt is a float specifying the step size. When dt is not provided, it will
        be set so that the t_span takes 16 steps.
    """

    def make_delay(ts: np.ndarray, ys: np.ndarray, delay: float, t_current: float) -> np.ndarray:
        """
        Compute delayed values of ys.

        Parameters
        ----------
        ts: np.ndarray
        ys: np.ndarray
        delay: float
        t_current: float

        Returns
        -------

        """
        # Find the place where we meet or exceed the current time. The time we want should be
        # between ts[idx-1] and ts[idx].
        # back fill all before-times with the initial values of y
        if t_current - delay < ts[0]:
            return ys[0]
        # now try to find the index
        idx = index(ts, t_current - delay)
        # if we hit the value more-or-less exactly (should be most common case?), return already
        # computed value
        if np.isclose(ts[idx] + delay, t_current):
            return ys[idx]
        # otherwise interpolate
        dt = ts[idx] - ts[idx - 1]
        remaining_delay = delay - (t_current - ts[idx])
        return ys[idx] * (1 - remaining_delay / dt) + ys[idx - 1] * (remaining_delay / dt)

    if jit_compile:
        make_delay = njit(make_delay, inline='always')

    def newton_raphson(
        t_npk: float,
        dt: float,
        bdf_const: float,
        target_const: float,
        y_guess: np.ndarray,
        ys: np.ndarray,
        ys_delay: np.ndarray,
        idx: int,
        dim: int,
    ) -> None:
        ys[idx, :] = y_guess  # store initial guess
        for _ in range(10):  # bounded number of attempts to improve guess
            # noinspection PyBroadException
            try:
                correction = np.linalg.solve(
                    np.identity(dim) - bdf_const * dt * jac(t_npk, ys[idx], ys_delay),
                    target_const - ys[idx] + bdf_const * dt * f(t_npk, ys[idx], ys_delay),
                )
                ys[idx] += correction
            except Exception:  # LinAlgError <- Numba doesn't like this
                break

            # Numba wants these (usually) default-optional arguments explicit
            if np.allclose(correction, 0.0, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
                break

    if jit_compile:
        newton_raphson = njit(newton_raphson, inline='always')

    def implicit_euler_step(
        t_np1: float,
        dt: float,
        ys: np.ndarray,
        ys_delay: np.ndarray,
        idx: int,
    ):
        # initial guess (y_guess) from forward Euler method
        newton_raphson(
            t_np1,
            dt,
            1.0,
            ys[idx - 1, :],
            ys[idx - 1, :] + dt * f(t_np1 - dt, ys[idx - 1, :], ys_delay),
            ys,
            ys_delay,
            idx,
            ys.shape[1],
        )

    if jit_compile:
        implicit_euler_step = njit(implicit_euler_step, inline='always')

    def implicit_euler_advance(
        *,
        broadcast_shape,
        broadcast_size,
        dt,
        is_broadcast,
        next_time_idx,
        ts,
        ys,
    ):
        if is_broadcast:
            for broadcast_idx_flat in numba.prange(broadcast_size):
                broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                ys_delay = np.stack(
                    [
                        make_delay(ts, ys[(np.s_[:], *broadcast_idx)], delay, ts[next_time_idx])
                        for delay in delays
                    ],
                    axis=-1,
                )
                implicit_euler_step(
                    ts[next_time_idx],
                    dt,
                    ys[(np.s_[:], *broadcast_idx)],
                    ys_delay,
                    next_time_idx,
                )
        else:
            ys_delay = np.stack(
                [make_delay(ts, ys, delay, ts[next_time_idx]) for delay in delays], axis=-1
            )
            implicit_euler_step(ts[next_time_idx], dt, ys, ys_delay, next_time_idx)

    if jit_compile:
        implicit_euler_advance = njit(implicit_euler_advance, parallel=True, inline='always')

    if implicit_euler:

        def implicit_euler_solver(
            y0: np.ndarray,
            t_span: Tuple[float, float],
            dt: float = -1.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
            is_broadcast = len(y0.shape) >= 2
            broadcast_shape = y0.shape[:-1]
            broadcast_size = -1 if not is_broadcast else int(np.prod(broadcast_shape))

            # https://en.wikipedia.org/wiki/Backward_Euler_method
            t_start, t_end = t_span
            if dt <= 0.0:
                dt = (t_end - t_start) / 16

            # number of datapoints, equiv to 1 + total number of integration steps to take
            num_datapoints = 1 + int(np.ceil((t_end - t_start) / dt))
            assert num_datapoints >= 1

            # initialize the ts, ys arrays, size one more than number of integrations to include IVs
            ts = np.zeros(num_datapoints, dtype=np.float64)
            ts[0] = t_start
            ys = np.zeros((num_datapoints,) + y0.shape, dtype=np.float64)
            ys[0] = y0

            for prev_time_idx in range(num_datapoints - 2):
                ################################################################################
                # implicit Euler step
                ################################################################################
                next_time_idx = prev_time_idx + 1
                ts[next_time_idx] = ts[prev_time_idx] + dt
                implicit_euler_advance(
                    broadcast_shape=broadcast_shape,
                    broadcast_size=broadcast_size,
                    dt=dt,
                    is_broadcast=is_broadcast,
                    next_time_idx=next_time_idx,
                    ts=ts,
                    ys=ys,
                )

            # there is only one step left, but it might be irregularly spaced
            ts[num_datapoints - 1] = t_end
            implicit_euler_advance(
                broadcast_shape=broadcast_shape,
                broadcast_size=broadcast_size,
                dt=t_end - ts[num_datapoints - 2],
                is_broadcast=is_broadcast,
                next_time_idx=num_datapoints - 1,
                ts=ts,
                ys=ys,
            )

            return ts, ys

        if jit_compile:
            implicit_euler_solver = njit(implicit_euler_solver)

        return implicit_euler_solver

    def bdf2_step(
        t_np1: float,
        dt: float,
        ys: np.ndarray,
        ys_delay: np.ndarray,
        idx: int,
    ):
        # solve
        # y_{n+2}-(4/3)*y_{n+1}+(1/3)*y_{n}=(2/3)*dt*f(t_{n+2},y_{n+2})
        # using Newton-Raphson
        # initial guess from forward Euler method
        newton_raphson(
            t_np1,
            dt,
            2.0 / 3.0,
            (4.0 * ys[idx - 1] - ys[idx - 2]) / 3.0,
            ys[idx - 1] + dt * f(t_np1 - dt, ys[idx - 1], ys_delay),
            ys,
            ys_delay,
            idx,
            ys.shape[1],
        )

    if jit_compile:
        bdf2_step = njit(bdf2_step, inline='always')

    def bdf3_step(
        t_np1: float,
        dt: float,
        ys: np.ndarray,
        ys_delay: np.ndarray,
        idx: int,
    ):
        # solve
        # y_{n+3}-(18/11)*y_{n+2}+(9/11)*y_{n+1}-(2/11)*y_{n}=(6/11)*dt*f(t_{n+2},y_{n+2})
        # using Newton-Raphson
        # initial guess from forward Euler method
        newton_raphson(
            t_np1,
            dt,
            6.0 / 11.0,
            (18.0 * ys[idx - 1] - 9.0 * ys[idx - 2] + 2.0 * ys[idx - 3]) / 11.0,
            ys[idx - 1] + dt * f(t_np1 - dt, ys[idx - 1], ys_delay),
            ys,
            ys_delay,
            idx,
            ys.shape[1],
        )

    if jit_compile:
        bdf3_step = njit(bdf3_step, inline='always')

    def bdf2_advance(
        *,
        broadcast_shape,
        broadcast_size,
        dt,
        is_broadcast,
        next_time_idx,
        ts,
        ys,
    ):
        if is_broadcast:
            for broadcast_idx_flat in numba.prange(broadcast_size):
                broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                ys_delay = np.stack(
                    [
                        make_delay(ts, ys[(np.s_[:], *broadcast_idx)], delay, ts[next_time_idx])
                        for delay in delays
                    ],
                    axis=-1,
                )
                bdf2_step(
                    ts[next_time_idx],
                    dt,
                    ys[(np.s_[:], *broadcast_idx)],
                    ys_delay,
                    next_time_idx,
                )
        else:
            ys_delay = np.stack(
                [make_delay(ts, ys, delay, ts[next_time_idx]) for delay in delays], axis=-1
            )
            bdf2_step(ts[next_time_idx], dt, ys, ys_delay, next_time_idx)

    if jit_compile:
        bdf2_advance = njit(bdf2_advance, parallel=True, inline='always')

    def bdf3_advance(
        *,
        broadcast_shape,
        broadcast_size,
        dt,
        is_broadcast,
        next_time_idx,
        ts,
        ys,
    ):
        if is_broadcast:
            for broadcast_idx_flat in numba.prange(broadcast_size):
                broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                ys_delay = np.stack(
                    [
                        make_delay(ts, ys[(np.s_[:], *broadcast_idx)], delay, ts[next_time_idx])
                        for delay in delays
                    ],
                    axis=-1,
                )
                bdf3_step(
                    ts[next_time_idx],
                    dt,
                    ys[(np.s_[:], *broadcast_idx)],
                    ys_delay,
                    next_time_idx,
                )
        else:
            ys_delay = np.stack(
                [make_delay(ts, ys, delay, ts[next_time_idx]) for delay in delays], axis=-1
            )
            bdf3_step(ts[next_time_idx], dt, ys, ys_delay, next_time_idx)

    if jit_compile:
        bdf3_advance = njit(bdf3_advance, parallel=True, inline='always')

    def bdf_solver(
        y0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        is_broadcast = len(y0.shape) >= 2
        broadcast_shape = y0.shape[:-1]
        broadcast_size = -1 if not is_broadcast else int(np.prod(broadcast_shape))

        # https://en.wikipedia.org/wiki/Backward_differentiation_formula
        t_start, t_end = t_span
        if dt <= 0.0:
            dt = (t_end - t_start) / 16

        # number of datapoints left to compute,
        # equiv to total number of integration steps left to take
        remaining_datapoints = int(np.ceil((t_end - t_start) / dt))
        assert remaining_datapoints >= 1

        # initialize the ts, ys arrays, size one more than number of integrations to include IVs
        ts = np.zeros(remaining_datapoints + 1, dtype=np.float64)
        ts[0] = t_start
        ys = np.zeros((remaining_datapoints + 1,) + y0.shape, dtype=np.float64)
        ys[0] = y0

        ################################################################################
        # begin with an implicit Euler step
        ################################################################################
        # compute t_{n} and t_{n+1}
        prev_time_idx = 0
        next_time_idx = 1
        ts[next_time_idx] = (
            ts[0] + dt if remaining_datapoints > 1 else t_end
        )  # make sure that we nail the final t, if this is it.
        # solve y_{n+1} = y_n + dt*f(t_{n+1},y_{n+1}) using Newton-Raphson
        implicit_euler_advance(
            broadcast_shape=broadcast_shape,
            broadcast_size=broadcast_size,
            dt=dt,
            is_broadcast=is_broadcast,
            next_time_idx=1,
            ts=ts,
            ys=ys,
        )
        remaining_datapoints -= 1

        # if there is only one step left, and it is irregularly spaced,
        # finish with implicit Euler.
        if remaining_datapoints == 1 and ts[next_time_idx] + dt > t_end:
            prev_time_idx = next_time_idx
            next_time_idx += 1
            ts[-1] = t_end
            implicit_euler_advance(
                broadcast_shape=broadcast_shape,
                broadcast_size=broadcast_size,
                dt=t_end - ts[prev_time_idx],
                is_broadcast=is_broadcast,
                next_time_idx=next_time_idx,
                ts=ts,
                ys=ys,
            )
            remaining_datapoints -= 1

        # if this is the end, return results
        if remaining_datapoints == 0:
            return ts, ys

        ################################################################################
        # second step comes from an order 2 BDF method
        ################################################################################
        prev_time_idx += 1  # = 1
        next_time_idx += 1  # = 2
        ts[next_time_idx] = ts[prev_time_idx] + dt
        bdf2_advance(
            broadcast_shape=broadcast_shape,
            broadcast_size=broadcast_size,
            dt=dt,
            is_broadcast=is_broadcast,
            next_time_idx=next_time_idx,
            ts=ts,
            ys=ys,
        )
        remaining_datapoints -= 1

        # if there is only one step left, and it is irregularly spaced,
        # finish with implicit Euler.
        if remaining_datapoints == 1 and ts[next_time_idx] + dt > t_end:
            prev_time_idx = next_time_idx
            next_time_idx += 1
            ts[-1] = t_end
            implicit_euler_advance(
                broadcast_shape=broadcast_shape,
                broadcast_size=broadcast_size,
                dt=t_end - ts[prev_time_idx],
                is_broadcast=is_broadcast,
                next_time_idx=next_time_idx,
                ts=ts,
                ys=ys,
            )
            remaining_datapoints -= 1

        # if this is the end, return results
        if remaining_datapoints == 0:
            return ts, ys

        ################################################################################
        # third and later steps come from the order 3 BDF method
        ################################################################################
        while remaining_datapoints > 0:
            prev_time_idx += 1
            next_time_idx += 1
            ts[next_time_idx] = ts[prev_time_idx] + dt
            bdf3_advance(
                broadcast_shape=broadcast_shape,
                broadcast_size=broadcast_size,
                dt=dt,
                is_broadcast=is_broadcast,
                next_time_idx=next_time_idx,
                ts=ts,
                ys=ys,
            )
            remaining_datapoints -= 1

            # if there is only one step left, and it is irregularly spaced,
            # finish with implicit Euler.
            if remaining_datapoints == 1 and ts[next_time_idx] + dt > t_end:
                prev_time_idx = next_time_idx
                next_time_idx += 1
                ts[-1] = t_end
                implicit_euler_advance(
                    broadcast_shape=broadcast_shape,
                    broadcast_size=broadcast_size,
                    dt=t_end - ts[prev_time_idx],
                    is_broadcast=is_broadcast,
                    next_time_idx=next_time_idx,
                    ts=ts,
                    ys=ys,
                )
                remaining_datapoints -= 1

        return ts, ys

    if jit_compile:
        bdf_solver = njit(bdf_solver, parallel=True)

    return bdf_solver
