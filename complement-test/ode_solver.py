from typing import Callable, Tuple

import numba
from numba import njit
import numpy as np
from numpy.linalg import LinAlgError


def make_ode_solver(
    f: Callable[[float, np.ndarray], np.ndarray],
    jac: Callable[[float, np.ndarray], np.ndarray],
    implicit_euler: bool = False,
    jit_compile: bool = False,
) -> Callable[[np.ndarray, Tuple[float, float], float], Tuple[np.ndarray, np.ndarray]]:
    """
    Create a vectorizable ODE solver for stiff first order ODEs.

    Parameters
    ----------
    f: Callable[[float, np.ndarray], np.ndarray]
        The function f in y'=f(t,y) where y is an n-vector.
    jac: Callable[[float, np.ndarray], np.ndarray]
        The Jacobian of f, taking values as (n,n) matrices. Parameters are identical to those of f.
    implicit_euler: bool
        If True, use the implicit Euler scheme to do integration. Otherwise, the solver uses BDF
        formulas orders 1 through 3. The BDF solver is a higher order solver and may have a higher
        accuracy, but the implicit Euler scheme is simpler.
    jit_compile: bool
        Use Numba's jit compilation to optimize the solver. This imparts a one-time startup cost.

    Returns
    -------
    solver: Callable[[np.ndarray, Tuple[float, float], float], Tuple[np.ndarray, np.ndarray]]
        A function with three keyword parameters, y0, t_span, and an optional dt. Here,
        y0 is an (m,n) np.ndarray of initial conditions (where n is the dimension of the ODE and m
        is the number of initial conditions), t_span is a Tuple[float, float] of the time range to
        integrate over and dt is a float specifying the step size. When dt is not provided, it will
        be set so that the t_span takes 16 steps.
    """

    def newton_raphson(
        t_npk: float,
        dt: float,
        bdf_const: float,
        target_const: float,
        y_guess: np.ndarray,
        ys: np.ndarray,
        idx: int,
        dim: int,
    ) -> None:
        ys[idx, :] = y_guess  # store initial guess
        for _ in range(10):  # bounded number of attempts to improve guess
            # noinspection PyBroadException
            try:
                correction = np.linalg.solve(
                    np.identity(dim) - bdf_const * dt * jac(t_npk, ys[idx]),
                    target_const - ys[idx] + bdf_const * dt * f(t_npk, ys[idx]),
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
        idx: int,
    ):
        # initial guess (y_guess) from forward Euler method
        newton_raphson(
            t_np1,
            dt,
            1.0,
            ys[idx - 1, :],
            ys[idx - 1, :] + dt * f(t_np1 - dt, ys[idx - 1, :]),
            ys,
            idx,
            ys.shape[1],
        )

    if jit_compile:
        implicit_euler_step = njit(implicit_euler_step, inline='always')

    if implicit_euler:

        def implicit_euler_solver(
            y0: np.ndarray,  # (...,n) e.g. (m,n) m=#examples n=dimension of ode
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
                if is_broadcast:
                    for broadcast_idx_flat in numba.prange(broadcast_size):
                        broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                        implicit_euler_step(
                            ts[next_time_idx], dt, ys[(np.s_[:], *broadcast_idx)], next_time_idx
                        )
                else:
                    implicit_euler_step(ts[next_time_idx], dt, ys, next_time_idx)

            # there is only one step left, but it might be irregularly spaced
            ts[num_datapoints - 1] = t_end
            if is_broadcast:
                for broadcast_idx_flat in numba.prange(broadcast_size):
                    broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                    implicit_euler_step(
                        t_end,
                        t_end - ts[num_datapoints - 2],
                        ys[(np.s_[:], *broadcast_idx)],
                        num_datapoints - 1,
                    )
            else:
                implicit_euler_step(
                    t_end,
                    t_end - ts[num_datapoints - 2],
                    ys,
                    num_datapoints - 1,
                )

            return ts, ys

        if jit_compile:
            implicit_euler_solver = njit(implicit_euler_solver, parallel=True)

        return implicit_euler_solver

    def bdf2_step(
        t_np1: float,
        dt: float,
        ys: np.ndarray,
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
            ys[idx - 1] + dt * f(t_np1 - dt, ys[idx - 1]),
            ys,
            idx,
            ys.shape[1],
        )

    if jit_compile:
        bdf2_step = njit(bdf2_step, inline='always')

    def bdf3_step(
        t_np1: float,
        dt: float,
        ys: np.ndarray,
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
            ys[idx - 1] + dt * f(t_np1 - dt, ys[idx - 1]),
            ys,
            idx,
            ys.shape[1],
        )

    if jit_compile:
        bdf3_step = njit(bdf3_step, inline='always')

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
        if is_broadcast:
            for broadcast_idx_flat in numba.prange(broadcast_size):
                broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                implicit_euler_step(
                    ts[next_time_idx], dt, ys[(np.s_[:], *broadcast_idx)], next_time_idx
                )
        else:
            implicit_euler_step(ts[next_time_idx], ts[next_time_idx] - ts[prev_time_idx], ys, 1)
        remaining_datapoints -= 1

        # if there is only one step left, and it is irregularly spaced,
        # finish with implicit Euler.
        if remaining_datapoints == 1 and ts[next_time_idx] + dt > t_end:
            ts[-1] = t_end
            if is_broadcast:
                for broadcast_idx_flat in numba.prange(broadcast_size):
                    broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                    implicit_euler_step(
                        t_np1=t_end,
                        dt=t_end - ts[next_time_idx],
                        ys=ys[(np.s_[:], *broadcast_idx)],
                        idx=next_time_idx + 1,
                    )
            else:
                implicit_euler_step(
                    t_np1=t_end, dt=t_end - ts[next_time_idx], ys=ys, idx=next_time_idx + 1
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
        if is_broadcast:
            for broadcast_idx_flat in numba.prange(broadcast_size):
                broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                bdf2_step(
                    t_np1=ts[next_time_idx],
                    dt=dt,
                    ys=ys[(np.s_[:], *broadcast_idx)],
                    idx=next_time_idx,
                )
        else:
            bdf2_step(t_np1=ts[next_time_idx], dt=dt, ys=ys, idx=next_time_idx)
        remaining_datapoints -= 1

        # if there is only one step left, and it is irregularly spaced,
        # finish with implicit Euler.
        if remaining_datapoints == 1 and ts[next_time_idx] + dt > t_end:
            ts[-1] = t_end
            if is_broadcast:
                for broadcast_idx_flat in numba.prange(broadcast_size):
                    broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                    implicit_euler_step(
                        t_np1=t_end,
                        dt=t_end - ts[next_time_idx],
                        ys=ys[(np.s_[:], *broadcast_idx)],
                        idx=next_time_idx + 1,
                    )
            else:
                implicit_euler_step(
                    t_np1=t_end, dt=t_end - ts[next_time_idx], ys=ys, idx=next_time_idx + 1
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
            if is_broadcast:
                for broadcast_idx_flat in numba.prange(broadcast_size):
                    broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                    bdf3_step(
                        t_np1=ts[next_time_idx],
                        dt=dt,
                        ys=ys[(np.s_[:], *broadcast_idx)],
                        idx=next_time_idx,
                    )
            else:
                bdf3_step(t_np1=ts[next_time_idx], dt=dt, ys=ys, idx=next_time_idx)
            remaining_datapoints -= 1

            # if there is only one step left, and it is irregularly spaced,
            # finish with implicit Euler.
            if remaining_datapoints == 1 and ts[next_time_idx] + dt > t_end:
                ts[-1] = t_end
                if is_broadcast:
                    for broadcast_idx_flat in numba.prange(broadcast_size):
                        broadcast_idx = np.unravel_index(broadcast_idx_flat, broadcast_shape)
                        implicit_euler_step(
                            t_np1=t_end,
                            dt=t_end - ts[next_time_idx],
                            ys=ys[(np.s_[:], *broadcast_idx)],
                            idx=next_time_idx + 1,
                        )
                else:
                    implicit_euler_step(t_end, t_end - ts[next_time_idx], ys, next_time_idx + 1)
                remaining_datapoints -= 1

        return ts, ys

    if jit_compile:
        bdf_solver = njit(bdf_solver, parallel=True)

    return bdf_solver
