from typing import Callable, Optional, Tuple

from numba import jit
import numpy as np
from numpy.linalg import LinAlgError


@jit
def bdf_solver(
        f: Callable,
        jac: Callable,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        dt: Optional[float] = None,
        ):
    if dt is None:
        dt = (t_span[1] - t_span[0]) / 10

    remaining_datapoints = np.ceil((t_span[1] - t_span[0]) / dt)
    assert remaining_datapoints >= 1

    ts = np.zeros(remaining_datapoints + 1, dtype=np.float64)
    ys = np.zeros((remaining_datapoints + 1,) + y0.shape, dtype=np.float64)
    ys[0] = y0

    # begin with an implicit Euler step
    t_n = t_span[0]
    ts[0] = t_n
    t_np1 = (
        t_n + dt if remaining_datapoints > 1 else t_span[1]
    )  # make sure that we nail the final t
    ts[1] = t_np1
    # solve y_{n+1} = y_n + dt*f(t_{n+1},y_{n+1}) using Newton-Raphson
    newton_raphson(f, jac, t_np1, y0, ys, 1)
    remaining_datapoints -= 1

    # second step comes from an order 2 BDF method


@jit
def newton_raphson(f, jac, t_np1, y_guess, ys, idx):
    ys[idx] = y_guess  # initial guess
    for _ in range(10):
        try:
            correction = np.linalg.solve(jac(t_np1, ys[idx]), f(t_np1, ys[idx]))
        except LinAlgError:
            break
        ys[idx] -= correction
        if np.allclose(correction, 0.0):
            break
