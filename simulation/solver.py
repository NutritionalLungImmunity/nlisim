from typing import Iterator, Optional

from simulation.differences import gradient, laplacian
from simulation.state import State

"""
Solves the 2D advection-diffusion equation:

    ∂T
    -- = ∇ ⋅ (d ∇T) - ∇ ⋅ (wT) + S
    ∂t

With homogeneous diffusivity and incompressible flow this becomes:

    ∂T
    -- =  d ∆T - w ⋅ ∇T + S
    ∂t

The terms in the RHS of this equation expand to (where w = (u, v)):

1. diffusion
        d * T_xx + d * T_yy

2. advection
        - u * T_x - v * T_y

3. source
        S

Here, we solve this using Euler's method:

T(t + dt, x, y) <- T(t, x, y) + ( diffusion + advection + source ) * dt
"""


def step(state: State, stop_time: Optional[float] = None) -> State:
    """Advance the state by a single time step."""
    delta_time = state.config.getfloat('simulation', 'time_step')
    concentration = state.concentration

    if stop_time is not None and state.time + delta_time > stop_time:
        delta_time = stop_time - state.time
    else:
        stop_time = state.time + delta_time

    # diffusion
    dq = state.diffusivity * laplacian(state, concentration)

    # advection
    dq -= state.wind_x * gradient(state, concentration, 1)
    dq -= state.wind_y * gradient(state, concentration, 0)

    # source
    dq += state.source

    # mutate the original concentration value
    concentration[:, :] += dq * delta_time
    return state.replace(time=stop_time)


def advance(state: State, target_time: float, initialize: bool = True) -> Iterator[State]:
    validate = state.config.validate
    if initialize:
        for p, f in state.config.initialization_plugins.items():
            with validate.context(p):
                state = f(state)
                validate(state)

    while state.time < target_time:
        for p, f in state.config.iteration_plugins.items():
            with validate.context(p):
                state = f(state)
                validate(state)

        state = step(state, target_time)
        yield state
