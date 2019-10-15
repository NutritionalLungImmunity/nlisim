from math import ceil
from typing import Iterator

import attr

from simulation.state import State
from simulation.validation import context as validation_context


def advance(state: State, target_time: float, initialize: bool = True) -> Iterator[State]:
    """Advance a simulation to the given target time."""
    if initialize:
        for m in state.config.modules:
            with validation_context(m.name):
                state = m.initialize(state)
                attr.validate(state)

    dt = state.config.getfloat('simulation', 'time_step')
    n_steps = ceil((target_time - state.time) / dt)
    initial_time = state.time

    for i in range(n_steps):
        previous_time = state.time
        state.time = min(initial_time + (i + 1) * dt, target_time)

        for m in state.config.modules:
            with validation_context(m.name):
                state = m.advance(state, previous_time)
                attr.validate(state)

        yield state
