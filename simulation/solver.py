from math import ceil
from typing import Iterator

import attr

from simulation.config import SimulationConfig
from simulation.state import State
from simulation.validation import context as validation_context


def initialize(state: State) -> State:
    """Initialize a simulation state."""
    for m in state.config.modules:
        with validation_context(f'{m.name} (initialization)'):
            state = m.initialize(state)

    # run validation after all initialization is done otherwise validation
    # could fail on a module's private state before it can initialize itself
    with validation_context('global initialization'):
        attr.validate(state)
    return state


def advance(state: State, target_time: float) -> Iterator[State]:
    """Advance a simulation to the given target time."""
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


def finalize(state: State) -> State:
    for m in state.config.modules:
        with validation_context(m.name):
            state = m.finalize(state)

    return state


def run_iterator(config: SimulationConfig, target_time: float) -> Iterator[State]:
    """Initialize and advance a simulation to the target time.

    This method is a convenience method to automate running the
    methods above that will be sufficient for most use cases.  It
    will:
    1. Construct a new state object
    2. Initialize the state object (yielding the result)
    3. Advance the simulation by single time steps (yielding the result)
    4. Finalize the simulation (yielding the result)
    """
    attr.set_run_validators(config.getboolean('simulation', 'validate'))
    state = initialize(State.create(config))
    yield state

    for state in advance(state, target_time):
        yield state

    yield finalize(state)


def run(config: SimulationConfig, target_time: float) -> State:
    """Run a simulation to the target time and return the result."""
    for state_iteration in run_iterator(config, target_time):
        state = state_iteration
    return state
