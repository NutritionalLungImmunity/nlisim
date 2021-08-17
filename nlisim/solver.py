from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
from typing import Iterator, Tuple

import attr

from nlisim.config import SimulationConfig
from nlisim.module import ModuleModel
from nlisim.random import rg
from nlisim.state import State
from nlisim.validation import context as validation_context


class Status(Enum):
    initialize: int = 0
    time_step: int = 1
    finalize: int = 2


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
    initial_time = state.time

    @dataclass(order=True)
    class ModuleUpdateEvent:
        event_time: float
        previous_update: float
        module: ModuleModel = field(compare=False)

    # Create and fill a queue of modules to run. This allows for modules to
    # operate on disparate time scales. Modules which do not have a time step
    # set will not be run.
    queue: PriorityQueue[ModuleUpdateEvent] = PriorityQueue()
    for module in state.config.modules:
        if module.time_step is not None and module.time_step > 0:
            queue.put(
                ModuleUpdateEvent(
                    event_time=initial_time, previous_update=initial_time, module=module
                )
            )

    # run the simulation until we meet or surpass the desired time
    # while-loop conditional is on previous time so that all pending
    # modules are run on final iteration
    previous_time: float = initial_time
    while previous_time < target_time and not queue.empty():
        # fill a list with update events that are concurrent, and randomize their order
        concurrent_update_events = [queue.get()]
        event_time = concurrent_update_events[0].event_time
        while not queue.empty():
            update_event = queue.get()
            if update_event.event_time == event_time:
                concurrent_update_events.append(update_event)
            else:
                queue.put(update_event)
                break
        rg.shuffle(concurrent_update_events)

        for update_event in concurrent_update_events:
            m: ModuleModel = update_event.module
            previous_time = update_event.previous_update
            state.time = update_event.event_time

            with validation_context(m.name):
                state = m.advance(state, previous_time)
                attr.validate(state)

            # reinsert module with updated time
            queue.put(
                ModuleUpdateEvent(
                    event_time=state.time + m.time_step, previous_update=state.time, module=m
                )
            )
        yield state


def finalize(state: State) -> State:
    for m in state.config.modules:
        with validation_context(m.name):
            state = m.finalize(state)

    return state


def run_iterator(config: SimulationConfig, target_time: float) -> Iterator[Tuple[State, Status]]:
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
    yield state, Status.initialize

    for state in advance(state, target_time):
        yield state, Status.time_step

    yield finalize(state), Status.finalize


# def run(config: SimulationConfig, target_time: float) -> State:
#     """Run a simulation to the target time and return the result."""
#     for state_iteration, _ in run_iterator(config, target_time):
#         state = state_iteration
#     return state
