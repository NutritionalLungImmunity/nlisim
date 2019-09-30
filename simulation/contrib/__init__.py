from math import floor

from simulation.state import State


def constant(state: State) -> State:
    """Initialize all state to constant values configurable at runtime."""
    for name in ('concentration', 'diffusivity', 'wind_x', 'wind_y', 'source'):
        var = getattr(state, name)
        var[:] += state.config.getfloat('constant_mutation', name, fallback=0)
    return state


def point_source(state: State) -> State:
    """Initialize a single point source somewhere inside the domain."""
    px = state.config.getfloat('point_source_mutation', 'px', fallback=0.5)
    py = state.config.getfloat('point_source_mutation', 'py', fallback=0.5)
    value = state.config.getfloat('point_source_mutation', 'value', fallback=1)

    if not (0 <= px <= 1) or not (0 <= py <= 1):
        raise Exception('Invalid point provided')

    ix = floor(px * (state.concentration.shape[1] - 1))
    iy = floor(py * (state.concentration.shape[0] - 1))

    state.concentration[ix, iy] += value
    return state


def random_background_noise(state: State) -> State:
    return state
