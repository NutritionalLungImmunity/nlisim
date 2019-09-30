import numpy as np

from simulation.config import SimulationConfig
from simulation.state import State


def create_state(config: SimulationConfig) -> State:
    """Return a new allocated state object from the provided config."""
    nx = config.getint('simulation', 'nx')
    ny = config.getint('simulation', 'ny')
    dx = config.getfloat('simulation', 'dx')
    dy = config.getfloat('simulation', 'dy')
    shp = (ny, nx)

    # set up domain
    time = 0

    # allocate state arrays
    concentration = np.zeros(shp)
    diffusivity = np.zeros(1)
    wind_x = np.zeros(shp)
    wind_y = np.zeros(shp)
    source = np.zeros(shp)

    bc = config.boundary_conditions()

    return config.StateClass(
        time=time, dx=dx, dy=dy,
        concentration=concentration,
        diffusivity=diffusivity,
        wind_x=wind_x,
        wind_y=wind_y,
        source=source,
        bc=bc,
        config=config
    )
