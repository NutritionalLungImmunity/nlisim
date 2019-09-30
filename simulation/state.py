from io import BytesIO
from pathlib import Path, PurePath
import pickle
from typing import Any, cast, IO, NamedTuple, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:  # prevent circular imports for type checking
    from simulation.boundary import BoundaryCondition  # noqa
    from simulation.config import SimulationConfig  # noqa


class State(NamedTuple):
    """A container for storing the simulation state at a single time step."""

    # time
    time: float

    # the rectangular grid coordinates (1D each)
    dx: float
    dy: float

    # the quantity being advected (e.g. temperature, pollutant, etc.)
    concentration: np.ndarray

    # simulation parameters
    diffusivity: np.ndarray
    wind_x: np.ndarray
    wind_y: np.ndarray
    source: np.ndarray

    # boundary conditions
    bc: 'BoundaryCondition'

    # simulation configuration
    config: 'SimulationConfig'

    @classmethod
    def load(cls, arg: Union[str, bytes, PurePath, IO[bytes]]) -> 'State':
        if isinstance(arg, (str, PurePath)):
            arg = Path(arg).open('rb')
        if isinstance(arg, bytes):
            arg = BytesIO(arg)

        return cast('State', pickle.load(arg))

    def save(self, arg: Union[str, PurePath, IO[bytes]]) -> None:
        if isinstance(arg, (str, PurePath)):
            arg = Path(arg).open('wb')

        arg.write(self.serialize())

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    def replace(self, **kwargs: Any) -> 'State':
        """Return a new copy of the state with new values."""
        d = self._asdict()
        d.update(**kwargs)
        return self.__class__(**d)
