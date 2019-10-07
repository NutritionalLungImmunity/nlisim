from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np

from simulation.state import State

ValidatorMethod = Callable[[State], None]


class ValidationError(Exception):
    """
    This is an error type raised when a simulation validation condition is
    violated.  It also provides the ability to report the context of a validation
    error... that is, to report the specific step causing the validation error.
    """
    def __init__(self, msg: str):
        super().__init__(msg)
        self._ctx: List['str'] = []

    def push_context(self, ctx: str) -> None:
        """Push a new execution context onto the exception.

        The context is an arbitrary string meant to provide users a hint as to
        the cause of an error even when plugins are allowed to make arbitrary
        changes to the system state.
        """
        self._ctx.append(ctx)

    def __str__(self) -> str:
        msg = super().__str__()
        for ctx in self._ctx:
            msg = f'After execution of "{ctx}": ' + msg
        return msg


class Validator(object):
    """
    This class is a callable object that validates the state of a simulation.  By default
    it checks state attribute types, numpy array shapes, and numeric values.  It provides
    a hook to inject extra validators from the configuration.
    """
    def __init__(self, extra: Iterable[ValidatorMethod] = None, skip: bool = False):
        """Initialize the validator.

        extra:
            A list of functions that will called with the validation.  These functions take
            a single argument (the system state) and should raise a validation error or return
            None on success.

        skip:
            When true, skip all validation to accelerate the simuation in production mode.
        """
        self.skip = skip
        self.extra = extra or []
        self._ctx: Optional[str] = None

    @contextmanager
    def context(self, ctx: str) -> Iterator['Validator']:
        """Create a new "validation context".

        This returns a context manager.  Any validation error thrown within this context
        will contain a reference to the value provided.
        """
        self._ctx = ctx
        try:
            yield self
        except ValidationError as e:
            e.push_context(self._ctx)
            raise
        except Exception:
            # TODO: print warning message to give exception context
            raise
        finally:
            self._ctx = None

    def __call__(self, state: State) -> None:
        """Perform all validation checks on the provided state."""
        if self.skip:
            return

        if not isinstance(state, State):
            raise ValidationError('Invalid state class')

        if not state.time >= 0:
            raise ValidationError('Invalid time stamp')

        if state.dx <= 0 or state.dy <= 0:
            raise ValidationError('Invalid grid spacing')

        if state.diffusivity.shape != (1,) or state.diffusivity[0] < 0:
            raise ValidationError('Invalid diffusivity')

        for name in ['concentration', 'wind_x', 'wind_y', 'source']:
            self.check_variable(state, name, getattr(state, name))

        for validate in self.extra:
            validate(state)

    @classmethod
    def check_variable(cls, state: State, name: str, variable: np.ndarray) -> None:
        """Check a single numpy variable for valid numeric values."""
        c = state.concentration
        if not isinstance(variable, np.ndarray):
            raise ValidationError(f'Invalid data type for variable {name}')

        if variable.shape != c.shape:
            raise ValidationError(f'Invalid shape for variable {name}')

        if not np.isfinite(variable).all():
            raise ValidationError(f'Variable {name} contains invalid values')
