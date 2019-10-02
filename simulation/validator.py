from contextlib import contextmanager
from typing import Callable, List, Iterable, Iterator, Optional

import numpy as np

from simulation.state import State

ValidatorMethod = Callable[[State], None]


class ValidationError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
        self._ctx: List['str'] = []

    def push_context(self, ctx: str) -> None:
        self._ctx.append(ctx)

    def __str__(self) -> str:
        msg = super().__str__()
        for ctx in self._ctx:
            msg = f'After execution of "{ctx}": ' + msg
        return msg


class Validator(object):
    def __init__(self, extra: Iterable[ValidatorMethod]=None, skip: bool=False):
        self.skip = skip
        self.extra = extra or []
        self._ctx: Optional[str] = None

    @contextmanager
    def context(self, ctx: str) -> Iterator['Validator']:
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
        c = state.concentration
        if not isinstance(variable, np.ndarray):
            raise ValidationError(f'Invalid data type for variable {name}')

        if variable.shape != c.shape:
            raise ValidationError(f'Invalid shape for variable {name}')

        if not np.isfinite(variable).all():
            raise ValidationError(f'Variable {name} contains invalid values')
