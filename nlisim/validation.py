from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, List

if TYPE_CHECKING:  # prevent circular imports for type checking
    from nlisim.state import State  # noqa

ValidatorMethod = Callable[['State'], None]


class ValidationError(Exception):
    """
    An error type raised when a simulation validation condition is violated.

    This also provides the ability to report the context of a validation
    error... that is, to report the specific step causing the validation error.
    """

    def __init__(self, msg: str):
        super().__init__(msg)
        self._ctx: List['str'] = []

    def push_context(self, ctx: str) -> None:
        """
        Push a new execution context onto the exception.

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


@contextmanager
def context(name: str) -> Iterator[None]:
    """Create a new "validation context".

    This returns a context manager.  Any validation error thrown within this context
    will contain a reference to the value provided.
    """
    try:
        yield
    except ValidationError as e:
        e.push_context(name)
        raise
    except Exception:
        # TODO: use simulation logger
        print(f'ERROR: Unhandled exception raised while executing "{name}"')
        raise
