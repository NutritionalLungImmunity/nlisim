from simulation.state import State


class ValidationError(Exception):
    pass


class Validator(object):
    def __init__(self, skip: bool=False):
        self.skip = skip

    def validate(self, state: State) -> None:
        if not isinstance(state, State):
            raise ValidationError('Invalid state class')
