import pytest

from simulation.validator import ValidationError, Validator


def test_validate_initial_state(config, state):
    config.validate(state)


def test_custom_validator(state):
    def v(state):
        raise ValidationError('test')

    validate = Validator(extra=[v])
    with pytest.raises(ValidationError):
        validate(state)


def test_validation_context(config):
    with pytest.raises(ValidationError) as excinfo, \
            config.validate.context('test context'):
        config.validate(None)

    error = excinfo.value
    assert 'After execution of "test context":' in str(error)
