from tempfile import NamedTemporaryFile

from simulation.config import SimulationConfig


def test_config_defaults():
    config = SimulationConfig()
    assert config.getboolean('simulation', 'validate') is True


def test_config_default_merging():
    config = SimulationConfig({'simulation': {'custom_val': 5}})
    # Existing defaults should be merged
    assert config.getboolean('simulation', 'validate') is True


def test_config_default_overwrite():
    config = SimulationConfig({'simulation': {'validate': False}})
    # Defaults should be overwritable
    assert config.getboolean('simulation', 'validate') is False


def test_config_dict():
    config = SimulationConfig({'custom_section': {'custom_val': 5}})
    assert config.getint('custom_section', 'custom_val') == 5


def test_config_file():
    with NamedTemporaryFile('r+') as cf:
        cf.write('[custom_section]\ncustom_val = 5\n')
        cf.flush()
        cf.seek(0)
        config = SimulationConfig(cf.name)
    assert config.getint('custom_section', 'custom_val') == 5


def test_config_multiple():
    config = SimulationConfig(
        {'custom_section': {'custom_val_1': 5, 'custom_val_2': 10}},
        {'custom_section': {'custom_val_1': 15}},
    )
    # Configs should merge, with granular overwrites
    assert config.getint('custom_section', 'custom_val_1') == 15
    assert config.getint('custom_section', 'custom_val_2') == 10
