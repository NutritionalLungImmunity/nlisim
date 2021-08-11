from io import StringIO
from tempfile import NamedTemporaryFile

from pytest import raises

from nlisim.config import SimulationConfig
from nlisim.module import ModuleModel

# Destroy all defaults!
# def test_config_defaults():
#     config = SimulationConfig()
#     assert config.getboolean('simulation', 'validate') is True

# Destroy all defaults!
# def test_config_default_merging():
#     config = SimulationConfig({'simulation': {'custom_val': 5}})
#     # Existing defaults should be merged
#     assert config.getboolean('simulation', 'validate') is True

# Destroy all defaults!
# def test_config_default_overwrite():
#     config = SimulationConfig({'simulation': {'validate': False}})
#     # Defaults should be overwritable
#     assert config.getboolean('simulation', 'validate') is False


def test_config_dict():
    config = SimulationConfig({'custom_section': {'custom_val': 5}})
    assert config.getint('custom_section', 'custom_val') == 5


def test_config_stream():
    with StringIO('[custom_section]\ncustom_val = 5\n') as cf:
        config = SimulationConfig(cf)
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


def test_config_add_module_string():
    config = SimulationConfig()
    config.add_module('nlisim.oldmodules.fungus.Fungus')
    assert len(config.modules) == 1
    assert isinstance(config.modules[0], ModuleModel)


def test_config_add_module_object():
    class ValidModuleModel(ModuleModel):
        name = 'ValidModule'

    config = SimulationConfig()
    config.add_module(ValidModuleModel)
    assert len(config.modules) == 1
    assert isinstance(config.modules[0], ValidModuleModel)


def test_config_add_module_invalid_subclass():
    class NonModule:
        name = 'NonModule'

    config = SimulationConfig()
    with raises(TypeError, match=r'^Invalid module class for'):
        config.add_module(NonModule)


def test_config_add_module_invalid_name():
    class NoNameModuleModel(ModuleModel):
        pass

    config = SimulationConfig()
    with raises(ValueError, match=r'^Invalid module name'):
        config.add_module(NoNameModuleModel)
