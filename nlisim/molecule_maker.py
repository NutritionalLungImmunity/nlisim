import inspect
import sys
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State
from nlisim.util import Datatype, name_validator


def upper_first(s: str) -> str:
    return s[0].upper() + s[1:]


@attrs(kw_only=True)
class MoleculeFactory:
    module_name: str = attrib(validator=name_validator)
    components: Optional[List[str]] = attrib(default=None)
    config_fields: Dict[str, Datatype] = attrib(factory=dict)
    computed_fields: Dict[str, Tuple[Datatype, Callable]] = attrib(factory=dict)
    custom_advance: Optional[Callable] = attrib(default=None)
    custom_summary_stats: Optional[Callable] = attrib(default=None)
    custom_visualization_data: Optional[Callable] = attrib(default=None)

    def set_components(self, components: Optional[List[str]]) -> 'MoleculeFactory':
        """Set the components of the molecular field"""
        if len(components) <= 0:
            self.components = None
        else:
            from copy import deepcopy

            self.components = deepcopy(components)

        return self

    def add_config_field(
        self,
        field_name: str,
        data_type: Datatype,
    ) -> 'MoleculeFactory':
        pass

    def add_computed_field(
        self,
        field_name: str,
        data_type: Datatype,
        initializer,
    ) -> 'MoleculeFactory':
        pass

    def build(self) -> Tuple[Type[ModuleState], Type[ModuleModel]]:
        module_state_class = molecule_state_class_builder(
            module_name=self.module_name,
            state_fields=dict(**self.config_fields, **self.computed_fields),
            dtype=np.float64,
        )
        module_model_class = molecule_model_class_builder(
            module_name=self.module_name,
            state_class=module_state_class,
            config_fields=self.config_fields,
            computed_fields=self.computed_fields,
        )
        return module_state_class, module_model_class


def molecule_state_class_builder(
    *, module_name: str, state_fields: Dict[str, Any], dtype=np.float64
) -> Type[ModuleState]:
    def molecule_grid_factory(self) -> np.ndarray:
        """Factory method for creating molecular field"""
        return np.zeros(shape=self.global_state.grid.shape, dtype=dtype)

    new_class: Type[ModuleState] = typing.cast(
        attr.make_class(
            name=upper_first(module_name) + "State",
            attrs={
                'grid': attrib(default=attr.Factory(molecule_grid_factory, takes_self=True)),
                **{
                    field_name: attrib(type=field_type)
                    for field_name, field_type, initializer in state_fields
                },
            },
            bases=(ModuleState,),
            kw_only=True,
            repr=False,
        ),
        ModuleState,
    )
    return new_class


def molecule_model_class_builder(
    *,
    module_name: str,
    state_class: Type[ModuleState],
    config_fields: Dict[str, Datatype],
    computed_fields: Dict[str, Tuple[Datatype, Callable]],
    docstring=None,
) -> Type[ModuleModel]:
    initialize = create_initialize(
        module_name=module_name, config_fields=config_fields, computed_fields=computed_fields
    )
    advance = create_advance(module_name=module_name)
    summary_stats = create_summary_stats(module_name=module_name)
    visualization_data = create_visualization_data(module_name=module_name)

    new_class = typing.cast(
        type(
            upper_first(module_name),
            (ModuleModel,),
            {
                "name": module_name,
                "__doc__": docstring if docstring is not None else upper_first(module_name),
                "StateClass": state_class,
                "initialize": initialize,
                "advance": advance,
                "summary_stats": summary_stats,
                "visualization_data": visualization_data,
            },
        ),
        ModuleModel,
    )

    return new_class


def create_initialize(
    *,
    module_name: str,
    config_fields: Dict[str, Datatype],
    computed_fields: Dict[str, Tuple[Datatype, Callable]],
):
    def initialize_prototype(self, state: State) -> State:
        try:
            module_state = getattr(state, module_name)
        except AttributeError:
            print(f"{module_name} not found in global state!", file=sys.stderr)
            sys.exit(-1)

        # first thing to do is to read values from the configuration file. After all are done, we
        # can use them to compute the computed values.

        # config file values
        config_vals: Dict[str, Any] = dict()
        for field, field_type in config_fields.items():
            if field_type == int:
                setattr(module_state, field, self.config.getint(field))
            elif field_type == float:
                setattr(module_state, field, self.config.getfloat(field))
            config_vals[field] = getattr(module_state, field)

        # computed values
        for field, (field_type, initializer) in computed_fields.items():
            signature = inspect.signature(initializer).parameters.keys()
            params = {
                variable: value for variable, value in config_vals.items() if variable in signature
            }
            computed_value = initializer(
                **params
            )  # TODO: document that this must be keyword based, also ordering issues
            setattr(module_state, field, computed_value)

        return state

    return initialize_prototype


def create_advance(*, module_name: str, custom_function: Callable = None):
    if custom_function is not None:
        return custom_function

    # TODO: degrade, defuse

    pass


def create_summary_stats(*, module_name: str, custom_function: Callable = None):
    if custom_function is not None:
        return custom_function

    def summary_stats(self, state: State) -> Dict[str, Any]:
        try:
            module_state = getattr(state, module_name)
        except AttributeError:
            print(f"{module_name} not found in global state!", file=sys.stderr)
            sys.exit(-1)

        voxel_volume = state.voxel_volume

        return {
            'concentration (nM)': float(np.mean(module_state.grid) / voxel_volume / 1e9),
        }

    return summary_stats


def create_visualization_data(*, module_name: str, custom_function: Callable = None):
    if custom_function is not None:
        return custom_function

    def visualization_data(self, state: State):
        try:
            module_state = getattr(state, module_name)
        except AttributeError:
            print(f"{module_name} not found in global state!", file=sys.stderr)
            sys.exit(-1)

        return 'molecule', module_state.grid

    return visualization_data


# TODO: Hold off on this for the moment, see afumigatus.py and cell_maker.py
# def agent_state_class_factory(module_name: str, field_names: List[str]):
#     new_class = attr.make_class(name=upper_first(module_name) + "State",
#                                 attrs=[],
#                                 bases=(ModuleState,),
#                                 kw_only=True, repr=False)
#     return new_class
