import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import attr
from attr import attrib
import numpy as np

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


def upper_first(s: str) -> str:
    return s[0].upper() + s[1:]


StateFields = List[Tuple[str, type, Optional[Callable]]]


def molecule_state_class_factory(*, module_name: str, state_fields: StateFields, dtype=float):
    def molecule_grid_factory(self) -> np.ndarray:
        return np.zeros(shape=self.global_state.grid.shape, dtype=dtype)

    new_class = attr.make_class(
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
            )
    return new_class


def molecule_model_class_factory(*,
                                 module_name: str,
                                 state_class: ModuleState,
                                 state_field_names: StateFields,
                                 docstring=None,
                                 ):
    initialize = create_initialize(module_name, state_field_names)
    advance = create_advance()
    summary_stats = create_summary_stats()
    visualization_data = create_visualization_data()

    new_class = type(
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
            )

    return new_class


def create_initialize(module_name: str, state_field_names: StateFields):
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
        for field, field_type, initializer in state_field_names:
            if initializer is not None:
                # then it is a computed value
                continue
            if field_type == int:
                setattr(module_state, field, self.config.getint(field))
            elif field_type == float:
                setattr(module_state, field, self.config.getfloat(field))
            config_vals[field] = getattr(module_state, field)

        # computed values
        for field, field_type, initializer in state_field_names:
            if initializer is None:
                # then it is a config value
                continue
            from inspect import signature
            sig = signature(initializer).parameters.keys()
            params = {variable: value for variable, value in config_vals.items() if variable in sig}
            computed_value = initializer(
                    **params
                    )  # TODO: document that this must be keyword based, also ordering issues
            setattr(module_state, field, computed_value)

        return state

    return initialize_prototype


def create_advance():
    pass


def create_summary_stats():
    pass


def create_visualization_data():
    pass

# TODO: Hold off on this for the moment, see afumigatus.py and cell_maker.py
# def agent_state_class_factory(module_name: str, field_names: List[str]):
#     new_class = attr.make_class(name=upper_first(module_name) + "State",
#                                 attrs=[],
#                                 bases=(ModuleState,),
#                                 kw_only=True, repr=False)
#     return new_class
