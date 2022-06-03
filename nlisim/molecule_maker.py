import inspect
import itertools
import sys
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.diffusion import apply_diffusion
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import Datatype, name_validator, turnover_rate

AdvanceAction = Callable[[State, ModuleModel], State]


def upper_first(s: str) -> str:
    return s[0].upper() + s[1:]


@attrs(kw_only=True)
class MoleculeFactory:
    module_name: str = attrib(validator=name_validator)
    components: Optional[List[str]] = attrib(default=None)
    config_fields: Dict[str, Datatype] = attrib(factory=dict)
    computed_fields: Dict[str, Tuple[Datatype, Callable]] = attrib(factory=dict)
    advance_actions: List[Tuple[int, AdvanceAction]] = attrib(factory=list)
    custom_summary_stats: Optional[Callable] = attrib(default=None)
    custom_visualization_data: Optional[Callable] = attrib(default=None)

    def get_module_state(self, state: State) -> ModuleState:
        """
        Get the state class for this module.

        Parameters
        ----------
        state : State
            Global simulation state

        Returns
        -------
        ModuleState
            The state class for this module.
        """
        try:
            module_state = getattr(state, self.module_name)
            return module_state
        except AttributeError:
            print(f"{self.module_name} not found in global state!", file=sys.stderr)
            sys.exit(-1)

    def get_grid(self, state: State) -> np.ndarray:
        """
        Get the grid array for this molecule.

        Parameters
        ----------
        state : State
            Global Simulation state

        Returns
        -------
        numpy array for the molecule
        """
        module_state = self.get_module_state(state)
        try:
            grid = getattr(module_state, 'grid')
        except AttributeError:
            print(
                f"{self.module_name} does not have a `grid` field! Is this really a molecule?",
                file=sys.stderr,
            )
            sys.exit(-1)

        return grid

    def set_components(self, components: Optional[List[str]] = None) -> 'MoleculeFactory':
        """
        Set the components of the molecular field.

        Parameters
        ----------
        components : list of strings, optional


        Returns
        -------
        MoleculeFactory (self), for chaining
        """
        if len(components) <= 0:
            self.components = None
        else:
            from copy import deepcopy

            self.components = deepcopy(components)
        return self

    def add_diffusion(self, priority: int = 0) -> 'MoleculeFactory':
        """
        Add a diffusion step to the advance function.

        Parameters
        ----------
        priority : int, optional
            Used to set the order in which the step is performed. (low to high)

        Returns
        -------
        MoleculeFactory (self), for chaining
        """

        def diffusion(state: State, module_model: ModuleModel) -> State:
            grid = self.get_grid(state)
            molecules: MoleculesState = state.molecules

            if self.components is not None:
                for component in self.components:
                    grid[component][:] = apply_diffusion(
                        variable=grid[component],
                        laplacian=molecules.laplacian,
                        diffusivity=molecules.diffusion_constant,
                        dt=module_model.time_step,
                    )
            else:
                grid[:] = apply_diffusion(
                    variable=grid,
                    laplacian=molecules.laplacian,
                    diffusivity=molecules.diffusion_constant,
                    dt=module_model.time_step,
                )
            return state

        self.advance_actions.append((priority, diffusion))

        return self

    def add_degradation(
        self,
        priority: int = 0,
        half_life_multiplier_name: Optional[str] = None,
        turnover: bool = True,
        system_amount_per_voxel_name: Optional[str] = None,
    ) -> 'MoleculeFactory':
        """
        Add a degradation step to the advance function.

        Parameters
        ----------
        priority : int, optional
            Used to set the order in which the step is performed. (low to high)
        half_life_multiplier_name : str, optional
            Name of the field (in the state class) which holds the per-time-step
            half-life multiplier. If not supplied or None, this step is skipped
        turnover : bool
            Perform the turnover step. Default: True
        system_amount_per_voxel_name
            Name of the field (in the state class) which holds the system concentration, for
            turn-over. If not supplied or None, the value is considered to be zero.

        Returns
        -------
        MoleculeFactory (self), for chaining
        """

        # noinspection PyUnusedLocal
        def degradation(state: State, module_model: ModuleModel) -> State:
            module_state = self.get_module_state(state)
            molecules: MoleculesState = state.molecules

            grid = getattr(module_state, 'grid')

            if half_life_multiplier_name is not None:
                grid *= getattr(module_state, half_life_multiplier_name)

            if turnover:
                grid *= turnover_rate(
                    x=grid,
                    x_system=getattr(module_state, system_amount_per_voxel_name)
                    if system_amount_per_voxel_name is not None
                    else 0.0,
                    base_turnover_rate=molecules.turnover_rate,
                    rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
                )
            return state

        self.advance_actions.append((priority, degradation))

        return self

    def add_advance_action(self, action: AdvanceAction, priority: int = 0) -> 'MoleculeFactory':
        """
        Add an action to the advance function.

        Parameters
        ----------
        action : AdvanceAction, a function taking the global state and the module's model class
            The action to be performed
        priority : int, optional
            Used to set the order in which the step is performed. (low to high)

        Returns
        -------
        MoleculeFactory (self), for chaining
        """
        self.advance_actions.append((priority, action))
        return self

    def add_config_field(
        self,
        field_name: str,
        data_type: Datatype,
    ) -> 'MoleculeFactory':
        """
        Add a field to the module's state, to be read from the config file.

        Parameters
        ----------
        field_name : str
            Name of field to be added, must agree with the field's name in the config file.
        data_type : data-type
            Only `int` and `float` are supported at this time

        Returns
        -------
        MoleculeFactory (self), for chaining
        """
        if field_name in self.config_fields or field_name in self.computed_fields:
            raise RuntimeError(f"Field {field_name} added twice to {self.module_name}")
        self.config_fields[field_name] = data_type
        return self

    def add_computed_field(
        self,
        field_name: str,
        data_type: Datatype,
        initializer: Callable,
    ) -> 'MoleculeFactory':
        """
        Add a field to the module's state, to be computed from fields read from the config file.

        Note: Since python's dict implementation maintains insertion order, it is possible to
        create a computed field that depends on other computed fields by making sure that the
        dependant field is added afterwards. However, if this is ever changed, this trick will
        no-longer work and so we advise against this.

        Parameters
        ----------
        field_name : str
            Name of field to be added.
        data_type : data-type
            Only `int` and `float` are supported at this time.
        initializer : Callable
            a function which computes the value of the field based on fields from the config file
            This function must take _only_ keyword arguments and must accept _all_ keyword
            arguments. e.g.
            ```
            def half_life_multiplier(*, time_step, half_life, **kwargs):
                return 0.5 ** (time_step / half_life)
            ```

        Returns
        -------
        MoleculeFactory (self), for chaining
        """
        if field_name in self.config_fields or field_name in self.computed_fields:
            raise RuntimeError(f"Field {field_name} added twice to {self.module_name}")
        self.computed_fields[field_name] = (data_type, initializer)
        return self

    def build(self) -> Tuple[Type[MoleculesState], Type[ModuleModel]]:
        """
        Construct the state and model classes.

        Returns
        -------
        Tuple of the state class (MoleculesState) and the model class (ModuleModel)
        """
        module_state_class = molecule_state_class_builder(
            module_name=self.module_name,
            state_fields=dict(**self.config_fields, **self.computed_fields),
            components=None
            if self.components is None
            else list(zip(self.components, itertools.repeat(np.float64))),
        )
        module_model_class = molecule_model_class_builder(
            module_name=self.module_name,
            state_class=module_state_class,
            config_fields=self.config_fields,
            computed_fields=self.computed_fields,
            advance_actions=self.advance_actions,
            summary_stats=self.custom_summary_stats,
            visualization_data=self.custom_visualization_data,
        )
        return module_state_class, module_model_class


def molecule_grid_factory(self) -> np.ndarray:
    """Factory method for creating molecular field"""
    if self.components is not None:
        return np.zeros(shape=self.global_state.grid.shape, dtype=self.components)
    else:
        return np.zeros(shape=self.global_state.grid.shape, dtype=np.float64)


@attr.define
class MoleculeState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    components: Optional[List[Tuple[str, Datatype]]] = attrib(default=None)


def molecule_state_class_builder(
    *,
    module_name: str,
    state_fields: Dict[str, Any],
    components: Optional[List[Tuple[str, Datatype]]] = None,
) -> Type[MoleculesState]:
    new_class: Type[MoleculesState] = typing.cast(
        attr.make_class(
            name=upper_first(module_name) + "State",
            attrs={
                'grid': attrib(default=attr.Factory(molecule_grid_factory, takes_self=True)),
                'components': attrib(default=components)
                ** {
                    field_name: attrib(type=field_type)
                    for field_name, (field_type, initializer) in state_fields.items()
                },
            },
            bases=(MoleculesState,),
            kw_only=True,
            repr=False,
        ),
        MoleculesState,
    )
    return new_class


def molecule_model_class_builder(
    *,
    module_name: str,
    state_class: Type[ModuleState],
    config_fields: Dict[str, Datatype],
    computed_fields: Dict[str, Tuple[Datatype, Callable]],
    advance_actions: List[Tuple[int, AdvanceAction]],
    docstring=None,
) -> Type[ModuleModel]:
    initialize = create_initialize(
        module_name=module_name, config_fields=config_fields, computed_fields=computed_fields
    )
    advance = create_advance(module_name=module_name, advance_actions=advance_actions)
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
            computed_value = initializer(**params)
            setattr(module_state, field, computed_value)

        return state

    return initialize_prototype


def create_advance(*, module_name: str, advance_actions: List[Tuple[int, AdvanceAction]]):
    def advance(self, state: State) -> State:
        rg.shuffle(advance_actions)
        advance_actions.sort(key=lambda k, a: k)

        for priority, action in advance_actions:
            pass

        return state

    return advance


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
