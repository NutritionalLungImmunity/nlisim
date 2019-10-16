# Introduction

This repository contains a "proof of concept" implementation of an extensible
simulation framework.  The problem it solves is not intended to be relevant,
nor is the numeric method used intended to be the most appropriate.

The goals of this code are to illustrate a number of proposed patterns
and best practices:
* how to load external code via configuration vectorizing code using
* using numpy primatives to accelerate computation organizing numeric operations as
  "pure functions" and keeping all state in a single serializable object
* classic issues related to numeric stability when using low order methods
* modern best practices for developing python code including unit testing,
  type checking, generating command line interfaces, and packaging for external
  users

## Simulation

The core code solves the 2D advection-diffusion equation given by
```
    ∂T
    -- =  d ∆T - w ⋅ ∇T + S
    ∂t
```
where
* `T`: concentration of the advected quantity (e.g. heat)
* `d`: diffusivity (assumed homogeneous)
* `w`: velocity (e.g. wind)
* `S`: source

This equation is solved using low order finite differences in space and first
order explicit time steps (Euler's method).

## Installing

While not required, it is recommended you use
[pipenv](https://github.com/pypa/pipenv) to create an isolated environment for
running and developing.  To do so, run
```
pipenv install --dev
```
See [pipenv](https://github.com/pypa/pipenv) for details.  Once this is done,
you can run `pipenv shell` to enter the newly created environment.


## Running

With the simulation package installed, you should have a new commandline
program called `simulation` available.  Try running `simulation --help` to get
more information.  To run a simulation, you will need configure a configuration.
There are two example configurations in the repository to get you started.

Now run simulation up to 50 seconds using the first example config.
```
cp config.ini.example config.ini
simulation run 50
```
This will open a plot window that will update roughly every half second as the
simulation executes.  This contains a single point source in the middle of the
domain with a constant and homogeneous wind.

The output was saved for every second on simulation time.  You can plot the
contents of one of these files with the command.
```
simulation show output/simulation-000010.000.pkl
```

Now try running with an unstable configuration.  This is the same as the
original, but the time step has been increased to `0.04`.
```
cp config.ini.unstable config.ini
rm -fr output
simulation run 50
```
Notice how the solution develops large oscillations despite starting from the
same conditions as the original.  This is a classic example of numeric
instability encountered when solving differential equations computationally.

## Testing

There is a basic test suite included in the package using [tox](https://tox.readthedocs.io/en/latest/)
as a test runner.  Try running the tests now with
```
tox
```
This will install the simulation code into a new isolated environment and run
all of the tests.  It will even test against multiple python versions if you
have them installed.

If you want, you can even run these tests outside of tox.  For example, try
running `pytest --cov` to get coverage information for the tests run from your
current python environment.  Tox also handles running the linting and type
checking as well.  These can be run standalone with `flake8 simulation` or
`mypy simulation` respectively.

# Code organization

* `state.py`

    This module defines classes intended to contain all of the variables that
    make up the simulation state.  An object of the "State" type is passed
    around through all of the simulation methods.

    By design, the state class does not contain any methods that mutate its own
    data.  The state object is primarily a container that can be serialized and
    deserialized easily for simulation diagnostic output and checkpoints.

* `config.py`

    This module defines a subclass of a python
    [ConfigParser](https://docs.python.org/3/library/configparser.html?highlight=configparser#configparser.ConfigParser)
    which parses ".ini" style files for runtime configuration of the
    simulation.  While not currently enforced, it is expected that all values
    in the configuration object are immutable.  This is the primary distinction
    between the simulation config and the simulation state.

* `module.py`

    This defines the "module" API for extending the simulation.  See below for
    a detailed description of this API.

* `modules/*`

    Modules under `modules` are "optional" extensions to the main simulation.
    They contain functions that are called in one of the extension points in
    the main solver: initialization and iteration.  These functions could be
    moved to an external package, but for simplicity in this proof of concept
    are included with the main simulation.

* `validator.py`

    This module contains a custom exception type dedicated for errors thrown by
    validation methods.  This along with a custom context generator defined
    here are intended to provide extra information about the source of errors
    (e.g. which module was executing) after they are thrown.

* `cli.py`

    This module uses [click](https://click.palletsprojects.com/en/7.x/) to generate
    a commandline interface for executing the simulation.  Click provides a flexible
    API for designing high quality CLI's.

# Simulation state

Absent any additional modules, the simulation state is an object containing only the
following attributes:

* time: The simulation time as a floating point number
* grid: An instance of `RectangularGrid` defining the simulation discretization
* config: An instance of `SimulationConfig` providing configuration options

When a module is included in the runtime configuration, its own state is
included as an additional "dynamic" attribute on this object.  For example when
the "advection" module is included, you have access to `state.advection`
containing the advection module's state.

The State object API directly uses the API of the [attr](https://www.attrs.org)
library.  This provides things like type annotation integration, data
validation, initialization, and many more features.  Developers should read the
attrs documentation to learn more about how to work with the state object.

# Extension modules

At a high level, an extension module contains the following features:

* configuration options
* state variables
* simulation lifecycle handlers:
  * construction (state memory allocation)
  * initialization (initial conditions)
  * iteration (advancing the state in time)

An extension module is registered with the simulation by providing a subclass
of `simulation.module.Module`.  Features are added by overriding attributes on
this class.  The following is small example demonstrating some of the features:

```python
import attr

from simulation.module import Module, ModuleState


class HelloWorld(Module):
    name = 'hello_world'
    defaults = {
        'target': 'World'
    }

    @attr.s(kw_only=True, auto_attribs=True)
    class StateClass(ModuleState):
        target: str = attr.ib(default='')

    def initialize(self, state):
        state.hello_world.target = self.config.get('target')
        return state

    def advance(self, state, previous_time):
        print(f'Hello {state.hello_world.target}!')
        return state
```

When enabled, this module will
* add its `hello_world` namespace to the simulation state containing a new
  scalar value, `target`
* add a configuration option under the section `[hello_world]`
* initialize the state variable `target` to what is provided in the
  config file
* print a message on every time step

