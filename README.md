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
original, but the time step has been increased to `0.05`.
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

    This module defines a
    [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple)
    containing all of the variables that make up the simulation state.  An
    object of this type is passed around through all of the simulation methods.

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

* `solver.py`

    This module contains the logic that advances the simulation state in time.
    It is also responsible for calling the methods that extend the behavior
    of the simulation that are provided in the configuration.

* `contrib/*`

    Modules under `contrib` are "optional" extensions to the main simulation.
    They contain functions that are called in one of the extension points in
    the main solver: initialization and iteration.  These functions could be
    moved to an external package, but for simplicity in this proof of concept
    are included with the main simulation.

* `validator.py`

    This module contains a "validation" class that is optionally run after each
    mutation of the simulation state.  By default, it only detects invalid
    types and floating point values.  It can be extended by the configuration
    to include additional validation functions as desired.

* `differences.py`

    This module provides functions to compute numeric derivatives of numpy
    arrays using finite differences.  The implementation uses vectorized
    operations to avoid explicit loops in python, which are significantly
    slower.

* `boundary.py`

    This module defines an abstract interface for handling boundary conditions
    when performing numeric differentiation.  Basic Dirichlet and Neumann are
    provided.

* `cli.py`

    This module uses [click](https://click.palletsprojects.com/en/7.x/) to generate
    a commandline interface for executing the simulation.  Click provides a flexible
    API for designing high quality CLI's.


# Discussion

There are a number of issues present in this architecture that should be considered
before moving forward.  I will attempt to outline those that I am aware of here.

## Explicit time stepping

The idea that the system is solved explicitly from `t -> t + ∆t` is baked in to
the architecture.  The iteration-based extensions in particular rely on this
behavior.  As shown in the example, explicit methods have issues with stability
that put upper bounds on the possible time steps.  Stabilized methods including
upwinded and implicit solvers are able to get around this limitation, but don't
offer the same ability to extend and modify the simulation as easily and
efficiently as the existing implementation

## Consider the use of a CFD library

If we want to embed a true fluid dynamic simulation, we might want to consider
the use of a proven fluid dynamic solver rather than building one up on our
own.

## Passing the configuration inside the state

For simplicity, I included the config object with the simulation state.  A
better pattern might be to make the configuration a module-level global...
perhaps something like how flask does
[configuration](https://flask.palletsprojects.com/en/1.1.x/config/) e.g.
`current_app.config`.

## Provide a better way to add state via an extension

The configuration object can specify the class used as the state object.  The
main `State` type can be subclassed to provide additional state variables, but
there is no way at configuration time to append state variables from multiple
sources.

## Provide an integrated API for extensions

Currently, there are a number places a simulation extension can modify the
behavior of the solver: initialization, iteration, state class, and validation.
A potentially common use case for an extension is to do the following:

* Add a configuration section
* Add a new state variable
* Add a validator for the new variable
* Add initialization for the new variable
* Add an iteration method to update the new variable

These are all possible in the current framework, but it is left up to the user
to configure all of the pieces together correctly.  If this is indeed a common
use case, it would be better to provide a unified API to do all of this from a
single extension point.
