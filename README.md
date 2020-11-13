# Introduction

## Installing

While not required, it is recommended you use
[pipenv](https://github.com/pypa/pipenv) to create an isolated environment for
running and developing.  To do so, run
```bash
pipenv install --dev
```

See [pipenv](https://pipenv.kennethreitz.org/) for details.  Once this is done,
you can run `pipenv shell` to enter the newly created environment.

## Running

### Run with Python virtual environment

With the simulation package installed, you should have a new command-line
program called `nlisim ` available.  Try running `nlisim --help` to get
more information.  To run a simulation, you will need configure a configuration.
There is an example configuration in the repository to get you started.

Now run simulation up to 50 hours using the first example config:
```bash
nlisim --config config.ini.example run 50
```

### Run with Docker

As an alternative to local installation, the simulation may be run within a Docker container. This
will download the simulation code from
[the latest published version](https://hub.docker.com/repository/docker/nutritionallungimmunity/nlisim).

To run the same simulation up to 50 hours using the first example config:

```bash
mkdir -p output
docker run \
    --rm \
    --mount type=bind,source="$(pwd)/config.ini.example",destination=/opt/nlisim/config.ini,readonly \
    --mount type=bind,source="$(pwd)/output/",destination=/opt/nlisim/output/ \
    nutritionallungimmunity/nlisim run 50
```

You should now have files like `output/simulation-000001.000.hdf5` containing
the simulation state at 1 hour intervals through the full simulation.

Note, since the application requires read access to files, 
[Docker must mount](https://docs.docker.com/storage/bind-mounts/#use-a-read-only-bind-mount) 
them within the container; this example uses `--mount` to 
[prevent nonexistent host paths from being accidentally created](https://github.com/moby/moby/issues/13121).

## Testing

There is a basic test suite included in the package using [tox](https://tox.readthedocs.io/en/latest/)
as a test runner.  Try running the tests now with
```bash
tox
```

This will install the simulation code into a new isolated environment and run
all of the tests.  It will even test against multiple Python versions if you
have them installed.

You can selectively run test environments, or add additional options to
test environments. Useful sub-commands include:

* `tox -e lint`: Run only the style checks.
* `tox -e type`: Run only the type checks.
* `tox -e py3`: Run only the unit tests.
* `tox -e py3 -- --cov`: Run the unit tests and output coverage information.

Finally, you can run `tox -e format` to automatically reformat your code to
comply with some (but unfortunately not all) of the style checks.

# Code organization

* `state.py `

    This module defines classes intended to contain all of the variables that
    make up the simulation state.  An object of the "State" type is passed
    around through all of the simulation methods.

    By design, the state class does not contain any methods that mutate its own
    data.  The state object is primarily a container that can be serialized and
    deserialized easily for simulation diagnostic output and checkpoints.

* `grid.py `

    This module defines a class representing the discretization of the 3D
    simulation domain.  Any variable representing a quantity that exists over
    the entire spatial domain should be split into chunks defined by this grid.
    For more details, see `nlisim.grid`.

* `cell.py `

    This module contains high level, but efficient, data structures representing
    "cells" in the simulation.  At minimum, a cell is an object containing two
    attributes representing the position of the cell in the domain and whether
    or not it is alive.  This data structure is intended to be extended by
    modules to create cells with additional attributes and behavior.

* `config.py `

    This module defines a subclass of a Python
    [ConfigParser](https://docs.python.org/3/library/configparser.html?highlight=configparser#configparser.ConfigParser)
    which parses ".ini" style files for runtime configuration of the
    simulation.  While not currently enforced, it is expected that all values
    in the configuration object are immutable.  This is the primary distinction
    between the simulation config and the simulation state.

* `module.py `

    This defines the "module" API for extending the simulation.  See below for
    a detailed description of this API.

* `modules/* `

    Modules under `modules` are "optional" extensions to the main simulation.
    They contain functions that are called in one of the extension points in
    the main solver: initialization and iteration.  These functions could be
    moved to an external package, but for simplicity in this proof of concept
    are included with the main simulation.

* `validation.py `

    This module contains a custom exception type dedicated for errors thrown by
    validation methods.  This along with a custom context generator defined
    here are intended to provide extra information about the source of errors
    (e.g. which module was executing) after they are thrown.

* `cli.py `

    This module uses [click](https://click.palletsprojects.com/en/7.x/) to generate
    a command-line interface for executing the simulation.  Click provides a flexible
    API for designing high quality CLIs.

# Simulation state

Absent any additional modules, the simulation state is an object containing only the
following attributes:

* time: The simulation time as a floating point number
* grid: An instance of `nlisim.grid.RectangularGrid` defining the simulation discretization
* config: An instance of `nlisim.config.SimulationConfig` providing configuration options

When a module is included in the runtime configuration, its own state is
included as an additional "dynamic" attribute on this object.  For example when
the "afumigatus" module is included, you have access to `state.afumigatus `
containing the afumigatus module's state.

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
of `nlisim.module.Module`.  Features are added by overriding attributes on
this class.  The following is small example demonstrating some of the features:
```python
import attr

from nlisim.module import ModuleModel, ModuleState


class HelloWorld(ModuleModel):
    name = 'hello_world'

    @attr.s(kw_only=True, auto_attribs=True)
    class StateClass(ModuleState):
        target: str = attr.ib(default='')

    def initialize(self, state):
        state.hello_world.target = self.config.get('target_string')
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

# Visualization Config

To visualize the output of the simulation, please add the variable names to the list `visual_variables` in the config file, under the `[visualization]` section:
```ini
[visualization]
# vtk_type: STRUCTURED_POINTS, STRUCTURED_GRID, RECTILINEAR_GRID, UNSTRUCTURED_GRID, POLY_DATA
visual_variables =  [
    {
        "module":"afumigatus",
        "variable":"tree",
        "vtk_type":"POLY_DATA",
        "attributes":["iron_pool", "iteration"]
    },
    {
        "module":"geometry",
        "variable":"lung_tissue",
        "vtk_type":"STRUCTURED_POINTS",
        "attributes":[]
    }
]
```

For example, to visualize the aspergillus and the alveolar geometry, the variable `afumigatus.tree` with attributes `iron_pool` and `iteration` and the variable `geometry.lung_tissue` are added to the list, followed by their [VTK dataset formats](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf).

* structured point: points data are regularly and uniformly spaced
* rectilinear grid: points data are regularly spaced but can be not uniform
* structured grid: points data are not regularly and not uniformly spaced
* unstructured grid: consists of arbitrary combinations of any possible cell type
* polygonal data: consists of a set of discrete points, vertices, lines or polygons
