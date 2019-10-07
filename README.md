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
