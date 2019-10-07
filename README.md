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

## Running
