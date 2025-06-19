# Computation_of_random_time-shifts

Code supporting the manuscript:
> Dylan Morris, John Maclean and Andrew J. Black, 2024. Computation of random time-shift distributions for stochastic population models. Journal of Mathematical Biology, 89, 33, 10.1007/s00285-024-02132-6.

## Installation

To locally reproduce this project you'll need to clone the repository and then run in a Julia terminal: 

```
julia> using Pkg
julia> Pkg.activate("path/to/this/project")
julia> Pkg.instantiate()
```

This will download all depdencies for the project except our module `RandomTimeShifts.jl`. 
To install this see the documentation of [RandomTimeShifts.jl](https://github.com/djmorris7/RandomTimeShifts.jl). 
This package needs only be added once in a particular environment and can take some time due to the dependency on `OrdinaryDiffEq.jl` which is a relatively large package. 
Note that all the dependencies for `RandomTimeShifts.jl` are installed automatically. 

## Running examples

All the examples feature in the `examples/` folder and are self contained.
To run an example model run the `*_main.jl` functions inside the relevant examples folder.
You can the `*_main.jl` files as scripts and all plots and results will be saved in relevant locations. 
These run the different examples from the paper and produce the same results from the paper.

## Visualising the outputs

The `plots.ipynb` notebook produces all the plots for the paper pulling results from the `examples/results/` directory.
**Note:** this notebook and all the plot code is in Python and so you'll need an install of that in order to reproduce the figures presented in the paper.
There is also an additional example of how this methodology
works on a simple branching process model where results are well established, the linear fractional model (as shown in
Kimmel and Axelrod (2015)).

## Tutorial

A tutorial for working with the package is hosted [here](https://djmorris7.github.io/Computation_of_random_time-shifts/).
This example walks through how the SEIR model as presented in the paper, can be formulated in the time-shift framework for estimating $W$ and hence the time-shift $\tau$.
