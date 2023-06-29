# Computation_of_random_time-shifts

Code supporting the manuscript:
> Dylan Morris, John Maclean and Andrew J. Black., 2023. Computation of random time shift distributions for stochastic population models.

## Dependencies

All code in this repository is to run the examples in the manuscript and users will also need `RandomTimeShifts.jl` to
run the different computations needed to recreate the analyses.
You can clone `RandomTimeShifts.jl` inside this directory and then the code will load the namespace in easily
(**Note that we intend to change this behaviour once the package is stable enough.**)

## Running examples

All the examples feature in the `examples/` folder and are self contained.
To run an example model run the `*_main.jl` functions inside the relevant examples folder.
These run the different examples from the paper and produce the same results from the paper.
These scripts will setup the appropriate file paths needed for saving output.

## Visualising the outputs

The `plots.ipynb` notebook produces all the plots for the paper pulling results from the `examples/results/` directory.
There is also an additional example of how this methodology
works on a simple branching process model where results are well established, the linear fractional model (as shown in
Kimmel and Axelrod (2015)).
