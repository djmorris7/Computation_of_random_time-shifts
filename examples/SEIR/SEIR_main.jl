"""
    SEIR_main.jl

    Author: Dylan Morris
    Date: 2023

This script runs the code to produce the benchmark figures for the SEIR
example in the paper.
"""

include("SEIR_W.jl")

function main()
    println("Running the SEIR example...")

    # Runs the M3 method for the example which is shown in all plots
    println("Running M3 method: control...")
    run_m3_timeshift_distribution_control()

    # Run the experiments
    println("Running M3 method: moment experiment...")
    run_num_moment_experiment()
    println("Running M3 method: step size experiment...")
    run_h_experiment()
    println("Running M3 method: initial condition experiment...")
    run_initial_condition_experiment()
    println("Running M3 method: system size experiment...")
    run_system_size_experiment()

    # Produce benchmark results
    println("Benchmarking M3 method...")
    run_h_benchmarks()

    return nothing
end

main()
