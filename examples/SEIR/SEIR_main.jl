"""
SEIR_main.jl 
    This script runs the code to produce the benchmark figures for the SEIR 
    example in the paper. 

Author: Dylan Morris
Date: 2023 
"""

include("SEIR_W.jl")

# Runs the M3 method for the example which is shown in all plots 
run_m3_timeshift_distribution_control()

# Run the experiments 
run_num_moment_experiment()
run_h_experiment()
run_initial_condition_experiment()
run_system_size_experiment()

# Produce benchmark results
run_h_benchmarks()