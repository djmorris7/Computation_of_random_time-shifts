"""
SIR_main.jl 
    This script runs the code to produce the figures for the SIR 
    example (Section 2) in the paper. 

Author: Dylan Morris
Date: 2023 
"""

include("SIR.jl")
include("SIR_W.jl")

function main()
    figures_dir, results_dir = make_dirs("SIR")

    Random.seed!(2022)

    K = Int(10^6)
    Z0 = [K - 1, 1, 0]
    pars = [1.9, 2.0]

    # Simulation based results
    simulate_intro_data(pars, Z0, K, results_dir)
    simulate_peak_timings(pars, Z0, K, results_dir)
    simulate_tau_mapping(pars, Z0, K, results_dir)
    draw_W_samples(pars, Z0, K, results_dir)
    # Run time-shift methods 
    estimate_time_shifts(pars, K, Z0, results_dir)
    compute_time_shift_distribution(pars, K, Z0, results_dir)
end

main()