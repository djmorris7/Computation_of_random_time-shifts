using DataFrames
using OrdinaryDiffEq
using Distributions
using ProgressMeter
using Random
using Statistics
using CSV
using LinearAlgebra
using SpecialFunctions
using StatsBase
using Roots

include("../../helper_functions.jl")

##

"""
    get_pmf(pars; max_offspring = 25)
    
Computes the pmf for the linear fractional example from Kimmel (2015).
"""
function get_pmf(pars; max_offspring = 25)
    b, p = pars
    pmf = zeros(Float64, max_offspring + 1)
    pmf[1] = (1 - b - p) / (1 - p)
    for (n, k) in enumerate(1:max_offspring)
        pmf[n + 1] = b * (p^(k - 1))
    end
    return Weights(pmf)
end

"""
    simulate_linear_fractional(pars, Z0, T)
    
Simulates T generations of the linear fractional model from Kimmel (2015).
"""
function simulate_linear_fractional(pars, Z0, T)
    Z = zeros(Int, T + 1)
    Z[begin] = Z0
    pmf = get_pmf(pars)

    for n in 2:(T + 1)
        offspring_count = 0
        for _ in 1:Z[n - 1]
            offspring_count += sample(pmf) - 1
        end
        Z[n] = offspring_count
    end
    return Z
end