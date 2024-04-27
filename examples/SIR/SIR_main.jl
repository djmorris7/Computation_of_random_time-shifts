"""
    SIR_main.jl

Author: Dylan Morris
Date: 2023

This script runs the code to produce the figures for the SIR
example (Section 2) in the paper.
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
    println("Running simulations for the intro plots...")
    simulate_intro_data(pars, Z0, K, results_dir)
    println("Running peak timing simulations...")
    simulate_peak_timings(pars, Z0, K, results_dir)
    println("Getting the tau mapping for translating the intro figures...")
    simulate_tau_mapping(pars, Z0, K, results_dir)
    println("Drawing real W samples...")
    draw_W_samples(pars, Z0, K, results_dir)
    # Run time-shift methods
    println("Running time shift distribution stuff...")
    estimate_time_shifts(pars, K, Z0, results_dir)
    return compute_time_shift_distribution(pars, K, Z0, results_dir)
    # Run moment matching method

end

main()

##

# Random.seed!(2022)

# K = Int(10^6)
# Z0 = [K - 1, 1, 0]
# pars = [1.9, 2.0]

# (R0, γ_inv) = pars
# γ = 1 / γ_inv
# β = R0 * γ

# a = γ + β
# λ = β - γ
# q = γ / β

# num_moments = 21

# μ = 2 * β / (β + γ)

# diff_SIR_coeffs_(phis) = diff_SIR_coeffs(β, a, λ, phis)

# moments = RandomTimeShifts.calculate_moments_1D(diff_SIR_coeffs_; num_moments = 5)
# pars_m3 = RandomTimeShifts.minimise_loss(moments, q)

# ##

# (R0, γ_inv) = pars
# γ = 1 / γ_inv
# β = R0 * γ

# a = γ + β
# λ = β - γ
# q = γ / β

# num_moments = 21

# μ = 2 * β / (β + γ)

# diff_SIR_coeffs_(phis) = diff_SIR_coeffs(β, a, λ, phis)

# moments = RandomTimeShifts.calculate_moments_1D(diff_SIR_coeffs_)

# moments_err = moments[end]
# moments = moments[1:(end - 1)]

# ϵ_target = 1e-10
# L = RandomTimeShifts.error_bounds(ϵ_target, moments_err, num_moments - 1)
# # L = determine_L(ϵ_target, num_moments - 1, moments_err)

# u0 = 0.5
# h = 0.1

# prob = ODEProblem(F_fixed_s, u0, (0, h), pars)
# sol = solve(prob, Tsit5(); abstol = 1e-11, reltol = 1e-11)

# μ = exp(λ * h)
# F_offspring(s) = F_offspring_ode(s, h, pars)

# coeffs = RandomTimeShifts.moment_coeffs(moments)
# lst_w = RandomTimeShifts.construct_lst(coeffs, μ, F_offspring, L, Z0[2], λ, h)

# K = Int(1e7)
# I0 = 1
# Z0 = [K - I0, I0, 0]

# q_star = (γ / β)^I0

# W_cdf = RandomTimeShifts.construct_W_cdf_ilst(lst_w, q_star)

# ##

# η, β = load_cme_hyper_params(21)

# invert_lst(s -> lst_w(s) / s, 10, η, β)

# W_cdf(10)

# invert_lst(s -> lst_w(s) / s, 10, η, β)

# # function W_cdf(x)
# #     if x == 0
# #         return q_star
# #     else
# #         # Invert the LST (1 - ϕ(θ)) / θ to get the CCDF then take it from 1.
# #         return 1.0 - invert_lst(s -> (1 - lst_w(s)) / s, x, η, β)
# #     end
# # end
