using DataFrames
using OrdinaryDiffEq
using Distributions
using ProgressMeter
using Random
using Statistics
using CSV
using LinearAlgebra
using SpecialFunctions

include("./SIR_simulators.jl")
include("../../helper_functions.jl")

using Pkg
Pkg.add(url = "https://github.com/djmorris7/RandomTimeShifts.jl")
using RandomTimeShifts

##

"""
    diff_SIR_coeffs(β, lifetimes, λ, phis)
    
Function which differentiates the functional equation of the PGF and LST. 
        
Arguments: 
    β = effective transmission parameter 
    lifetimes = β + γ
    λ = the growth rate 
    phis = the previous moments used to calculate the nth moment
    
Outputs: 
    A[1] = coefficient of nth moment
    b = constant term
"""
function diff_SIR_coeffs(β, lifetimes, λ, phis)
    n_previous = length(phis)
    n = n_previous + 1
    A, b = RandomTimeShifts.diff_quadratic_1D(β, lifetimes, n, λ, phis)
    A += RandomTimeShifts.lhs_coeffs(1)
    return A[1], b
end

"""
    F_fixed_s(u, pars, t)

Evaluates the Ricatti ODE governing F(s, t) for fixed u0 = s.

Arguments: 
    u = current state
    pars = (R0, γ_inv)
    t = dummy variable for the current time 
    
Outputs: 
    du = value of ODE
"""
function F_fixed_s(u, pars, t)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ
    du = -(β + γ) * u + γ + β * u^2
    return du
end

"""
    F_offspring_ode(s, t, pars)
    
Calculates the PGF for the imbedded process given a fixed s and integrating over (0, t
with pars = (R0, γ_inv).
"""
function F_offspring_ode(s, t, pars)
    u0 = s[1]
    prob = ODEProblem(F_fixed_s, u0, (0, t), pars)
    sol = solve(prob, Tsit5(); save_start = false)

    return sol.u[end]
end

"""
sir_deterministic!(du, u, pars, t)
    
Evaluate the system of ordinary differential equations for the SIR model with parameters 
pars = (R0, γ_inv).

Arguments: 
    du = required for inplace calculations when using OrdinaryDiffEq
    u = current state
    pars = model parameters in form (R0, γ_inv)
    t = dummy variable for the current time
    
Outputs: 
    None
"""
function sir_deterministic!(du, u, pars, t)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ

    s, i, i_log = u

    du[1] = ds = -β * i * s
    du[2] = di = β * i * s - γ * i
    du[3] = di_log = β * s - γ

    return nothing
end

"""
    compute_time_shift_distribution(pars, K, Z0, results_dir)
    
Computes the LST using our method and inverts it. 

Arguments: 
    pars = (R0, γ_inv)
    K = population size 
    Z0 = initial condition for the SIR model
    results_dir = output directory
    
Outputs: 
    None
"""
function compute_time_shift_distribution(pars, K, Z0, results_dir)
    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β = R0 * γ

    a = γ + β
    λ = β - γ
    q = γ / β

    num_moments = 21

    μ = 2 * β / (β + γ)

    diff_SIR_coeffs_(phis) = diff_SIR_coeffs(β, a, λ, phis)

    moments = RandomTimeShifts.calculate_moments_1D(diff_SIR_coeffs_)

    moments_err = moments[end]
    moments = moments[1:(end - 1)]

    ϵ_target = 1e-10
    L = RandomTimeShifts.error_bounds(ϵ_target, moments_err, num_moments - 1)
    # L = determine_L(ϵ_target, num_moments - 1, moments_err)

    u0 = 0.5
    h = 0.1

    prob = ODEProblem(F_fixed_s, u0, (0, h), pars)
    sol = solve(prob, Tsit5(); abstol = 1e-11, reltol = 1e-11)

    μ = exp(λ * h)
    F_offspring(s) = F_offspring_ode(s, h, pars)

    coeffs = RandomTimeShifts.moment_coeffs(moments)
    lst_w = RandomTimeShifts.construct_lst(coeffs, μ, F_offspring, L, Z0[2], λ, h)

    K = Int(1e7)
    I0 = 1
    Z0 = [K - I0, I0, 0]

    q_star = (γ / β)^I0

    W_cdf = RandomTimeShifts.construct_W_cdf_ilst(lst_w, q_star)

    Δt = 0.1
    s_range = 0:Δt:10

    CSV.write(joinpath(results_dir, "phi_vals.csv"),
              DataFrame(s = s_range, phi = lst_w.(s_range)))

    # Distance between points of the CDF 
    h = 0.1
    t_range = -30:h:15

    EW = 1.0

    F_τ_cdf(t) = W_cdf(exp(λ * t) * EW)

    cdf_vals = RandomTimeShifts.eval_cdf(F_τ_cdf, t_range)
    pdf_vals = RandomTimeShifts.pdf_from_cdf(cdf_vals, h) ./ (1 - q_star)

    CSV.write(joinpath(results_dir, "pdf_vals.csv"),
              DataFrame(; t = t_range, pdf = pdf_vals))
end

"""
    estimate_time_shifts(pars, K, Z0, results_dir)

Estimates the time-shift distributions empirically using the difference in hitting times 
between the deterministic and stochastic simulations. 

Arguments: 
    pars = (R0, γ_inv)
    K = population size 
    Z0 = initial condition for the SIR model
    results_dir = output directory
    
Outputs: 
    None
"""
function estimate_time_shifts(pars, K, Z0, results_dir)
    obs_t = 70
    n_peaks = 10000
    peak_timings = zeros(Float64, n_peaks)

    target_Z = 2000

    @showprogress for i in eachindex(peak_timings)
        peak_time = -Inf
        while isinf(peak_time)
            peak_time = sir_hitting_times(pars, K, Z0, target_Z; tf = obs_t)
        end
        peak_timings[i] = peak_time
    end

    u0 = Z0 / K
    u0[3] = log(u0[2])
    tspan = (0, obs_t)

    prob = ODEProblem(sir_deterministic!, u0, tspan, pars)
    sol = solve(prob, Tsit5(); saveat = 1e-5, abstol = 1e-9, reltol = 1e-9, save_idxs = 3)

    u_idx = findfirst(K * exp.(sol.u) .> target_Z)
    peak_timing_det = sol.t[u_idx]
    time_delays_approx = peak_timing_det .- peak_timings

    CSV.write(joinpath(results_dir, "time_delays_approx.csv"),
              DataFrame(; time_shift = time_delays_approx))
end