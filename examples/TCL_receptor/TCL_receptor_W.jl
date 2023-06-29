using DataFrames
using CSV
using OrdinaryDiffEq
using KernelDensity
using ProgressMeter
using Distributions
using Random
using Statistics
using StaticArrays

##

include("TCL_receptor_simulators.jl")
include("../../helper_functions.jl")
# using RandomTimeShifts
using Pkg
Pkg.develop(PackageSpec(path = "RandomTimeShifts.jl"))
using RandomTimeShifts

##

"""
    diff_TCL_receptor_coeffs!(A, b, pars, lifetimes, λ, phis)
    
Differentiates the functional equation governing the LST's and progeny generating functions 
for the innate response model updating A and b inplace.
    
Arguments: 
    A = square matrix of size number of types
    b = vector of length number of types 
    pars = parameters of model
    lifetimes = vector of lifetimes
    λ = the growth rate 
    phis = the previous moments used to calculate the nth moment
    
Outputs: 
    None
"""
function diff_TCL_receptor_coeffs!(A, b, pars, lifetimes, λ, phis)
    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars

    n_previous, num_phis = size(phis)

    n = n_previous + 1

    # Calculate the coefficients and constants from differentiation of the functional equation. 
    coeffs1, C1 = RandomTimeShifts.diff_linear(lifetimes[1], n, λ, 2, num_phis; b = σ)
    # Coefficients for LT 2 is the sum of 2 factors. 
    coeffs21, C21 = RandomTimeShifts.diff_quadratic(p_v, lifetimes[2], n, λ, phis;
                                                    phi_idxs = [2, 2, 3])
    coeffs22, C22 = RandomTimeShifts.diff_quadratic(β2, lifetimes[2], n, λ, phis;
                                                    phi_idxs = [2, 2, 1])
    coeffs2 = coeffs21 + coeffs22
    C2 = C21 + C22
    coeffs3, C3 = RandomTimeShifts.diff_linear(lifetimes[3], n, λ, 1, num_phis; b = β1)

    coeffs = [coeffs1, coeffs2, coeffs3]

    b .= [C1, C2, C3]

    for (i, c) in enumerate(coeffs)
        c .+= RandomTimeShifts.lhs_coeffs(i; num_phis = num_phis)
        A[i, :] .= c
    end

    return nothing
end

"""
    F_fixed_s_ode!(du, u, pars, t)

Evaluates the Ricatti ODE's governing Fᵢ(s, t) for fixed u0 = s updating du in place.

Arguments: 
    du = required for inplace calculations when using OrdinaryDiffEq
    u = current state
    pars = parameters of model (see function)
    t = dummy variable for the current time 
    
Outputs: 
    None
"""
function F_fixed_s_ode!(du, u, pars, t)
    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars
    du .= [(σ + η) * ((η + σ * u[2]) / (σ + η) - u[1]),
        (γ + p_v + β2) *
        ((γ + p_v * u[2] * u[3] + β2 * u[2] * u[1]) / (γ + p_v + β2) - u[2]),
        (c_v + β1) * ((c_v + β1 * u[1]) / (c_v + β1) - u[3])]
    return nothing
end

"""
    tcl_extinct_ode!(dq, q, pars, t)
    
Formulates the system of ODEs for the extinction probabilities in the innate response model.
    
Arguments: 
    dq = required for inplace calculations when using OrdinaryDiffEq
    q = current state
    pars = parameters of model
    t = dummy variable for the current time
    
Outputs: 
    None
"""
function tcl_extinct_ode!(dq, q, pars, t)
    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars

    lifetimes = [σ + η, γ + p_v + β2, c_v + β1]

    dq .= [
        lifetimes[1] * ((η + σ * q[2]) / lifetimes[1] - q[1]),
        lifetimes[2] * ((γ + p_v * q[2] * q[3] + β2 * q[2] * q[1]) / lifetimes[2] - q[2]),
        lifetimes[3] * ((c_v + β1 * q[1]) / lifetimes[3] - q[3]),
    ]

    return nothing
end

"""
    calculate_moments(pars, num_moments)
    
Calculates the moments for the innate response model. 
    
Arguments: 
    pars = model parameters
    num_moments = the number of moments to calculate
    
Outputs: 
    moments = an array of shape (num_moments, 3) with the moments for W_i in column i
"""
function calculate_moments(pars; num_moments = 31)
    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars
    Ω = [-(σ + η) σ 0
         β2 -γ p_v
         β1 0 -(c_v + β1)]

    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)

    lifetimes = [σ + η, γ + p_v + β2, c_v + β1]

    function diff_TCL_receptor_coeffs_!(A, b, phis)
        diff_TCL_receptor_coeffs!(A, b, pars, lifetimes, λ1, phis)
    end

    moments = RandomTimeShifts.calculate_moments_ND(diff_TCL_receptor_coeffs_!,
                                                    num_moments, Ω)

    return moments
end

"""
    calculate_extinction_probs(pars)
    
Solves for the extinction probabilities. Note that this solves the ODEs which is equivalent to 
the equations stated in the manuscript. 
    
Arguments: 
    pars = model parameters
    
Outputs: 
    q1 = an array of extinction probabilities where element i corresponds to starting with an
         individual of type i.
"""
function calculate_extinction_probs(pars)
    q0 = [0, 0, 0]
    tspan = (0, 10000)

    prob = ODEProblem(tcl_extinct_ode!, q0, tspan, pars)
    prob = ODEProblem(F_fixed_s_ode!, q0, tspan, pars)
    sol = solve(prob,
                Tsit5();
                save_start = false,
                save_everystep = false,
                save_end = true,
                abstol = 1e-9,
                reltol = 1e-9)

    q1 = sol.u[1]
    return q1
end

"""
    compute_time_shift_distribution(pars, Z0; ϵ = 1e-6, h = 0.1)
    
Computes the pdf and cdf values for the time-shift distribution. 
    
Arguments: 
    pars = model parameters
    Z0 = initial condition for the Branching process (E, I, V)
    num_moments = number of moments to use in the calculation
    ϵ = the tolerance for the neighbourhood about 0, has a default
    h = the step size for the imbedded process, defaults at h = 1.0
    
Outputs: 
    t_range = range of values where the pdf is evaluated
    pdf_vals = pdf values corresponding to t_range
    cdf_vals = cdf values corresponding to t_range
"""
function compute_time_shift_distribution(pars, Z0; ϵ = 1e-6, h = 0.1)
    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars
    Ω = [-(σ + η) σ 0
         β2 -γ p_v
         β1 0 -(c_v + β1)]

    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)
    EW = sum(Z0 .* u_norm)

    moments = calculate_moments(pars)
    error_moment = moments[end, :]
    moments = moments[begin:(end - 1), :]
    num_moments = size(moments, 1)
    L = RandomTimeShifts.error_bounds(ϵ, error_moment, num_moments)

    prob = ODEProblem(F_fixed_s_ode!, zeros(length(Z0)), (0, h), pars,
                      save_start = false, saveat = h)
    F_offspring(s) = RandomTimeShifts.F_offspring_ode(s, prob)
    coeffs = RandomTimeShifts.moment_coeffs(moments)

    μ = exp(h * λ1)
    lst_w = RandomTimeShifts.construct_lst(coeffs, μ, F_offspring, L, Z0, λ1, h)

    q1 = calculate_extinction_probs(pars)
    q_star = prod(q1 .^ Z0)
    W_cdf_ilst = RandomTimeShifts.construct_W_cdf_ilst(lst_w, q_star)

    # Distance between points of the CDF 
    Δt = 0.001
    t_range = -2.5:Δt:1.5

    F_τ_cdf(t) = W_cdf_ilst(exp(λ1 * t) * EW)
    cdf_vals = RandomTimeShifts.eval_cdf(F_τ_cdf, t_range)
    # Differentiating removes the point mass and then we just need to renormalise 
    pdf_vals = RandomTimeShifts.pdf_from_cdf(cdf_vals, Δt) ./ (1 - q_star)
    # We need to remove off the jump in the CDF and then renormalise
    # Think about this at the limits x -> 0 and x -> ∞ of 
    # (F(x) - q_star) / (1 - q_star) and it makes sense.
    cdf_vals = (cdf_vals .- q_star) ./ (1 - q_star)

    return t_range, pdf_vals, cdf_vals
end

"""
    tcl_receptor_deterministic!(dx, x, pars, t)
    
Evaluates the deterministic approximation at a particular value of x. 

Arguments: 
    dx = required for inplace calculations when using OrdinaryDiffEq
    x = current state
    pars = model parameters
    t = dummy variable for the current time
    
Outputs: 
    None
"""
function tcl_receptor_deterministic!(dx, x, pars, t)
    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars
    u, r, e, i, v, f = x

    dx[1] = -β1 * v * u - β2 * i * u - ϱ * f * u + δ * r
    dx[2] = ϱ * f * u - δ * r
    dx[3] = β1 * v * u + β2 * i * u - σ * e - η * e
    dx[4] = σ * e - γ * i
    dx[5] = p_v * i - c_v * v - β1 * u * v
    dx[6] = p_a * i - c_a * f - ϱ * u * f

    return nothing
end

"""
    estimate_time_shifts(pars, Z0, K)
    
Estimates the time-shift distributions using simulation. 

Arguments: 
    pars = model parameters
    Z0 = initial condition 
    K = population size
    
Outputs: 
    time_delays_approx = approximate time-shifts 
"""
function estimate_time_shifts(pars, Z0, K)
    obs_t = 10.0
    n_peaks = 50000
    peak_timings = zeros(Float64, n_peaks)

    target_Z = 2000

    @showprogress for i in eachindex(peak_timings)
        peak_time = -Inf
        while isinf(peak_time)
            peak_time = tcl_receptor_hitting_time(pars, K, Z0, target_Z; tf = obs_t)
        end
        peak_timings[i] = peak_time
    end

    u0 = Z0 / K
    tspan = (0, obs_t)

    prob = ODEProblem(tcl_receptor_deterministic!, u0, tspan, pars)
    sol = solve(prob, Tsit5(); saveat = 1e-5, abstol = 1e-9, reltol = 1e-9, save_idxs = 5)

    u_idx = findfirst(K * sol.u .> target_Z)
    peak_timing_det = sol.t[u_idx]
    time_delays_approx = peak_timing_det .- peak_timings

    return time_delays_approx
end