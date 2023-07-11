using DataFrames
using CSV
using OrdinaryDiffEq
using KernelDensity
using ProgressMeter
using Distributions
using Random
using Statistics
using StaticArrays
using Roots
using BenchmarkTools

##

include("SEIR_simulators.jl")
include("../../helper_functions.jl")

# using Pkg
# Pkg.add(url="https://github.com/djmorris7/RandomTimeShifts.jl")
using RandomTimeShifts

##

"""
    epi_pars_to_sim_pars(pars)
    
Maps the epi parameters (R0, σ_inv, γ_inv) to (β, σ, γ).

Arguments: 
    pars = (R0, σ_inv, γ_inv)
    
Outputs: 
    (β, σ, γ)
"""
function epi_pars_to_sim_pars(pars)
    (R0, σ_inv, γ_inv) = pars
    σ = 1 / σ_inv
    γ = 1 / γ_inv
    β = R0 * γ
    return (β, σ, γ)
end

"""
    diff_SEIR_coeffs!(A, b, β, lifetimes, λ, phis)
    
Differentiates the functional equations for the SEIR model updating 
A and b inplace. 
    
Arguments: 
    A = square matrix of size number of types
    b = vector of length number of types 
    β = effective transmission parameter 
    lifetimes = vector of lifetimes [σ, β + γ]
    λ = the growth rate 
    phis = the previous moments used to calculate the nth moment
    
Outputs: 
    None
"""
function diff_SEIR_coeffs!(A, b, β, lifetimes, λ, phis)
    n_previous, num_phis = size(phis)
    n = n_previous + 1

    coeffs_linear, C1 = RandomTimeShifts.diff_linear(lifetimes[1], n, λ, 2, 2)
    coeffs_quadratic, C2 = RandomTimeShifts.diff_quadratic(β, lifetimes[2], n, λ, phis,
                                                           [2, 1, 2])

    coeffs_linear += RandomTimeShifts.lhs_coeffs(1; num_phis = 2)
    coeffs_quadratic += RandomTimeShifts.lhs_coeffs(2; num_phis = 2)

    A[1, :] .= coeffs_linear
    A[2, :] .= coeffs_quadratic
    b[1] = C1
    b[2] = C2

    return nothing
end

"""
    F_fixed_s!(du, u, pars, t)

Evaluates the Ricatti ODE's governing Fᵢ(s, t) for fixed u0 = s updating du in place.

Arguments: 
    du = required for inplace calculations when using OrdinaryDiffEq
    u = current state
    pars = (R0, σ_inv, γ_inv)
    t = dummy variable for the current time 
    
Outputs: 
    None
"""
function F_fixed_s_ode!(du, u, pars, t)
    β, σ, γ = epi_pars_to_sim_pars(pars)
    du[1] = -σ * u[1] + σ * u[2]
    du[2] = γ - (β + γ) * u[2] + β * u[1] * u[2]
    return nothing
end

"""
    seir_extinct_ode!(dq, q, pars, t)
    
Evaluates the ODE's (inplace) governing the extinction probability for the SEIR BP model. 

Arguments: 
    dq = required for inplace calculations when using OrdinaryDiffEq
    q = current state
    pars = (R0, σ_inv, γ_inv)
    t = dummy variable for the current time
    
Outputs: 
    None
"""
function seir_extinct_ode!(dq, q, pars, t)
    β, σ, γ = epi_pars_to_sim_pars(pars)
    lifetimes = [σ, β + γ]
    dq .= [lifetimes[1] * (q[2] - q[1]), -q[2] * lifetimes[2] + γ + β * q[1] * q[2]]
    return nothing
end

"""
    seir_deterministic!(du, u, pars, t)
    
Evaluates the SEIR deterministic model (system of ODEs).

Arguments: 
    du = required for inplace calculations when using OrdinaryDiffEq
    u = current state
    pars = (R0, σ_inv, γ_inv)
    t = dummy variable for the current time
    
Outputs: 
    None
"""
function seir_deterministic!(du, u, pars, t)
    β, σ, γ = epi_pars_to_sim_pars(pars)
    s, e, i = u

    du[1] = -β * i * s
    du[2] = β * i * s - σ * e
    du[3] = σ * e - γ * i

    return nothing
end

"""
    calculate_moments(pars, num_moments)
    
Calculates the moments for the SEIR model. 
    
Arguments: 
    pars = (R0, σ_inv, γ_inv)
    num_moments = the number of moments to calculate
    
Outputs: 
    moments = an array of shape (num_moments, 2) with the moments for W_i in column i
"""
function calculate_moments(pars, num_moments)
    β, σ, γ = epi_pars_to_sim_pars(pars)
    Ω = [-σ σ
         β -γ]

    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)
    lifetimes = [σ, β + γ]

    diff_SEIR_coeffs_!(A, b, phis) = diff_SEIR_coeffs!(A, b, β, lifetimes, λ1, phis)
    moments = RandomTimeShifts.calculate_moments_ND(diff_SEIR_coeffs_!, num_moments, Ω)

    return moments
end

"""
    calculate_extinction_probs(pars)
    
Solves for the extinction probabilities. Note that this solves the ODEs which is equivalent to 
the equations stated in the manuscript. 

Arguments: 
    pars = (R0, σ_inv, γ_inv)
    
Outputs: 
    q1 = an array of extinction probabilities where element i corresponds to starting with an
         individual of type i.
"""
function calculate_extinction_probs(pars)
    q0 = [0, 0]
    tspan = (0, 10000)

    # prob = ODEProblem(seir_extinct_ode!, q0, tspan, pars)
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
    compute_time_shift_distribution(pars, Z0, num_moments; ϵ = 1e-6, h = 2.0)
    
Computes the time-shift distribution and saves the PDF values. 

Arguments: 
    pars = (R0, σ_inv, γ_inv)
    Z0 = initial condition for the Branching process (E, I)
    num_moments = number of moments to use in the calculation
    ϵ = the tolerance for the neighbourhood about 0, has a default
    h = the step size for the imbedded process, defaults at h = 1.0
    
Outputs: 
    t_range = range of values where the pdf is evaluated
    pdf_vals = pdf values corresponding to t_range
    cdf_vals = cdf values corresponding to t_range
"""
function compute_time_shift_distribution(pars, Z0, num_moments; ϵ = 1e-6, h = 1.0)
    β, σ, γ = epi_pars_to_sim_pars(pars)
    Ω = [-σ σ
         β -γ]

    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)

    EW = sum(Z0 .* u_norm)

    moments = calculate_moments(pars, num_moments + 1)
    error_moment = moments[end, :]
    moments = moments[begin:(end - 1), :]

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
    Δt = 0.1
    t_range = -40:Δt:30

    F_τ_cdf(t) = W_cdf_ilst(exp(λ1 * t) * EW)
    cdf_vals = RandomTimeShifts.eval_cdf(F_τ_cdf, t_range)
    pdf_vals = RandomTimeShifts.pdf_from_cdf(cdf_vals, Δt) ./ (1 - q_star)
    # We need to remove off the jump in the CDF and then renormalise
    # Think about this at the limits x -> 0 and x -> ∞ of 
    # (F(x) - q_star) / (1 - q_star) and it makes sense.
    cdf_vals = (cdf_vals .- q_star) ./ (1 - q_star)

    return t_range, pdf_vals, cdf_vals
end

"""
    estimate_time_shifts(pars, Z0, K)
    
Estimates the time-shift distributions using simulation. 

Arguments: 
    pars = (R0, σ_inv, γ_inv)
    Z0 = initial condition 
    K = population size
    
Outputs: 
    time_delays_approx = approximate time-shifts 
"""
function estimate_time_shifts(pars, Z0, K)
    obs_t = 1000.0
    n_peaks = 50000
    peak_timings = zeros(Float64, n_peaks)

    target_Z = min(K * 0.05, 1000)

    @showprogress for i in eachindex(peak_timings)
        peak_time = -Inf
        while isinf(peak_time)
            peak_time = seir_hitting_time(pars, K, Z0, target_Z; tf = obs_t)
        end
        peak_timings[i] = peak_time
    end

    u0 = Z0 / K
    tspan = (0, obs_t)

    prob = ODEProblem(seir_deterministic!, u0, tspan, pars)
    sol = solve(prob, Tsit5(); saveat = 1e-5, abstol = 1e-9, reltol = 1e-9, save_idxs = 3)

    u_idx = findfirst(K * sol.u .> target_Z)
    peak_timing_det = sol.t[u_idx]
    time_delays_approx = peak_timing_det .- peak_timings

    return time_delays_approx
end

"""
    simulate_seir_timeshift(Z0, K, pars, results_dir, obs_t)
    
Simulate the SEIR model using the time-shift methodology.  

Arguments: 
    Z0 = vector of initial conditions 
    K = population size 
    pars = vector of parameters for the SEIR model in form (R0, σ_inv, γ_inv)
    results_dir = where to save the samples 
    obs_t = maximum time 
    
Outputs: 
    None
"""
function simulate_seir_timeshift(Z0, K, pars, results_dir, obs_t)
    u0 = Z0 / K
    tspan = (0, 300)

    prob = ODEProblem(seir_deterministic!, u0, tspan, pars)
    prob = remake(prob; tspan = (0, 600))
    # Save on a much smaller grid to keep consistency.
    sol = solve(prob, Tsit5(); saveat = 0.0001, abstol = 1e-11, reltol = 1e-11)

    times = collect(0:0.01:obs_t)

    df_τ = CSV.read(joinpath(results_dir, "tau_quantiles.csv"), DataFrame)

    intervals = ["quantiles_lower", "medians", "quantiles_upper"]

    τ_test = [df_τ[df_τ.methods .== "PEM", q] for q in intervals]

    samples_det = zeros(length(times), length(τ_test))

    for (i, τ) in enumerate(τ_test)
        times_shifted = times .+ τ

        if any(times_shifted .> 0)
            sol = solve(prob,
                        Tsit5();
                        saveat = times_shifted[times_shifted .>= 0],
                        abstol = 1e-15,
                        reltol = 1e-15,
                        save_idxs = 3)

            samples_det[times_shifted .>= 0, i] .= K * (sol.u)
            samples_det[times_shifted .< 0, i] .= Z0[3]
        else
            samples_det[times_shifted .< 0, i] .= Z0[3]
        end
    end

    samples_det[samples_det .< 0] .= 0

    df_samples_det = DataFrame(max.(0, log.(samples_det)), :auto)
    df_samples_det.t = times
    CSV.write(joinpath(results_dir, "det_samples.csv"), df_samples_det)

    return nothing
end

"""
    run_initial_condition_experiment()
    
Run the experiment for the initial conditions. 
"""
function run_initial_condition_experiment()
    figures_dir, results_dir = make_dirs("SEIR")
    Random.seed!(2023)

    # Fixed model parameters
    R0 = 1.7
    σ_inv = 2.0
    γ_inv = 3.0
    pars = [R0, σ_inv, γ_inv]
    K = Int(10^6)

    # Initial condition vectors 
    E0_v = [1, 5, 5, 15]
    I0_v = [0, 0, 5, 10]
    Z0s = [[K - (e + i), e, i] for (e, i) in zip(E0_v, I0_v)]
    num_moments = 20

    for (id, Z0) in enumerate(Z0s)
        Z0_bp = Z0[2:end]

        t_range, pdf_vals, cdf_vals = compute_time_shift_distribution(pars, Z0_bp,
                                                                      num_moments)
        df_pdf = DataFrame(t = t_range, p = pdf_vals)
        CSV.write(joinpath(results_dir, "pdf_vals_Z0_$id.csv"), df_pdf)

        # Estimate the distribution of the timeshifts using Gillespie simulation
        time_delays_approx = estimate_time_shifts(pars, Z0, K)

        # Estimate the distribution of timeshifts through m3 method
        moments = calculate_moments(pars, num_moments)
        q1 = calculate_extinction_probs(pars)
        pars_m3 = RandomTimeShifts.minimise_loss(moments, q1)
        W_samples_m3 = RandomTimeShifts.sample_W(100000, pars_m3, q1, Z0_bp)

        df_pars_m3 = DataFrame(stack(pars_m3), :auto)
        CSV.write(joinpath(results_dir, "pars_m3_$id.csv"), df_pars_m3)

        # Calculate BP stuff required for getting the timeshifts using simulation and m3
        β, σ, γ = epi_pars_to_sim_pars(pars)
        Ω = [-σ σ
             β -γ]

        λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)
        EW = sum(Z0_bp .* u_norm)

        time_delays_m3 = timeshifts_from_W(W_samples_m3, EW, λ1)

        df_times = DataFrame(method = "approx", time_delay = time_delays_approx)
        df_times = [df_times; DataFrame(method = "m3", time_delay = time_delays_m3)]
        CSV.write(joinpath(results_dir, "time_delays_Z0_$id.csv"), df_times)
    end

    return nothing
end

"""
    run_system_size_experiment()
    
Run the experiment for the size of the system. 
"""
function run_system_size_experiment()
    figures_dir, results_dir = make_dirs("SEIR")
    Random.seed!(2023)

    # Fixed model parameters
    R0 = 1.7
    σ_inv = 2.0
    γ_inv = 3.0
    pars = [R0, σ_inv, γ_inv]
    Ks = [10^x for x in 3:6]

    # Initial condition vectors 
    E0 = 1
    I0 = 0
    num_moments = 20

    for (id, K) in enumerate(Ks)
        Z0 = [K - E0 - I0, E0, I0]
        Z0_bp = Z0[2:end]

        t_range, pdf_vals, cdf_vals = compute_time_shift_distribution(pars, Z0_bp,
                                                                      num_moments)
        df_pdf = DataFrame(t = t_range, p = pdf_vals)
        CSV.write(joinpath(results_dir, "pdf_vals_N_$id.csv"), df_pdf)

        # Estimate the distribution of the timeshifts using Gillespie simulation
        time_delays_approx = estimate_time_shifts(pars, Z0, K)

        # Estimate the distribution of timeshifts through m3 method
        moments = calculate_moments(pars, num_moments)
        q1 = calculate_extinction_probs(pars)
        pars_m3 = RandomTimeShifts.minimise_loss(moments, q1)
        W_samples_m3 = RandomTimeShifts.sample_W(100000, pars_m3, q1, Z0_bp)

        df_pars_m3 = DataFrame(stack(pars_m3), :auto)
        CSV.write(joinpath(results_dir, "pars_m3_N_$id.csv"), df_pars_m3)

        # Calculate BP stuff required for getting the timeshifts using simulation and m3
        β, σ, γ = epi_pars_to_sim_pars(pars)
        Ω = [-σ σ
             β -γ]

        λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)
        EW = sum(Z0_bp .* u_norm)

        time_delays_m3 = timeshifts_from_W(W_samples_m3, EW, λ1)

        df_times = DataFrame(method = "approx", time_delay = time_delays_approx)
        df_times = [df_times; DataFrame(method = "m3", time_delay = time_delays_m3)]
        CSV.write(joinpath(results_dir, "time_delays_N_$id.csv"), df_times)
    end

    return nothing
end

"""
    run_m3_timeshift_distribution_control()
    
Run the experiment for the control which involves saving the estimated time-shift distribution
through simulation and through the M3 method.
"""
function run_m3_timeshift_distribution_control()
    figures_dir, results_dir = make_dirs("SEIR")
    Random.seed!(2023)

    # Fixed model parameters
    R0 = 1.7
    σ_inv = 2.0
    γ_inv = 3.0
    pars = [R0, σ_inv, γ_inv]
    K = Int(10^6)

    # Initial condition is fixed
    Z0 = [K - 1, 1, 0]
    Z0_bp = Z0[2:end]
    # Estimate the distribution of the timeshifts using Gillespie simulation
    time_delays_approx = estimate_time_shifts(pars, Z0, K)

    # Require 5 moments for the surrogate approach 
    num_moments = 5
    # Estimate the distribution of timeshifts through m3 method
    moments = calculate_moments(pars, num_moments)
    q1 = calculate_extinction_probs(pars)
    pars_m3 = RandomTimeShifts.minimise_loss(moments, q1)
    W_samples_m3 = RandomTimeShifts.sample_W(100000, pars_m3, q1, Z0_bp)

    # Calculate BP stuff required for getting the timeshifts using simulation and m3
    β, σ, γ = epi_pars_to_sim_pars(pars)
    Ω = [-σ σ
         β -γ]

    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)
    EW = sum(Z0_bp .* u_norm)

    time_delays_m3 = timeshifts_from_W(W_samples_m3, EW, λ1)

    df_times = DataFrame(method = "approx", time_delay = time_delays_approx)
    df_times = [df_times; DataFrame(method = "m3", time_delay = time_delays_m3)]
    CSV.write(joinpath(results_dir, "time_delays_control.csv"), df_times)

    return nothing
end

"""
    run_num_moment_experiment()
    
Run number of moments experiment. 
"""
function run_num_moment_experiment()
    figures_dir, results_dir = make_dirs("SEIR")
    Random.seed!(2023)

    # Fixed model parameters
    R0 = 1.7
    σ_inv = 2.0
    γ_inv = 3.0
    pars = [R0, σ_inv, γ_inv]
    K = Int(10^6)

    # Initial condition is fixed
    Z0 = [K - 1, 1, 0]
    Z0_bp = Z0[2:end]
    # Explore sensitivity to number of moments
    num_moments_test = [3, 5, 10, 15, 30, 60]

    for (id, num_moments) in enumerate(num_moments_test)
        t_range, pdf_vals, cdf_vals = compute_time_shift_distribution(pars, Z0_bp,
                                                                      num_moments)
        df_pdf = DataFrame(t = t_range, pdf = pdf_vals, cdf = cdf_vals)
        CSV.write(joinpath(results_dir, "pdf_vals_moments_$id.csv"), df_pdf)
    end

    return nothing
end

"""
    run_h_experiment(do_benchmark)
    
Run the experiment for the h step sizes. 
"""
function run_h_experiment()
    figures_dir, results_dir = make_dirs("SEIR")
    Random.seed!(2023)

    # Fixed model parameters
    R0 = 1.7
    σ_inv = 2.0
    γ_inv = 3.0
    pars = [R0, σ_inv, γ_inv]
    K = Int(1e6)

    # Initial condition is fixed
    Z0 = [K - 1, 1, 0]
    Z0_bp = Z0[2:end]
    # Explore sensitivity to number of moments
    num_moments = 30

    h_test = [0.1, 0.5, 1.0, 5.0]

    for (id, h) in enumerate(h_test)
        t_range, pdf_vals, cdf_vals = compute_time_shift_distribution(pars, Z0_bp,
                                                                      num_moments; h = h)
        df_pdf = DataFrame(t = t_range, pdf = pdf_vals, cdf = cdf_vals)
        CSV.write(joinpath(results_dir, "pdf_vals_h_$id.csv"), df_pdf)
    end

    return nothing
end

"""
    preprocess_trial(t::BenchmarkTools.Trial, id::AbstractString; num_repeats = 1)
  
Processes benchmark results to a nicer format. 
    
Arguments: 
    t = output from @benchmark 
    id = id for the current trial 
    num_repeats = number of times the function was evaluated (if needed to be wrapped in a loop )
    
Outputs: 
    Array of details for the trial.
"""
function preprocess_trial(t::BenchmarkTools.Trial, id::AbstractString; num_repeats = 1)
    nanoseconds_to_seconds = 1e-9
    conversion_factor = 1e4
    times_s = conversion_factor * nanoseconds_to_seconds * (t.times / num_repeats)
    return (h_value = id,
            median = median(times_s),
            mean = mean(times_s),
            std = std(times_s),
            minimum = minimum(times_s),
            maximum = maximum(times_s),
            allocations = ceil(Int, t.allocs / num_repeats),
            memory_estimate = ceil(Int, t.memory / num_repeats))
end

function run_h_benchmarks()
    figures_dir, results_dir = make_dirs("SEIR")
    Random.seed!(2023)

    # Fixed model parameters
    R0 = 1.7
    σ_inv = 2.0
    γ_inv = 3.0
    pars = [R0, σ_inv, γ_inv]
    K = Int(1e6)

    # Initial condition is fixed
    Z0 = [K - 1, 1, 0]
    Z0_bp = Z0[2:end]
    # Explore sensitivity to number of moments
    num_moments = 30

    # Grid for h-values to explore
    h_test = [0.1, 0.5, 1.0, 5.0]

    b1 = Dict()
    b2 = Dict()

    num_repeats = 100
    s_vals = rand(Normal(0, 10), num_repeats) + rand(Normal(0, 10), num_repeats) * im
    w_vals = rand(Uniform(0, 10), num_repeats)

    for h in h_test
        println("Running benchmarks for h = $h...")

        # Calculate all preamble stuff and don't include in the timings 
        β, σ, γ = epi_pars_to_sim_pars(pars)
        Ω = [-σ σ
             β -γ]

        λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)

        moments = calculate_moments(pars, num_moments + 1)
        error_moment = moments[end, :]
        moments = moments[begin:(end - 1), :]
        L = RandomTimeShifts.error_bounds(1e-6, error_moment, num_moments)

        prob = ODEProblem(F_fixed_s_ode!, [0.0, 0.0], (0, h), pars,
                          saveat = h, save_start = false)
        F_offspring(s) = RandomTimeShifts.F_offspring_ode(s, prob)
        coeffs = RandomTimeShifts.moment_coeffs(moments)

        μ = exp(h * λ1)
        lst_w = RandomTimeShifts.construct_lst(coeffs, μ, F_offspring, L,
                                               Z0_bp, λ1, h)

        q1 = calculate_extinction_probs(pars)
        q_star = prod(q1 .^ Z0_bp)
        W_cdf_ilst = RandomTimeShifts.construct_W_cdf_ilst(lst_w, q_star)

        # Benchmark the LST evaluations and the CDF evaluations
        b1[h] = @benchmark begin for s in $s_vals
            $lst_w(s)
        end end
        b2[h] = @benchmark begin for w in $w_vals
            $W_cdf_ilst(w)
        end end
    end

    lst_benchmark = DataFrame()
    cdf_benchmark = DataFrame()

    for h in h_test
        push!(lst_benchmark, preprocess_trial(b1[h], "$h", num_repeats = 100))
        push!(cdf_benchmark, preprocess_trial(b2[h], "$h", num_repeats = 100))
    end

    CSV.write(joinpath(results_dir, "lst_benchmark.csv"), lst_benchmark)
    CSV.write(joinpath(results_dir, "cdf_benchmark.csv"), cdf_benchmark)

    return nothing
end