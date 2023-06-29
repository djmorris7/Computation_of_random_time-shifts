using DataFrames
using OrdinaryDiffEq
using Distributions
using KernelDensity
using ProgressMeter
using Random
using Statistics
using CSV

##

# Includes relative to this files location. 
include("./SIR_simulators.jl")
include("../../helper_functions.jl")

using Pkg
Pkg.develop(PackageSpec(path = "RandomTimeShifts.jl"))
using RandomTimeShifts

"""
    simulate_SIR_n_times(pars, Z0, K, nsims, obs_t; save_at = 1.0,
                         condition_on_non_extinction = true)

Simulate a Susceptible-Infected-Recovered (SIR) epidemic model nsims times using the Gillespie algorithm. 
The model is parameterized with the basic reproduction number (R0), the average infectious period (γ_inv), 
and the total population size (K), and is initialized with
an initial state of (S, I, R) = Z0. The simulation runs for a time period of tf, with outputs saved
every save_at time steps. 
"""
function simulate_SIR_n_times(pars, Z0, K, nsims, obs_t; save_at = 1.0,
                              condition_on_non_extinction = true)
    samples = zeros(Int, ceil(Int, obs_t / save_at) + 1, nsims)
    i = 1
    while i <= nsims
        t_out, Z_out, extinct = sir_exact_gillespie(pars, K, Z0; tf = obs_t,
                                                    save_at = save_at)

        if (condition_on_non_extinction & !extinct) | (!condition_on_non_extinction)
            samples[:, i] .= @view Z_out[2:3:end]
            i += 1
        end
    end

    return samples
end

"""
    sir_deterministic!(du, u, pars, t)
    
Evaluate the system of ordinary differential equations for the SIR model with parameters 
pars = (R0, γ_inv).
"""
function sir_deterministic!(du, u, pars, t)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ

    (s, i, i_log) = u

    du[1] = ds = -β * i * s
    du[2] = di = β * i * s - γ * i
    du[3] = di_log = β * s - γ

    return nothing
end

"""
    log_det_linear(x0, y0, x, λ)
Evaluates the log-det linear relationship for a given point (x0, y0) on the line with slope λ. 
Used to generate the figure in the SIR example. 
"""
function log_det_linear(x0, y0, x, λ)
    c = y0 - x0 * λ
    return λ * x + c
end

"""
    simulate_intro_data(pars, Z0, K, results_dir)
    
Simulates the introductory data for the SIR example (the first figures) from both 
the deterministic and stochastic models. 
"""
function simulate_intro_data(pars, Z0, K, results_dir)
    nsims = 5
    obs_t = 80
    Δt = 0.1

    condition_on_non_extinction = true
    samples = simulate_SIR_n_times(pars,
                                   Z0,
                                   K,
                                   nsims,
                                   obs_t;
                                   save_at = Δt,
                                   condition_on_non_extinction = condition_on_non_extinction)

    u0 = Z0 / K
    u0[3] = log(u0[2])
    tspan = (0, obs_t)

    prob = ODEProblem(sir_deterministic!, u0, tspan, pars)
    sol = solve(prob, Tsit5(); saveat = Δt, abstol = 1e-9, reltol = 1e-9, save_idxs = 3)

    t_range = 0:Δt:obs_t

    df_tmp = DataFrame(samples, :auto)
    df_tmp.t .= t_range

    CSV.write(joinpath(results_dir, "det_samples.csv"),
              DataFrame(; t = sol.t, I = K * exp.(sol.u)))
    CSV.write(joinpath(results_dir, "stoch_samples.csv"), df_tmp)
end

"""
    simulate_peak_timings(pars, Z0, K, results_dir)
    
This simulates peak timings using SSA and saves them.
"""
function simulate_peak_timings(pars, Z0, K, results_dir)
    obs_t = 70
    n_peaks = 100000
    peak_timings = zeros(Float64, n_peaks)
    τ = 0.01

    @showprogress for i in eachindex(peak_timings)
        peak_time = -Inf
        while isinf(peak_time)
            peak_time = sir_peak_times(pars, K, Z0, τ; tf = obs_t)
            peak_time = ifelse(peak_time < 20, Inf, peak_time)
        end
        peak_timings[i] = peak_time
    end

    CSV.write(joinpath(results_dir, "peak_timings.csv"),
              DataFrame(; peak_timings = peak_timings))
end

"""
    simulate_tau_mapping(pars, Z0, K, results_dir)
    
Simulates the data for the mapping between W(20), t_intercept and tau. 
"""
function simulate_tau_mapping(pars, Z0, K, results_dir)
    t_range = 0:50
    τ = 0.01

    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β = R0 * γ
    λ = β - γ
    save_at = 0.01
    t_range = 0:save_at:50
    W_scaling = exp.(-t_range * λ)

    nsims = 50000
    bp_samples = zeros(Int, ceil(Int, ceil(Int, t_range[end] / save_at) + 1), nsims)
    W_samples = zeros(Float64, ceil(Int, ceil(Int, t_range[end] / save_at) + 1), nsims)

    j = 1
    while j <= nsims
        t, Z, extinct = sir_bp_gillespie_hybrid(pars, 1, τ; tf = t_range[end],
                                                save_at = save_at)
        if !extinct
            bp_samples[:, j] .= Z
            W_samples[:, j] .= Z .* W_scaling
            j += 1
        end
    end

    I_target = 2000
    hitting_times = [ifelse(any(c .> I_target), t_range[findfirst(c .> I_target)], -Inf)
                     for c in eachcol(bp_samples)]

    quantiles = [0.025, 0.5, 0.975]
    hitting_times_to_plot = [
        quantile(hitting_times, quantiles[1]),
        median(hitting_times),
        quantile(hitting_times, quantiles[3]),
    ]
    hitting_time_intervals = []
    for (i, h) in enumerate(hitting_times_to_plot)
        lower = h - 0.1
        upper = h + 0.1
        push!(hitting_time_intervals, (lower, upper))
    end

    hitting_time_idx = [findfirst((hitting_times .>= hitting_time_intervals[i][1]) .&&
                                  (hitting_times .<= hitting_time_intervals[i][2]))
                        for i in eachindex(hitting_time_intervals)]

    slopes = diff(log.(bp_samples[:, hitting_time_idx]); dims = 1) / diff(t_range)[1]
    moving_avg_slopes = cumsum(slopes; dims = 1) ./ (1:size(slopes, 1))

    t_prime = 20

    t_range_lin = -10:0.1:(t_prime + 10)
    xs = hitting_times[hitting_time_idx]
    ys = [log(bp_samples[findfirst(t_range .>= x), i])
          for (x, i) in zip(xs, hitting_time_idx)]

    df = DataFrame()

    for (x, y, idx, q) in zip(xs, ys, hitting_time_idx, quantiles)
        df_tmp = DataFrame(;
                           t = t_range_lin,
                           y = log_det_linear.(x, y, t_range_lin, λ),
                           x0 = x,
                           y0 = y,
                           type = "proj_$(q*100)")
        df = [df; df_tmp]

        df_tmp = DataFrame(t = t_range[t_range .<= t_prime + 5],
                           y = log.(bp_samples[t_range .<= t_prime + 5, idx]),
                           x0 = missing,
                           y0 = missing,
                           type = "$(q*100)")

        df = [df; df_tmp]
    end

    CSV.write(joinpath(results_dir, "tau_mapping.csv"), df)

    I_target = 2000
    hitting_times = [ifelse(any(c .> I_target), t_range[findfirst(c .> I_target)], -Inf)
                     for
                     c in eachcol(bp_samples)]

    slopes = diff(log.(bp_samples); dims = 1) / diff(t_range)[1]
    moving_avg_slopes = cumsum(slopes; dims = 1) ./ (1:size(slopes, 1))

    t_range_lin = -10:0.1:30
    xs = hitting_times
    ys = [log(bp_samples[findfirst(t_range .>= x), i]) for (i, x) in enumerate(xs)]

    df = DataFrame()
    log_state = vec(log.(bp_samples[t_range .== t_prime, :]))
    xs = hitting_times

    for (idx, (x, y)) in enumerate(zip(xs, log_state))
        df_tmp = DataFrame(; tau = t_prime - λ^(-1) * y, y = y, ID = idx)
        df = [df; df_tmp]
    end

    CSV.write(joinpath(results_dir, "tau_log_state.csv"), df)
end

"""
    simulate_W_samples(pars, Z0, K, results_dir)
    
Draws samples of W from the distribution specified in Barbour (Exp(1 - q)). 
"""
function draw_W_samples(pars, Z0, K, results_dir)
    nsims = 100000
    obs_t = 10

    run_ssa = true
    condition_on_non_extinction = true

    if !isfile(joinpath(results_dir, "exact_samples.csv")) || run_ssa
        samples = simulate_SIR_n_times(pars, Z0, K, nsims, obs_t;
                                       condition_on_non_extinction = condition_on_non_extinction)
    else
        samples = Matrix(CSV.read(joinpath(results_dir, "exact_samples.csv"), DataFrame))
    end

    tf = Float64(obs_t)
    df_summ_exact = summarise_t(collect(0:tf), samples)

    u0 = Z0 / K
    tspan = (0, 100)
    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β = R0 * γ

    prob = ODEProblem(sir_deterministic!, u0, tspan, pars)
    sol = solve(prob, Tsit5(); saveat = 0.0001, abstol = 1e-13, reltol = 1e-13,
                save_idxs = 3)
    μ_w = 1
    r = β - γ

    n_samps = 100_000
    w = zeros(n_samps)
    t_shift = zeros(n_samps)

    for i in 1:n_samps
        w[i] = rand(Exponential(1 / (1 - 1 / R0)))
        t_shift[i] = 1 / r * (log(w[i]) - log(μ_w))
    end

    times = collect(0:obs_t)
    times_shifted = zeros(Float64, length(times))

    @showprogress for i in 1:nsims
        # The distribution of W is a unit Exponential following the 
        # arguments in Barbour et al. renormalised to account for 
        # removal of the point mass at 0. 

        if condition_on_non_extinction
            faded_out = false
        else
            faded_out = rand(Bernoulli(1 / R0))
        end

        if !faded_out
            w = rand(Exponential(1 / (1 - 1 / R0)))
            t_shift = 1 / r * (log(w) - log(μ_w))
            @. times_shifted = times + t_shift

            if any(x -> x > 0, times_shifted)
                sol = solve(prob,
                            Tsit5();
                            saveat = times_shifted[times_shifted .> 0],
                            abstol = 1e-12,
                            reltol = 1e-12,
                            save_idxs = 3)

                samples[times_shifted .<= 0, i] .= Z0[2]
                samples[times_shifted .> 0, i] .= round.(Int, exp.(log(K) .+ sol.u))
            else
                samples[times_shifted .<= 0, i] .= Z0[2]
            end
        end
    end

    tf = Float64(obs_t)
    df_summ_shift = summarise_t(collect(0:tf), samples)

    df_summ_exact.method .= "ctmc"
    df_summ_shift.method .= "timeshift"
    df_summ_comb = [df_summ_exact; df_summ_shift]

    CSV.write(joinpath(results_dir, "df_summ_both.csv"), df_summ_comb)
end