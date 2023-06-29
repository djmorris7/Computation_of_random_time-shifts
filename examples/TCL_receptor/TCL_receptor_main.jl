"""
TCL_receptor_main.jl 
    This script runs the code to produce the figures for the innate
    response model in the paper. 

Author: Dylan Morris
Date: 2023 
"""

include("TCL_receptor_W.jl")
include("TCL_receptor_simulators.jl")
function main()
    figures_dir, results_dir = make_dirs("TCL_receptor")

    # Set seed for reproducibility. 
    Random.seed!(2023)

    # Testing the gillespie simulator
    K = Int(8e7)

    β1 = 2.5e-8 * K
    β2 = 2.0e-8 * K
    σ = 4.0
    η = 1.0
    γ = 1.7
    p_v = 45.3
    c_v = 10.0
    p_a = 6.0
    c_a = 3.0
    ϱ = 1.3e-6 * K
    δ = 0.0044
    pars = [β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ]

    # X(t) = (U, R, E, I, V, F)
    Z0 = [K - 1, 0, 1, 0, 0, 0]

    # First we look at the agreement between the branching process approximation and the 
    # exact (gillespie) simulations.
    nsims = 20000
    # Run gillespie simulations over the time interval 0:obs_t - 1
    save_at = 0.01
    obs_t = 14.0

    t_out, Z_out, extinct = tcl_receptor_gillespie(pars, K, Z0; tf = obs_t,
                                                   save_at = save_at)

    Z_samples = stack([Z_out[i:length(Z0):end] for (i, z) in enumerate(Z0)])

    df_all_states = DataFrame([t_out Z_samples], [:time, :U, :R, :E, :I, :V, :A])
    CSV.write(joinpath(results_dir, "stoch_all_states.csv"), df_all_states)

    tf = obs_t
    t_range = collect(0:save_at:tf)
    # df_summ_exact = summarise_t(t_range, samples)

    u0 = Z0 / K
    prob = ODEProblem(tcl_receptor_deterministic!, u0, (0, 50), pars)
    # Save on a much smaller grid to keep consistency.
    sol = solve(prob, Tsit5(); saveat = 0.00001, abstol = 1e-11, reltol = 1e-11)

    ζ_samples = max.(0, stack(K * sol.u, dims = 1))

    τ = sol.t[findfirst(x -> x > 2000, ζ_samples[:, 5])] -
        t_range[findfirst(x -> x > 2000, Z_samples[:, 5])]

    println("Value of τ used is: $τ")

    times = collect(0:save_at:obs_t)

    samples_det = [zeros(length(times)) for _ in 1:6]

    times_shifted = times .+ τ

    if any(times_shifted .> 0)
        sol = solve(prob,
                    Tsit5(),
                    saveat = times_shifted[times_shifted .>= 0],
                    abstol = 1e-11,
                    reltol = 1e-11)

        for (i, s) in enumerate(samples_det)
            s[times_shifted .>= 0] .= K * [sol.u[t][i] for t in eachindex(sol.u)]
            s[times_shifted .< 0] .= Z0[i]
        end

    else
        for (i, s) in enumerate(samples_det)
            s[times_shifted .< 0] .= Z0[i]
        end
    end

    df_all_states = DataFrame([times stack(samples_det)], [:time, :U, :R, :E, :I, :V, :A])
    CSV.write(joinpath(results_dir, "det_all_states.csv"), df_all_states)

    ## Now do the timeshift stuff 

    Z0_bp = Z0[(begin + 2):(end - 1)]

    t_range, pdf_vals, cdf_vals = compute_time_shift_distribution(pars, Z0_bp, h = 0.1)
    df_pdf = DataFrame(t = t_range, p = pdf_vals)
    CSV.write(joinpath(results_dir, "pdf_vals.csv"), df_pdf)

    t_range, pdf_vals, cdf_vals = compute_time_shift_distribution(pars, Z0_bp, h = 1.0)
    df_pdf = DataFrame(t = t_range, p = pdf_vals)
    CSV.write(joinpath(results_dir, "pdf_vals_divergent.csv"), df_pdf)

    moments = calculate_moments(pars; num_moments = 5)
    q1 = calculate_extinction_probs(pars)
    pars_m3 = RandomTimeShifts.minimise_loss(moments, q1)
    W_samples_m3 = RandomTimeShifts.sample_W(100000, pars_m3, q1, Z0_bp)

    β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ = pars
    Ω = [-(σ + η) σ 0
         β2 -γ p_v
         β1 0 -(c_v + β1)]

    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)
    EW = sum(Z0_bp .* u_norm)

    time_delays_m3 = timeshifts_from_W(W_samples_m3, EW, λ1)

    time_delays_approx = estimate_time_shifts(pars, Z0, K)

    df_times = DataFrame(method = "approx", time_delay = time_delays_approx)
    df_times = [df_times; DataFrame(method = "m3", time_delay = time_delays_m3)]

    CSV.write(joinpath(results_dir, "time_delays.csv"), df_times)

    return nothing
end

main()