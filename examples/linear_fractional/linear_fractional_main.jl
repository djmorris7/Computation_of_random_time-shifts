include("linear_fractional_W.jl")

function main()
    figures_dir, results_dir = make_dirs("linear_fractional")

    pars = [0.3, 0.5]

    Z0 = 1
    T = 20
    pars = [0.3, 0.5]
    b, p = pars
    m = b / (1 - p)^2

    f(s) = 1 - b / (1 - p) + b * s / (1 - p * s)
    fq(s) = f(s) - s

    q = min(find_zero(fq, 1), find_zero(fq, 0))

    ϕ(s) = (q * (s - 1) + 1) / (-q + s + 1)
    g(w) = q * (w == 0) + (w > 0) * (1 - q)^2 * exp(-(1 - q) * w)
    g_star(w) = (1 - q) * exp(-(1 - q) * w)
    w = 0.01:0.01:10

    nsims = 1000
    w_samps = rand(Exponential(1 / (1 - q)), nsims)

    det_sol(n) = Z0 * m^n

    T = 50
    nsims = 10000
    Z_sims = [simulate_linear_fractional(pars, Z0, T) for _ in 1:nsims]
    w_samps = rand(Exponential(1 / (1 - q)), nsims)
    ζ_sims = [det_sol.((0:T) .+ log(w) / log(m)) for w in w_samps]

    Z_sims = [z for z in Z_sims if z[end] != 0]
    Z_mat = stack(Z_sims)
    ζ_sims = [z for z in ζ_sims if z[end] != 0]
    ζ_mat = stack(ζ_sims)

    CSV.write(joinpath(results_dir, "z_sims.csv"), DataFrame(Z_mat, :auto))
    CSV.write(joinpath(results_dir, "zeta_sims.csv"), DataFrame(ζ_mat, :auto))
end

main()