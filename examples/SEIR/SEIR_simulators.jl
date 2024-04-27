using Random
using Distributions

##

"""
    seir_hitting_time(args; kwargs)

Simulate a Susceptible-Exposed-Infected-Recovered (SEIR) epidemic model using SSA to determine the hitting time for
a given number of infected.
The model is parameterized with the basic reproduction number (R0), average incubation period (σ_inv),
the average infectious period (γ_inv), and the total population size (K), and is initialized with
an initial state of (S, E, I, R) = Z0. The simulation runs for a time period of tf, with outputs saved
every save_at time steps. The parameter τ is used for tau leaping to improve simulation speed.

# Arguments
    - pars::Vector{Float64}: vector of parameters for the SEIR model in form (R0, σ_inv, γ_inv)
    - K::Int: population size
    - Z0::Vector{Int}: vector of initial conditions
    - target_Z: target number of infected for hitting time calculation
    - tf::Float64: maximum simulation time


# Outputs
    - t: the hitting time
"""
function seir_hitting_time(
    pars::Vector{Float64}, K::Int, Z0::Vector{Int}, target_Z; tf::Float64 = 100.0
)
    R0, σ_inv, γ_inv = pars
    σ = 1 / σ_inv
    γ = 1 / γ_inv
    β = R0 * γ
    β_norm = β / K

    Z = deepcopy(Z0)
    # Propensity vector
    a = zeros(Float64, 3)
    # Representation of the stoichiometry matrix
    Q = [
        [-1, 1, 0],      # (U, E, V) --> (U-1, E+1, V-1)
        [0, -1, 1],      # (U, E, V) --> (U-1, E+1, V-1)
        [0, 0, -1]      # (U, E, V) --> (U-1, E+1, V-1)
    ]

    t = 0.0

    @inbounds while (t < tf) & !iszero(sum(Z[2:3]))
        (S, E, I) = Z
        a[1] = β_norm * I * S
        a[2] = σ * E
        # Removal rate of infected cells
        a[3] = γ * I

        a0 = sum(a)
        dt = -log(rand()) / a0
        t = t + dt

        # Determine next event via the next-reaction method
        ru = rand() * a0
        cumsum!(a, a)
        event_idx = searchsortedfirst(a, ru)

        for i in eachindex(Z)
            Z[i] += Q[event_idx][i]
        end

        if Z[3] > target_Z
            return t
        end
    end

    return -Inf
end
