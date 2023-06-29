using Random
using Distributions

##

"""
    tcl_receptor_gillespie(pars::Vector{Float64},
                           K::Int,
                           Z0::Vector{Int};
                           tf::Float64 = 100.0,
                           save_at::Float64 = 1.0,
                           τ::Float64 = 0.01)

Simulates the innate response model from the paper using the SSA and tau-leaping.

Arguments: 
    pars::Vector{Float64} = vector of parameters for the innate response model
    K::Int = population size 
    Z0::Vector{Int} = vector of initial conditions 
    tf::Float64 = maximum simulation time
    save_at = when to save the simulation 
    τ = step size for tau leaping
    
    
Outputs: 
    t_vec = simulation times 
    Z_mat = Output matrix 
    extinct = whether the process went extinct or not 
"""
function tcl_receptor_gillespie(pars::Vector{Float64},
                                K::Int,
                                Z0::Vector{Int};
                                tf::Float64 = 100.0,
                                save_at::Float64 = 1.0,
                                τ::Float64 = 0.01)
    (β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ) = pars
    β1_norm = β1 / K
    β2_norm = β2 / K
    ϱ_norm = ϱ / K

    Z = deepcopy(Z0)
    # Calculate the actual number of steps 
    n = ceil(Int, tf / save_at) + 1
    n_states = length(Z)
    Z_mat = zeros(Int, n * n_states)
    t_vec = zeros(Float64, n)
    # Propensity vector 
    a = zeros(Float64, 11)
    # Representation of the stoichiometry matrix
    # X(t) = (U, R, E, I, V, F)
    Q = [
        [-1, 0, 1, 0, -1, 0],        # (U, E, V) --> (U-1, E+1, V-1)
        [-1, 0, 1, 0, 0, 0],        # (U, E, I) --> (U-1, E+1, I)
        [-1, 1, 0, 0, 0, -1],         # (U, R, F) --> (U-1, R+1, F-1)
        [1, -1, 0, 0, 0, 0],         # (U, R) --> (U+1, R-1)
        [0, 0, -1, 1, 0, 0],         # (E, I) --> (E-1, I+1)
        [0, 0, -1, 0, 0, 0],         # (E) --> (E-1)
        [0, 0, 0, -1, 0, 0],         # (I) --> (I-1)
        [0, 0, 0, 0, 1, 0],         # (V) --> (V+1)
        [0, 0, 0, 0, -1, 0],         # (V) --> (V-1)
        [0, 0, 0, 0, 0, 1],          # (F) --> (F+1)
        [0, 0, 0, 0, 0, -1],          # (F) --> (F-1)
    ]

    # Store initial state
    Z_mat[1:n_states] .= Z0

    t = 0.0
    curr_t = save_at
    curr_ind = 2

    @inbounds while t < tf
        (U, R, E, I, V, F) = Z
        a[1] = β1_norm * U * V
        a[2] = β2_norm * U * I
        a[3] = ϱ_norm * U * F
        # Removal rate of infected cells 
        a[4] = δ * R
        # Production rate of viral particles
        a[5] = σ * E
        a[6] = η * E
        # Clearance rate of viral particles
        a[7] = γ * I
        a[8] = p_v * I
        a[9] = c_v * V
        a[10] = p_a * I
        a[11] = c_a * F

        # println("Z = $Z")

        if (E < 100) || (I < 100) || (V < 100) || (F < 100) || (R < 100)
            # if (E < 100) || (I < 100) || (V < 100)

            a0 = sum(a)
            dt = -log(rand()) / a0
            t = t + dt

            while t > curr_t && curr_ind <= n
                t_vec[curr_ind] = curr_t
                Z_mat[(n_states * (curr_ind - 1) + 1):(n_states * curr_ind)] .= Z
                curr_ind += 1
                curr_t += save_at
            end

            # Determine next event via the next-reaction method 
            ru = rand() * a0
            cumsum!(a, a)
            event_idx = findfirst(x -> x > ru, a)

            for i in eachindex(Z)
                Z[i] += Q[event_idx][i]
            end
        else
            t += τ
            while t > curr_t && curr_ind <= n
                t_vec[curr_ind] = curr_t
                Z_mat[(n_states * (curr_ind - 1) + 1):(n_states * curr_ind)] .= Z
                curr_ind += 1
                curr_t += save_at
            end

            for i in eachindex(Q)
                if τ * a[i] < 0
                    println("i = $i")
                    println("Z = $Z")
                end

                ne = rand(Poisson(τ * a[i]))
                for j in eachindex(Z)
                    Z[j] += ne * Q[i][j]
                end
            end
        end

        if (t > tf) || iszero(sum(Z[3:(end - 1)]))
            break
        end
    end

    extinct = iszero(sum(Z_mat[(end - 4):(end - 1)])) || iszero(sum(Z[3:(end - 1)]))

    return t_vec, Z_mat, extinct
end

"""
    tcl_receptor_hitting_time(pars::Vector{Float64},
                              K::Int,
                              Z0::Vector{Int},
                              target_Z::Int;
                              tf::Float64 = 100.0,
                              save_at::Float64 = 1.0,
                              τ::Float64 = 0.01)
                              
Simulates the innate response model using the SSA but focuses on hitting time to reach a particular viral 
load = target_Z. 

Arguments: 
    pars::Vector{Float64} = vector of parameters for the SEIR model in form (R0, σ_inv, γ_inv)
    K::Int = population size 
    Z0::Vector{Int} = vector of initial conditions 
    target_Z = target number of infected for hitting time calculation
    tf::Float64 = maximum simulation time
    τ = step size for tau-leaping
    
Outputs: 
    hitting_time = the hitting time
"""
function tcl_receptor_hitting_time(pars::Vector{Float64},
                                   K::Int,
                                   Z0::Vector{Int},
                                   target_Z::Int;
                                   tf::Float64 = 100.0,
                                   τ::Float64 = 0.01)
    (β1, β2, σ, η, γ, p_v, c_v, p_a, c_a, ϱ, δ) = pars
    β1_norm = β1 / K
    β2_norm = β2 / K
    ϱ_norm = ϱ / K

    Z = deepcopy(Z0)
    # Calculate the actual number of steps 
    n_states = length(Z)
    # Propensity vector 
    a = zeros(Float64, 11)
    # Representation of the stoichiometry matrix
    # X(t) = (U, R, E, I, V, F)
    Q = [
        [-1, 0, 1, 0, -1, 0],        # (U, E, V) --> (U-1, E+1, V-1)
        [-1, 0, 1, 0, 0, 0],        # (U, E, I) --> (U-1, E+1, I)
        [-1, 1, 0, 0, 0, -1],         # (U, R, F) --> (U-1, R+1, F-1)
        [1, -1, 0, 0, 0, 0],         # (U, R) --> (U+1, R-1)
        [0, 0, -1, 1, 0, 0],         # (E, I) --> (E-1, I+1)
        [0, 0, -1, 0, 0, 0],         # (E) --> (E-1)
        [0, 0, 0, -1, 0, 0],         # (I) --> (I-1)
        [0, 0, 0, 0, 1, 0],         # (V) --> (V+1)
        [0, 0, 0, 0, -1, 0],         # (V) --> (V-1)
        [0, 0, 0, 0, 0, 1],          # (F) --> (F+1)
        [0, 0, 0, 0, 0, -1],          # (F) --> (F-1)
    ]

    t = 0.0

    hitting_time = -Inf

    @inbounds while t < tf
        (U, R, E, I, V, F) = Z
        a[1] = β1_norm * U * V
        a[2] = β2_norm * U * I
        a[3] = ϱ_norm * U * F
        # Removal rate of infected cells 
        a[4] = δ * R
        # Production rate of viral particles
        a[5] = σ * E
        a[6] = η * E
        # Clearance rate of viral particles
        a[7] = γ * I
        a[8] = p_v * I
        a[9] = c_v * V
        a[10] = p_a * I
        a[11] = c_a * F

        # println("Z = $Z")

        if (E < 100) || (I < 100) || (V < 100) || (F < 100) || (R < 100)
            a0 = sum(a)
            dt = -log(rand()) / a0
            t = t + dt

            # Determine next event via the next-reaction method 
            ru = rand() * a0
            cumsum!(a, a)
            event_idx = findfirst(x -> x > ru, a)

            for i in eachindex(Z)
                Z[i] += Q[event_idx][i]
            end
        else
            t += τ

            for i in eachindex(Q)
                if τ * a[i] < 0
                    println("i = $i")
                    println("Z = $Z")
                end

                ne = rand(Poisson(τ * a[i]))
                for j in eachindex(Z)
                    Z[j] += ne * Q[i][j]
                end
            end
        end

        if Z[end - 1] > target_Z
            hitting_time = t
            break
        end

        if (t > tf) || iszero(sum(Z[3:(end - 1)]))
            break
        end
    end

    return hitting_time
end

"""
    tcl_receptor_deterministic!(dx, x, pars, t)
    
Evaluates the deterministic system for the innate response model at a given state x. 

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