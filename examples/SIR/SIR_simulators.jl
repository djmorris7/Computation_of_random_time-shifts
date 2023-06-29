using Random
using Distributions
using StaticArrays

"""
    sir_exact_gillespie(pars, K, Z0; tf = 100.0, save_at = 1.0)

Simulate a Susceptible-Infected-Recovered (SIR) epidemic model with exact stochastic simulation algorithm
using the Gillespie algorithm. The model is parameterized with the basic reproduction number (R0),
the average infectious period (γ_inv), and the total population size (K), and is initialized with
an initial state of (S, I, R) = Z0. The simulation runs for a time period of tf, with outputs saved
every save_at time steps.

Arguments: 
    pars = parameters of the SIR model (R0, γ_inv)
    K = population size
    Z0 = intial conditions for the SIR model 
    tf = max simulation time
    save_at = how frequently to save the simulation
    
Outputs:
    t_vec = simulation times 
    Z_mat = Output matrix 
    extinct = whether the process went extinct or not 
"""
function sir_exact_gillespie(pars, K, Z0; tf = 100.0, save_at = 1.0)
    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β_norm = R0 * γ / (K - 1)

    Z = deepcopy(Z0)
    # Calculate the actual number of steps which is the total simulation 
    # divided by the step size + the first time point
    n = ceil(Int, tf / save_at) + 1

    # Initialising variables 
    # Z_mat = zeros(Int, n, 3)
    Z_mat = zeros(Int, 3 * n)

    t_vec = zeros(Float64, n)
    # Propensity vector 
    a = zeros(Float64, 2)
    # Representation of the stoichiometry matrix
    Q = SA[[-1, 1, 0], [0, -1, 1]]

    # Store initial state
    # Z_mat[1, :] .= Z0
    Z_mat[1:length(Z0)] .= Z0

    # Track current time 
    t = 0.0
    # Track the current observation window [curr_t - save_at, curr_t)
    curr_t = save_at
    # Pointer to the element of the output arrays. 
    curr_ind = 2

    @inbounds while (t < tf) & (Z[2] > 0)
        (S, I, R) = Z
        a[1] = β_norm * I * S
        a[2] = γ * I

        a0 = sum(a)
        dt = -log(rand()) / a0
        t = t + dt

        while t > curr_t && curr_ind <= n
            t_vec[curr_ind] = curr_t
            Z_mat[(3 * (curr_ind - 1) + 1):(3 * curr_ind)] .= Z
            curr_ind += 1
            curr_t += save_at
        end

        # Determine next event via the next-reaction method 
        ru = rand() * a0
        cumsum!(a, a)
        event_idx = searchsortedfirst(a, ru)

        for i in eachindex(Z)
            Z[i] += Q[event_idx][i]
        end
    end

    # Flag for whether the simulation went extinct over [0, tf)
    extinct = Z_mat[end - 1] == 0

    return t_vec, Z_mat, extinct
end

"""
    sir_hitting_times(pars, K, Z0, target_Z; tf = 100.0)

Estimate the time taken for the infectious population in an Susceptible-Infected-Recovered (SIR) epidemic model 
to exceed target_Z using the Gillespie algorithm. The model is parameterized with the basic reproduction number (R0),
the average infectious period (γ_inv), and the total population size (K), and is initialized with
an initial state of (S, I, R) = Z0. 

Arguments: 
    pars = parameters of the SIR model (R0, γ_inv)
    K = population size
    Z0 = intial conditions for the SIR model 
    target_Z = the target number of infected to hit to estimate the hitting time
    tf = max simulation time
    
Outputs:
    t = the time taken to reach I >= target_Z 
"""
function sir_hitting_times(pars, K, Z0, target_Z; tf = 100.0)
    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β_norm = R0 * γ / (K - 1)

    Z = deepcopy(Z0)

    # Initialising variables 
    # Propensity vector 
    a = zeros(Float32, 2)

    # Track current time 
    t = 0.0

    @inbounds while (t < tf) && (Z[2] > 0)
        (S, I, R) = Z
        a[1] = β_norm * I * S
        a[2] = γ * I

        a0 = sum(a)
        dt = -log(rand()) / a0
        t = t + dt

        # Determine next event via the next-reaction method 
        ru = rand() * a0
        if a[1] > ru
            Z[1] -= 1
            Z[2] += 1
        else
            Z[2] -= 1
        end

        if Z[2] > target_Z
            return t
        end
    end

    return -Inf
end

"""
    sir_peak_times(pars, K, Z0, τ; tf = 100.0)

Estimate the time taken for the infectious population to peak in an Susceptible-Infected-Recovered (SIR) epidemic model 
using a hybrid Gillespie / Tau-leap algorithm. The model is parameterized with the basic reproduction number (R0),
the average infectious period (γ_inv), and the total population size (K), and is initialized with
an initial state of (S, I, R) = Z0.

Arguments: 
    pars = parameters of the SIR model (R0, γ_inv)
    K = population size
    Z0 = intial conditions for the SIR model 
    τ = step size for the tau-leaping
    tf = max simulation time
    
Outputs:
    peak_time = the timing of the peak
"""
function sir_peak_times(pars, K, Z0, τ; tf = 100.0)
    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β_norm = R0 * γ / (K - 1)

    Z = deepcopy(Z0)

    # Initialising variables 
    # Propensity vector 
    a = zeros(Float32, 2)

    # Track current time 
    t = 0.0

    max_Z = Z[2]
    peak_time = -Inf

    @inbounds while (t < tf) & (Z[2] > 0)
        (S, I, R) = Z
        a[1] = β_norm * I * S
        a[2] = γ * I

        if Z[2] < 1000
            a0 = sum(a)
            dt = -log(rand()) / a0
            t = t + dt

            # Determine next event via the next-reaction method 
            ru = rand() * a0
            if a[1] > ru
                Z[1] -= 1
                Z[2] += 1
            else
                Z[2] -= 1
            end
        else
            t += τ
            e1 = rand(Poisson(τ * a[1]))
            e2 = rand(Poisson(τ * a[2]))
            Z[1] = Z[1] - e1
            Z[2] = Z[2] + e1 - e2
        end

        if Z[2] > max_Z
            max_Z = Z[2]
            peak_time = t
        end
    end

    return peak_time
end

"""
    sir_bp_gillespie_hybrid(pars, I0, τ; tf = 100.0, save_at = 1.0)

Simulate a branching process approximation to the Susceptible-Infected-Recovered (SIR) epidemic model 
using a hybrid Gillespie / Tau-leap algorithm. The model is parameterized with the basic reproduction number (R0),
the average infectious period (γ_inv), and the total population size (K), and is initialized with
an initial state of (S, I, R) = Z0. The simulation runs for a time period of tf, with outputs saved
every save_at time steps.

Arguments: 
    pars = parameters of the SIR model (R0, γ_inv)
    I0 = the initial number of infected individuals
    τ = step size for the tau-leaping
    tf = max simulation time
    save_at = how frequently to save the simulation
    
Outputs:
    t_vec = simulation times 
    I_mat = output array for number of infected at times given by t_vec 
    extinct = whether the process went extinct or not 
"""
function sir_bp_gillespie_hybrid(pars, I0, τ; tf = 100.0, save_at = 1.0)
    (R0, γ_inv) = pars
    γ = 1 / γ_inv
    β = R0 * γ

    I = deepcopy(I0)
    # Calculate the actual number of steps which is the total simulation 
    # divided by the step size + the first time point
    n = ceil(Int, tf / save_at) + 1

    # Initialising variables 
    # Z_mat = zeros(Int, n, 3)
    I_mat = zeros(Int, n)

    t_vec = zeros(Float64, n)
    # Propensity vector 
    a = zeros(Float64, 2)

    # Store initial state
    I_mat[1] = I0

    # Track current time 
    t = 0.0
    # Track the current observation window [curr_t - save_at, curr_t)
    curr_t = save_at
    # Pointer to the element of the output arrays. 
    curr_ind = 2

    @inbounds while (t < tf) & (I > 0)
        a[1] = β * I
        a[2] = γ * I

        if I < 100
            a0 = sum(a)
            dt = -log(rand()) / a0
            t = t + dt

            while t > curr_t && curr_ind <= n
                t_vec[curr_ind] = curr_t
                I_mat[curr_ind] = I
                curr_ind += 1
                curr_t += save_at
            end

            # Determine next event via the next-reaction method 
            ru = rand() * a0
            cumsum!(a, a)
            event_idx = searchsortedfirst(a, ru)

            if event_idx == 1
                I += 1
            else
                I -= 1
            end
        else
            t += τ
            while t > curr_t && curr_ind <= n
                t_vec[curr_ind] = curr_t
                I_mat[curr_ind] = I
                curr_ind += 1
                curr_t += save_at
            end

            e1 = rand(Poisson(τ * a[1]))
            e2 = rand(Poisson(τ * a[2]))
            I += e1 - e2
        end
    end

    # Flag for whether the simulation went extinct over [0, tf)
    extinct = I_mat[end] == 0

    return t_vec, I_mat, extinct
end
