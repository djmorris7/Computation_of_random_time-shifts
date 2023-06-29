"""
    make_dirs(example_name)

Makes directories in the current project directory for figures and results based on the 
supplied name.     

Arguments: 
    example_name = Name of the example
    
Outputs:
    figures_dir = filepath to figures
    results_dir = filepath to results
"""
function make_dirs(example_name)
    figures_dir = "./examples/figures/"
    !ispath(figures_dir) && mkpath(figures_dir)
    results_dir = "./examples/results/$example_name/"
    !ispath(results_dir) && mkpath(results_dir)

    return figures_dir, results_dir
end

"""
    W_from_timeshifts(τ, EW, λ) 
    
Gets W samples from timeshifts. 

Arguments: 
    τ = vector of timeshifts 
    EW = expected value of W 
    λ = growth rate
    
Outputs: 
    Realisations of W 
"""
function W_from_timeshifts(τ, EW, λ)
    return exp.(λ * τ .+ log(EW))
end

"""
    timeshifts_from_W(W, EW, λ)
    
Gets time-shift samples from W. 

Arguments: 
    W = samples of W
    EW = expected value of W 
    λ = growth rate
    
Outputs: 
    Realisations of the time-shift 
"""
function timeshifts_from_W(W, EW, λ)
    return (log.(W) .- log(EW)) ./ λ
end