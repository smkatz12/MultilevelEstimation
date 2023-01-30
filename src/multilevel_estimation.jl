using GridInterpolations
using ProgressBars

# General problem
mutable struct GriddedProblem
    grid_points::Dict # Dictionary of grid cutpoints
    grid::RectangleGrid # Grid of points
    sim::Function # Function that takes in parameter set and runs n sims
    pfail_threshold::Float64 # probability of failure threshold for determining if safe
    conf_threshold::Float64 # desired confidence
    is_safe::Vector # Vector of booleans (matching grid indices)
end

# Models for estimating safe set
abstract type SetEstimationModel end

function run_estimation!(model::SetEstimationModel, problem::GriddedProblem, acquisition, nsamps;
    tuple_return=false, update_kernel_every=Inf)
    set_sizes = tuple_return ? [(0, 0)] : [0]
    neval = convert(Int64, floor(nsamps / model.nsamps))

    for i in ProgressBar(1:neval)
        # Select next point
        sample_ind = acquisition(model)
        params = to_params(model, sample_ind)

        # Evaluate
        res = problem.sim(params, model.nsamps)

        # Log internally
        log!(model, sample_ind, res)

        # Update kernel
        (i % update_kernel_every) == 0 ? update_kernel!(model) : nothing

        # Log safe set size
        sz = safe_set_size(model, problem.pfail_threshold, problem.conf_threshold)
        push!(set_sizes, sz)
    end

    return set_sizes
end

# General acquisition functions
function random_acquisition(model::SetEstimationModel)
    return rand(1:length(model.grid))
end