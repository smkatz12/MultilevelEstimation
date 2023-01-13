# Vanilla Monte Carlo
using GridInterpolations
using Distributions
using ProgressBars

"""
Set Estimation Model
"""
mutable struct MonteCarloModel <: SetEstimationModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point
    pfail::Vector # Estimated probability of failure at each point in the grid
    α::Vector # Failure counts
    β::Vector # Success counts
    curr_ind::Int # Current grid index (for acquisition)
    function MonteCarloModel(grid, nsamps)
        N = length(grid)
        return new(grid, nsamps, ones(N), ones(N), ones(N), 1)
    end
end

function reset!(model::MonteCarloModel)
    N = length(model.grid)
    model.pfail = zeros(N)
    model.α = ones(N)
    model.β = ones(N)
    model.curr_ind = 1
end

"""
Logging
"""
function log!(model::MonteCarloModel, sample_ind, res)
    nfail = sum(res)
    pfail = nfail / model.nsamps

    model.α[sample_ind] += nfail
    model.β[sample_ind] += 1 - nfail
    model.pfail[sample_ind] = pfail
    model.curr_ind += 1
end

"""
Acquisition Functions
"""
to_params(model::MonteCarloModel, sample_ind) = GridInterpolations.ind2x(model.grid, sample_ind)

mc_acquisition(model::MonteCarloModel) = model.curr_ind

"""
Estimation Functions
"""

function estimate_from_pfail!(problem::GriddedProblem, model::MonteCarloModel)
    for i = 1:length(problem.grid)
        params = GridInterpolations.ind2x(problem.grid, i)
        pfail = interpolate(model.grid, model.pfail, params)
        problem.is_safe[i] = pfail < problem.pfail_threshold
    end
end

function estimate_from_counts!(problem::GriddedProblem, model::MonteCarloModel)
    for i = 1:length(problem.grid)
        params = GridInterpolations.ind2x(problem.grid, i)
        α = interpolate(model.grid, model.α, params)
        β = interpolate(model.grid, model.β, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end

function safe_set_size(model::MonteCarloModel, pfail_threshold, conf_threshold)
    return sum(model.pfail .< pfail_threshold)
end