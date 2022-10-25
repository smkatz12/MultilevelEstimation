# Vanilla Monte Carlo
using GridInterpolations
using Distributions
using ProgressBars

mutable struct MonteCarloModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point
    pfail::Vector # Estimated probability of failure at each point in the grid
    α::Vector # Failure counts
    β::Vector # Success counts
end

function run_estimation!(model::MonteCarloModel, problem::GriddedProblem)
    nparams = length(model.grid)

    # Update pfail vector
    for i in ProgressBar(1:nparams)
        params = ind2x(model.grid, i)
        res = problem.sim(params, model.nsamps)

        nfail = sum(res)
        model.pfail[i] = nfail / nsamps
        model.α[i] += nfail
        model.β[i] += nsamps - nfail
    end
end

function estimate_from_pfail!(problem::GriddedProblem, model::MonteCarloModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        pfail = interpolate(model.grid, model.pfail, params)
        problem.is_safe[i] = pfail < problem.pfail_threshold
    end
end

function estimate_from_counts!(problem::GriddedProblem, model::MonteCarloModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α = interpolate(model.grid, model.α, params)
        β = interpolate(model.grid, model.β, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end