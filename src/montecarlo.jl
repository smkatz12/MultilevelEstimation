# Vanilla Monte Carlo
using GridInterpolations

mutable struct MonteCarloModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point
    pfail::Vector # Estimated probability of failure at each point in the grid
end

function run_estimation!(model::MonteCarloModel, problem::GriddedProblem)
    nparams = length(model.grid)

    # Update pfail vector
    for i = 1:nparams
        params = ind2x(model.grid, i)
        res = problem.sim(params, model.nsamps)
        model.pfail[i] = sum(res) / nsamps
    end

    # Update problem
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        pfail = interpolate(model.grid, model.pfail, params)
        problem.is_safe[i] = pfail < problem.pfail_threshold
    end
end