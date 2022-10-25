# Bandit model
using GridInterpolations
using Distributions
using ProgressBars

mutable struct BanditModel
    grid::RectangleGrid # Grid to evaluate on
    pfail::Vector # Estimated probability of failure at each point in the grid
    α::Vector # Failure counts
    β::Vector # Success counts
end

function run_estimation!(model::BanditModel, problem::GriddedProblem, acquisition, nsamps;
                         log_every=Inf)
    set_sizes = [0]
    
    for i in ProgressBar(1:nsamps)
        sample_ind = acquisition(model)
        params = ind2x(model.grid, sample_ind)
        res = problem.sim(params, 1)
        nfail = sum(res)


        model.α[sample_ind] += nfail
        model.β[sample_ind] += 1 - nfail
        model.pfail[sample_ind] = model.α[sample_ind] / (model.α[sample_ind] + model.β[sample_ind])

        if (i % log_every) == 0
            estimate_from_counts!(problem, model)
            push!(set_sizes, sum(problem.is_safe))
        end
    end

    return set_sizes
end

function estimate_from_pfail!(problem::GriddedProblem, model::BanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        pfail = interpolate(model.grid, model.pfail, params)
        problem.is_safe[i] = pfail < problem.pfail_threshold
    end
end

function estimate_from_counts!(problem::GriddedProblem, model::BanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α = interpolate(model.grid, model.α, params)
        β = interpolate(model.grid, model.β, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end