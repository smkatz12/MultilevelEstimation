# Multi-armed Bandit Model
using GridInterpolations
using Distributions
using LinearAlgebra

"""
Set Estimation Model
"""
mutable struct BanditModel <: SetEstimationModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point (planning for one)
    eval_inds::Vector # Vector of points that got evaluated
    α::Vector # Failure counts
    β::Vector # Success counts
    widths::Vector # Width in each dimension for each grid region
    min_vals::Vector # Minimum of grid in each dimension
    max_vals::Vector # Maximum of grid in each dimension
    function BanditModel(grid)
        N = length(grid)
        widths = [cps[2] - cps[1] for cps in grid.cutPoints]
        min_vals = [cps[1] for cps in grid.cutPoints]
        max_vals = [cps[end] for cps in grid.cutPoints]
        return new(grid, 1, Vector{Int64}(), ones(N), ones(N), widths, min_vals, max_vals)
    end
end

function reset!(model::BanditModel)
    model.eval_inds = []
    model.α = ones(N)
    model.β = ones(N)
end

"""
Logging
"""
function log!(model::BanditModel, sample_ind, res)
    push!(model.eval_inds, sample_ind)

    nfail = sum(res)
    model.α[sample_ind] += nfail
    model.β[sample_ind] += 1 - nfail
end

"""
Acquisition Functions
"""
function to_params(model::BanditModel, sample_ind)
    x = ind2x(model.grid, sample_ind)
    params = rand.(Uniform.(max.(x .- model.widths, model.min_vals),
                            min.(x .+ model.widths, model.max_vals)))
    return params
end

function max_improvement_acquisition(model::BanditModel, pfail_threshold, conf_threshold)
    curr_best = 0.0
    curr_ind = 0

    for i = 1:length(model.grid)
        # Check if already safe
        safe = cdf(Beta(model.α[i], model.β[i]), pfail_threshold) > conf_threshold
        if !safe
            ei = expected_improvement(model.α[i], model.β[i], pfail_threshold)
            if ei > curr_best
                curr_best = ei
                curr_ind = i
            end
        end
    end

    return curr_ind
end

"""
Estimation Functions
"""
function estimate_from_counts!(problem::GriddedProblem, model::BanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α, β = predict_beta(model, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end

function safe_set_size(model::BanditModel, pfail_threshold, conf_threshold)
    return sum([cdf(Beta(α, β), pfail_threshold) > conf_threshold for (α, β) in zip(model.α, model.β)])
end

"""
Region Bandit Specific Functions
"""
function predict_beta(model::BanditModel, params)
    s, p = interpolants(model.grid, params)
    ind = s[argmax(p)]
    α = model.α[ind]
    β = model.β[ind]
    return α, β
end

function expected_improvement(α, β, pfail_threshold)
    p_success = β / (α + β)
    new_prob_mass = cdf(Beta(α, β + 1), pfail_threshold)
    return p_success * new_prob_mass
end