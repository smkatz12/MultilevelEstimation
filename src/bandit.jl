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
        old_prob_mass = cdf(Beta(model.α[i], model.β[i]), pfail_threshold)
        safe = old_prob_mass > conf_threshold
        if !safe
            ei = expected_improvement_v2(model.α[i], model.β[i], pfail_threshold, old_prob_mass)
            if ei > curr_best
                curr_best = ei
                curr_ind = i
            end
        end
    end

    return curr_ind
end

function lcb_bayes_acquisition(model::BanditModel, pfail_threshold, conf_threshold)
    # Compute current lbs
    zvec = [quantile(Beta(α, β), conf_threshold) for (α, β) in zip(model.α, model.β)]
    
    # Mask out ones that are already safe
    zvec[zvec .< pfail_threshold] .= Inf

    # Return the lowest one
    return argmin(zvec)
end

function lcb_acquisition(model::BanditModel, pfail_threshold, conf_threshold; c=0.1)
    zvec = [quantile(Beta(α, β), conf_threshold) for (α, β) in zip(model.α, model.β)]
    t = length(model.eval_inds)

    A = zeros(length(zvec))
    for i = 1:length(zvec)
        if zvec[i] < pfail_threshold
            A[i] = Inf
        else
            N = (model.α[i] + model.β[i])
            Q = model.α[i] / N
            A[i] = Q - c * √(log(t) / N)
        end
    end

    # Return the lowest one
    return argmin(A)
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

function expected_improvement_v2(α, β, pfail_threshold, old_prob_mass)
    p_success = β / (α + β)
    new_prob_mass = cdf(Beta(α, β + 1), pfail_threshold)
    return p_success * (new_prob_mass - old_prob_mass)
end