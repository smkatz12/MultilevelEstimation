# Gaussian Process Model
using GridInterpolations
using Distributions
using LinearAlgebra
using ProgressBars

get_K(X, X′, k) = [k(x, x′) for x in X, x′ in X′]

mutable struct GaussianProcessModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point
    m::Function # Mean function
    k::Function # Kernel function
    ν::Float64 # Noise variance
    X::Vector # Evaluated points
    X_inds::Vector # Evaluated inds
    y::Vector # Pfail for evaluated points
    X_pred::Vector # Points to predict
    X_pred_inds::Vector # Inds of points to predict
    α::Vector # Failure counts
    β::Vector # Success counts
    function GaussianProcessModel(grid, nsamps, m, k, ν)
        N = length(grid)
        X_pred = [X for X in model.grid]
        X_pred_inds = collect(1:N)
        return new(grid, nsamps, m, k, ν, [], [], [], X_pred, X_pred_inds, ones(N), ones(N))
    end
end

function predict(GP::GaussianProcessModel)
    return predict(GP, GP.X_pred)
end

function predict(GP::GaussianProcessModel, X_pred)
    tmp = get_K(X_pred, GP.X, GP.k) / (get_K(GP.X, GP.X, GP.k) + ν * I)
    μ = GP.m(X_pred) + tmp * (GP.y - m(X))
    S = get_K(X_pred, X_pred, GP.k) - tmp * get_K(GP.X, X_pred, k)
    σ² = diag(S) .+ eps()
    return μ, σ²
end

function run_estimation!(model::GaussianProcessModel, problem::GriddedProblem, acquisition, nsamps;
    log_every=Inf)

    set_sizes = [0]

    neval = convert(Int64, floor(nsamps / model.nsamps))

    for i in ProgressBar(1:neval)
        # Select next point
        sample_ind = acquisition(model)
        params = ind2x(model.grid, sample_ind)

        # Evaluate
        res = problem.sim(params, model.nsamps)
        pfail = sum(res) / model.nsamps

        # Log
        push!(model.X, params)
        push!(model.X_inds, sample_ind)
        push!(model.y, pfail)
        deleteat!(model.X_pred, findall(x -> x == params, model.X_pred))
        deleteat!(model.X_pred_inds, findall(x -> x == params, model.X_pred_inds))

        model.α[sample_ind] += nfail
        model.β[sample_ind] += 1 - nfail

        if (i % log_every) == 0
            estimate_from_gp!(problem, model)
            push!(set_sizes, sum(problem.is_safe))
        end
    end

    return set_sizes
end

function estimate_from_gp!(problem::GriddedProblem, model::GaussianProcessModel)
    β = quantile(Normal(), problem.conf_threshold)
    
    X_pred = [X for X in problem.grid]
    μ, σ² = predict(model, X_pred)

    for i = 1:length(problem.grid)
        problem.is_safe[i] = μ[i] + β * √(σ²[i]) < problem.pfail_threshold
    end
end

function safe_set_size(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    β = quantile(Normal(), conf_threshold)

    X_pred = [X for X in model.grid]
    μ, σ² = predict(model, X_pred)

    is_safe = falses(length(model.grid))
    for i = 1:length(model.grid)
        is_safe[i] = μ[i] + β * √(σ²[i]) < pfail_threshold
    end

    return sum(is_safe)
end