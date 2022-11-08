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
    K::Matrix # Kernel matrix
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
        X_pred = [X for X in grid]
        X_pred_inds = collect(1:N)
        K = get_K(X_pred, X_pred, k)
        return new(grid, nsamps, m, k, K, ν, [], [], [], X_pred, X_pred_inds, ones(N), ones(N))
    end
end

function reset!(model::GaussianProcessModel)
    N = length(model.grid)
    model.X = []
    model.X_inds = []
    model.y = []
    model.X_pred = [X for X in model.grid]
    model.X_pred_inds = collect(1:N)
    model.α = ones(N)
    model.β = ones(N)
end

function predict(GP::GaussianProcessModel)
    return predict(GP, GP.X_pred, GP.X_pred_inds, GP.K)
end

function predict(GP::GaussianProcessModel, X_pred, X_pred_inds, K)
    return predict(GP, GP.X, GP.X_inds, GP.y, X_pred, X_pred_inds, K)
end

function predict(GP::GaussianProcessModel, X, X_inds, y, X_pred, X_pred_inds, K)
    tmp = K[X_pred_inds, X_inds] / (K[X_inds, X_inds] + GP.ν * I)
    μ = GP.m(X_pred) + tmp * (y - GP.m(X))
    σ² = 1.0 .- dot.(eachrow(tmp), eachcol(K[X_inds, X_pred_inds])) .+ eps()
    return μ, σ²
end

function predict(GP::GaussianProcessModel, X_pred)
    println("warning: you did the long one")
    tmp = get_K(X_pred, GP.X, GP.k) / (get_K(GP.X, GP.X, GP.k) + GP.ν * I)
    μ = GP.m(X_pred) + tmp * (GP.y - GP.m(GP.X))
    S = get_K(X_pred, X_pred, GP.k) - tmp * get_K(GP.X, X_pred, GP.k)
    σ² = diag(S) .+ eps()
    return μ, σ²
end

function predict_cov(GP::GaussianProcessModel)
    return predict_cov(GP, GP.X_pred, GP.X_pred_inds, GP.K)
end

function predict_cov(GP::GaussianProcessModel, X_pred, X_pred_inds, K)
    tmp = K[X_pred_inds, GP.X_inds] / (K[GP.X_inds, GP.X_inds] + GP.ν * I)
    μ = GP.m(X_pred) + tmp * (GP.y - GP.m(GP.X))
    S = GP.K[X_pred_inds, X_pred_inds] - tmp * GP.K[GP.X_inds, X_pred_inds]
    return μ, S
end

function run_estimation!(model::GaussianProcessModel, problem::GriddedProblem, acquisition, nsamps)

    set_sizes = [0]
    neval = convert(Int64, floor(nsamps / model.nsamps))

    for i in ProgressBar(1:neval)
        # Select next point
        sample_ind = acquisition(model)
        params = ind2x(model.grid, sample_ind)

        # Evaluate
        res = problem.sim(params, model.nsamps)
        
        # Log internally
        log!(model, sample_ind, res)

        # Log safe set size
        sz = safe_set_size(model, problem.pfail_threshold, problem.conf_threshold)
        push!(set_sizes, sz)
    end

    return set_sizes
end

function log!(model::GaussianProcessModel, sample_ind, res)
    nfail = sum(res)
    pfail = nfail / model.nsamps    

    push!(model.X, params)
    push!(model.X_inds, sample_ind)
    push!(model.y, pfail)
    deleteat!(model.X_pred_inds, findall(x -> x == params, model.X_pred))
    deleteat!(model.X_pred, findall(x -> x == params, model.X_pred))

    model.α[sample_ind] += nfail
    model.β[sample_ind] += 1 - nfail
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
    μ, σ² = predict(model, X_pred, collect(1:length(model.grid)), model.K)

    is_safe = (μ .+ β .* sqrt.(σ²)) .< pfail_threshold

    return sum(is_safe)
end