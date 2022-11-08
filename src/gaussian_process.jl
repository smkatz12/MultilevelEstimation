# Gaussian Process Model
using GridInterpolations
using Distributions
using LinearAlgebra

"""
Set Estimation Model
"""
mutable struct GaussianProcessModel <: SetEstimationModel
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

"""
Logging
"""
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

"""
Acquisition Functions
"""
function MILE(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    """
    NOTE: this will not reevaluate an already selected point
    """
    β = quantile(Normal(), conf_threshold)

    neval = length(model.X)
    npred = length(model.X_pred)

    if neval > 0
        objecs = zeros(npred)
        μ, S = predict_cov(model)
        σ² = diag(S)
        for i = 1:npred
            σ²GP⁺ = σ² .- (S[:, i] .^ 2) ./ (σ²[i] + model.ν)
            zvec = (√(σ²[i] + model.ν) ./ abs.(S[:, i])) .* (pfail_threshold .- μ .- β .* sqrt.(σ²GP⁺))
            objecs[i] = sum(cdf(Normal(), zvec))
        end

        max_ind = argmax(objecs)
        max_val = maximum(objecs)

        return max_val, model.X_pred_inds[max_ind] #, objecs #zout
    else
        return 0.0, rand(1:length(model.grid))
    end
end

function MILE_acquisition(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    """
    NOTE: this will not reevaluate an already selected point
    """
    _, ind = MILE(model, pfail_threshold, conf_threshold)
    return ind #, objecs
end

function RMILE_acquisition(model::GaussianProcessModel, pfail_threshold, conf_threshold; γ=1e-5)
    """
    NOTE: this will not reevaluate an already selected point
    """
    Δ_size, ind = MILE(model, pfail_threshold, conf_threshold)

    σ²_pred = predict(model)
    max_pred = maximum(σ²_pred)

    if γ * max_pred > Δ_size
        return model.X_pred_inds[argmax(σ²_pred)]
    else
        return ind
    end
end

"""
Estimation Functions
"""
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

"""
GP Specific Functions
"""
get_K(X, X′, k) = [k(x, x′) for x in X, x′ in X′]

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