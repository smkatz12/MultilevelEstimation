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

function predict(GP::GaussianProcessModel)
    return predict(GP, GP.X_pred, GP.X_pred_inds, GP.K)
end

function predict(GP::GaussianProcessModel, X_pred, X_pred_inds, K)
    # start = time()
    tmp = K[X_pred_inds, GP.X_inds] / (K[GP.X_inds, GP.X_inds] + GP.ν * I)
    # println("l1: ", time() - start)

    # start = time()
    μ = GP.m(X_pred) + tmp * (GP.y - GP.m(GP.X))
    # println("l2: ", time() - start)

    # start = time()
    # # σ² = 1.0 .- diag(tmp * K[GP.X_inds, X_pred_inds])
    # println("l3: ", time() - start)

    # start = time()
    σ² = 1.0 .- dot.(eachrow(tmp), eachcol(K[GP.X_inds, X_pred_inds]))
    # println("l4: ", time() - start)

    return μ, σ²
end

function predict_old(GP::GaussianProcessModel, X_pred, X_pred_inds, K)
    tmp = K[X_pred_inds, GP.X_inds] / (K[GP.X_inds, GP.X_inds] + GP.ν * I)
    μ = GP.m(X_pred) + tmp * (GP.y - GP.m(GP.X))
    S = K[X_pred_inds, X_pred_inds] - tmp * K[GP.X_inds, X_pred_inds]
    σ² = diag(S) .+ eps()
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

function run_estimation!(model::GaussianProcessModel, problem::GriddedProblem, acquisition, nsamps;
    log_every=Inf)

    set_sizes = [0]

    neval = convert(Int64, floor(nsamps / model.nsamps))

    for i in ProgressBar(1:neval)
        # Select next point
        println("calling aq")
        sample_ind = acquisition(model)
        params = ind2x(model.grid, sample_ind)

        # Evaluate
        res = problem.sim(params, model.nsamps)
        nfail = sum(res)
        pfail = nfail / model.nsamps

        # Log
        push!(model.X, params)
        push!(model.X_inds, sample_ind)
        push!(model.y, pfail)
        deleteat!(model.X_pred_inds, findall(x -> x == params, model.X_pred))
        deleteat!(model.X_pred, findall(x -> x == params, model.X_pred))

        model.α[sample_ind] += nfail
        model.β[sample_ind] += 1 - nfail

        if (i % log_every) == 0
            sz = safe_set_size(model, problem.pfail_threshold, problem.conf_threshold)
            push!(set_sizes, sz)
            # estimate_from_gp!(problem, model)
            # push!(set_sizes, sum(problem.is_safe))
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
    μ, σ² = predict(model, X_pred, collect(1:length(model.grid)), model.K)

    is_safe = falses(length(model.grid))
    for i = 1:length(model.grid)
        is_safe[i] = μ[i] + β * √(σ²[i]) < pfail_threshold
    end

    return sum(is_safe)
end