using GridInterpolations
using Distributions
using ProgressBars
using LinearAlgebra
using SpecialFunctions

mutable struct KernelBanditModel <: SetEstimationModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point (planning for one)
    eval_inds::Vector # Vector of points that got evaluated
    eval_res::Vector # Vector of booleans corresponding to success or failure for each simulation
    α::Vector # Failure counts
    β::Vector # Success counts
    widths::Vector # Width in each dimension for each grid region
    min_vals::Vector # Minimum of grid in each dimension
    max_vals::Vector # Maximum of grid in each dimension
    K::Matrix # Kernel matrix
    ℓ::Float64
    θs::Vector # Possible failure probabilities
    function KernelBanditModel(grid, k, ℓ; nbins=100)
        N = length(grid)
        widths = [cps[2] - cps[1] for cps in grid.cutPoints]
        min_vals = [cps[1] for cps in grid.cutPoints]
        max_vals = [cps[end] for cps in grid.cutPoints]
        X_pred = [X for X in grid]
        K = get_K(X_pred, X_pred, k)
        θs = collect(range(0, 1, length=nbins))
        return new(grid, 1, Vector{Int64}(), Vector{Bool}(), ones(N), ones(N), widths,
            min_vals, max_vals, K, ℓ, θs)
    end
end

function reset!(model::KernelBanditModel)
    model.eval_inds = Vector{Int64}()
    model.eval_res = Vector{Bool}()
    N = length(model.grid)
    model.α = ones(N)
    model.β = ones(N)
end

"""
Logging
"""
function log!(model::KernelBanditModel, sample_ind, res)
    push!(model.eval_inds, sample_ind)
    push!(model.eval_res, res[1])

    nfail = sum(res)
    model.α[sample_ind] += nfail
    model.β[sample_ind] += 1 - nfail
end

"""
Acquisition Functions
"""
to_params(model::KernelBanditModel, sample_ind) = ind2x(model.grid, sample_ind)

function gittens_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold, gi;
    rand_argmax=false)
    zvec = [quantile(Beta(α, β), conf_threshold) for (α, β) in zip(model.α, model.β)]

    gis = zeros(length(zvec))
    for i = 1:length(zvec)
        if zvec[i] < pfail_threshold
            gis[i] = -Inf
        else
            gis[i] = gi(convert(Int64, model.β[i]), convert(Int64, model.α[i]))
        end
    end

    if rand_argmax
        val = maximum(gis)
        inds = findall(gis .== val)
        return rand(inds)
    else
        return argmax(gis)
    end
end

function thompson_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold)
    zvec = [quantile(Beta(α, β), conf_threshold) for (α, β) in zip(model.α, model.β)]
    samps = [rand(Beta(α, β)) for (α, β) in zip(model.α, model.β)]
    samps[zvec.<pfail_threshold] .= Inf
    return argmin(samps)
end

function dkwucb_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold; δ=1.0,
    rand_argmax=false, buffer=0.0)
    pvec = [cdf(Beta(α, β), pfail_threshold) for (α, β) in zip(model.α, model.β)]
    N = model.α + model.β .- 2

    vals = zeros(length(pvec))
    for i = 1:length(pvec)
        if pvec[i] > conf_threshold + buffer
            vals[i] = -Inf
        else
            vals[i] = N[i] == 0 ? pvec[i] + 1 : pvec[i] + √(log(2 / δ) / (2N[i]))
        end
    end

    if rand_argmax
        val = maximum(vals)
        inds = findall(vals .== val)
        return rand(inds)
    else
        return argmax(vals)
    end
end

function kernel_dkwucb_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold; δ=1.0,
    rand_argmax=false, buffer=0.0)
    α̂ = 1 .+ model.K * (model.α .- 1)
    β̂ = 1 .+ model.K * (model.β .- 1)

    pvec = [cdf(Beta(α, β), pfail_threshold) for (α, β) in zip(α̂, β̂)]
    N = α̂ + β̂ .- 2

    vals = zeros(length(pvec))
    for i = 1:length(pvec)
        if pvec[i] > conf_threshold + buffer
            vals[i] = -Inf
        else
            vals[i] = N[i] == 0 ? pvec[i] + 1 : pvec[i] + √(log(2 / δ) / (2N[i]))
        end
    end

    if rand_argmax
        val = maximum(vals)
        inds = findall(vals .== val)
        return rand(inds)
    else
        return argmax(vals)
    end
end

"""
Estimation Functions
"""
function estimate_from_counts!(problem::GriddedProblem, model::KernelBanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α, β = predict_beta(model, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end

function estimate_from_est_counts!(problem::GriddedProblem, model::KernelBanditModel)
    α_est = 1 .+ model.K * (model.α .- 1)
    β_est = 1 .+ model.K * (model.β .- 1)

    for i = 1:length(problem.grid)
        problem.is_safe[i] = cdf(Beta(α_est[i], β_est[i]), problem.pfail_threshold) > problem.conf_threshold
    end
end

function safe_set_size(model::KernelBanditModel, pfail_threshold, conf_threshold)
    sz_nokernel = sum([cdf(Beta(α, β), pfail_threshold) > conf_threshold for (α, β) in zip(model.α, model.β)])

    α_est = 1 .+ model.K * (model.α .- 1)
    β_est = 1 .+ model.K * (model.β .- 1)
    sz_kernel = sum([cdf(Beta(α, β), pfail_threshold) > conf_threshold for (α, β) in zip(α_est, β_est)])
    return (sz_nokernel, sz_kernel)
end

"""
Kernel Bandit Specific Functions
"""
get_K(X, X′, k) = [k(x, x′) for x in X, x′ in X′]

function get_Ks(grid::RectangleGrid; w=[1.0, 0.04], ℓmin=1e-4, ℓmax=1e-2, nbins=200)
    X_pred = [X for X in grid]
    W = diagm(w ./ norm(w))

    ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
    Ks = [get_K(X_pred, X_pred, (x, x′) -> wsqe_kernel(x - x′, W, ℓ=ℓ)) for ℓ in ℓs]
    return ℓs, Ks
end