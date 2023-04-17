# Point Specific Learning Kernel Bandit Model
using GridInterpolations
using Distributions
using ProgressBars
using LinearAlgebra
using SpecialFunctions

mutable struct PSpecBanditModel <: SetEstimationModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point (planning for one)
    eval_inds::Vector # Vector of points that got evaluated
    eval_res::Vector # Vector of booleans corresponding to success or failure for each simulation
    α::Vector # Failure counts
    β::Vector # Success counts
    αₖ::Matrix # Kernel values of α for all possible K (values are in the columns)
    βₖ::Matrix # Kernel values of β for all possible K (values are in the columns)
    θdists::Matrix # Estimates of pfail distributions
    widths::Vector # Width in each dimension for each grid region
    min_vals::Vector # Minimum of grid in each dimension
    max_vals::Vector # Maximum of grid in each dimension
    curr_pspecℓs::Matrix # Current point specific distribution over ℓs
    ℓs::Vector # Possible values of ℓ
    Ks::Matrix # Kernel matrix for all possible ℓs (length(grid)*nbins, length(grid))
    θs::Vector # Possible failure probabilities
    function PSpecBanditModel(grid; ℓmin=1e-4, ℓmax=1e-2, nbins=100, w=[1.0, 0.04])
        N = length(grid)
        widths = [cps[2] - cps[1] for cps in grid.cutPoints]
        min_vals = [cps[1] for cps in grid.cutPoints]
        max_vals = [cps[end] for cps in grid.cutPoints]
        ℓs, Ks = get_Ks(grid; w=w, ℓmin=ℓmin, ℓmax=ℓmax, nbins=nbins)
        Kmat = cat(Ks..., dims=1)
        curr_pspecℓs = ones(N, nbins) / nbins
        θs = collect(range(0, 1, length=nbins))
        return new(grid, 1, Vector{Int64}(), Vector{Bool}(), ones(N), ones(N),
            ones(N, nbins), ones(N, nbins), ones(N, nbins) / nbins, widths, min_vals,
            max_vals, curr_pspecℓs, ℓs, Kmat, θs)
    end
end

function reset!(model::PSpecBanditModel)
    model.eval_inds = Vector{Int64}()
    model.eval_res = Vector{Bool}()
    N = length(model.grid)
    nbins = length(model.ℓs)
    model.α = ones(N)
    model.β = ones(N)
    model.αₖ = ones(N, nbins)
    model.βₖ = ones(N, nbins)
    model.θdists = ones(N, nbins) / nbins
    model.curr_pspecℓs = ones(N, nbins) / nbins
end

"""
Logging
"""
function log!(model::PSpecBanditModel, sample_ind, res)
    push!(model.eval_inds, sample_ind)
    push!(model.eval_res, res[1])

    nfail = sum(res)
    model.α[sample_ind] += nfail
    model.β[sample_ind] += 1 - nfail

    # Kernel estimates
    if nfail > 0
        model.αₖ = model.αₖ .+ reshape(model.Ks[:, sample_ind], length(model.grid), length(model.ℓs))
    else
        model.βₖ = model.βₖ .+ reshape(model.Ks[:, sample_ind], length(model.grid), length(model.ℓs))
    end
    update_kernel!(model)

    dists = Beta.(model.αₖ, model.βₖ) # N x nbins matrix of beta distributions
    nbins = size(dists, 2)
    for (i, θ) in enumerate(model.θs)
        model.θdists[:, i] = sum(pdf.(dists, θ) .* model.curr_pspecℓs, dims=2)
    end
    for i = 1:size(model.θdists, 1)
        model.θdists[i, :] = model.θdists[i, :] ./ sum(model.θdists[i, :])
    end
end

"""
Acquisition Functions
"""
to_params(model::PSpecBanditModel, sample_ind) = ind2x(model.grid, sample_ind)

function dkwucb_acquisition(model::PSpecBanditModel, pfail_threshold, conf_threshold; δ=1.0,
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

function kernel_dkwucb_acquisition(model::PSpecBanditModel, pfail_threshold, conf_threshold; δ=1.0,
    rand_argmax=false, buffer=0.0)

    pfail_ind = findfirst(model.θs .>= pfail_threshold) - 1
    # pvec = model.θdists[:, pfail_ind]
    N = model.α + model.β .- 2

    npoints = length(model.grid)
    vals = zeros(npoints)
    for i = 1:npoints
        pval = sum(model.θdists[i, 1:pfail_ind])
        if pval > conf_threshold + buffer
            vals[i] = -Inf
        else
            vals[i] = N[i] == 0 ? pval + 1 : pval + √(log(2 / δ) / (2N[i]))
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
function estimate_from_counts!(problem::GriddedProblem, model::PSpecBanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α, β = predict_beta(model, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end

function estimate_from_est_counts!(problem::GriddedProblem, model::PSpecBanditModel)
    pfail_ind = findfirst(model.θs .>= problem.pfail_threshold)
    for i = 1:length(problem.grid)
        conf_ind = findfirst(cumsum(model.θdists[i, :]) .> problem.conf_threshold)
        problem.is_safe[i] = conf_ind ≤ pfail_ind
    end
end

function safe_set_size(model::PSpecBanditModel, pfail_threshold, conf_threshold)
    sz_nokernel = sum([cdf(Beta(α, β), pfail_threshold) > conf_threshold for (α, β) in zip(model.α, model.β)])

    pfail_ind = findfirst(model.θs .>= pfail_threshold)
    sz_kernel = 0
    for i = 1:length(model.grid)
        conf_ind = findfirst(cumsum(model.θdists[i, :]) .> conf_threshold)
        if conf_ind ≤ pfail_ind
            sz_kernel += 1
        end
    end
    return (sz_nokernel, sz_kernel)
end

"""
Kernel Bandit Specific Functions
"""
function predict_beta(model::PSpecBanditModel, params)
    s, p = interpolants(model.grid, params)
    ind = s[argmax(p)]
    α = model.α[ind]
    β = model.β[ind]
    return α, β
end

get_K(X, X′, k) = [k(x, x′) for x in X, x′ in X′]

function get_Ks(grid::RectangleGrid; w=[1.0, 0.04], ℓmin=1e-4, ℓmax=1e-2, nbins=200)
    X_pred = [X for X in grid]
    W = diagm(w ./ norm(w))

    ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
    Ks = [get_K(X_pred, X_pred, (x, x′) -> wsqe_kernel(x - x′, W, ℓ=ℓ)) for ℓ in ℓs]
    return ℓs, Ks
end

"""
Length Parameter Estimation Functions
"""
function logp_αβ(α, β, αₖ, βₖ)
    n, m = α .- 1, α .+ β .- 2
    nₖ, mₖ = αₖ .- 1, αₖ .+ βₖ .- 2
    numerator = loggamma.(mₖ .+ 2) .+ loggamma.(m .+ 1) .+ loggamma.(nₖ .+ n .+ 1) .+ loggamma.(mₖ .- nₖ .+ m .- n .+ 1)
    denominator = loggamma.(nₖ .+ 1) .+ loggamma.(n .+ 1) .+ loggamma.(mₖ .- nₖ .+ 1) + loggamma.(mₖ .+ m .+ 2) .+ loggamma.(m .- n .+ 1)
    logp = numerator .- denominator
    return logp
end

function pspec_pℓ(αs, βs, αₖs, βₖs)
    npoints = length(αs)
    nℓ = size(αₖs, 2)

    pspec_lps = zeros(npoints, nℓ)
    for i = 1:nℓ
        pspec_lps[:, i] = logp_αβ(αs, βs, αₖs[:, i], βₖs[:, i])
    end

    pspec_pℓs = zeros(npoints, nℓ)
    for i = 1:npoints
        lsume = logsumexp(pspec_lps[i, :])
        lpℓs = pspec_lps[i, :] .- lsume
        pspec_pℓs[i, :] = exp.(lpℓs)
    end

    return pspec_pℓs
end

function pspec_pℓ(model::PSpecBanditModel)
    return pspec_pℓ(model.α, model.β, model.αₖ, model.βₖ)
end

function update_kernel!(model::PSpecBanditModel)
    model.curr_pspecℓs = pspec_pℓ(model)
end