# Learning Kernel Bandit Model
using GridInterpolations
using Distributions
using ProgressBars
using LinearAlgebra
using SpecialFunctions

mutable struct LearningBanditModel <: SetEstimationModel
    grid::RectangleGrid # Grid to evaluate on
    nsamps::Int # Number of samples to run per grid point (planning for one)
    eval_inds::Vector # Vector of points that got evaluated
    eval_res::Vector # Vector of booleans corresponding to success or failure for each simulation
    α::Vector # Failure counts
    β::Vector # Success counts
    αₖ::Matrix # Kernel values of α for all possible K (values are in the columns)
    βₖ::Matrix # Kernel values of β for all possible K (values are in the columns)
    αest::Vector # Estimated kernel values
    βest::Vector # Estimated kernel values
    widths::Vector # Width in each dimension for each grid region
    min_vals::Vector # Minimum of grid in each dimension
    max_vals::Vector # Maximum of grid in each dimension
    curr_pℓs::Vector # Current distribution over ℓs
    ℓs::Vector # Possible values of ℓ
    Ks::Matrix # Kernel matrix for all possible ℓs (length(grid)*nbins, length(grid))
    ℓconf::Float64 # Confidence interval for ℓ (to be used in CVaR eventually?)
    ℓests::Vector # Estimate of ℓ at each evaluation
    function LearningBanditModel(grid; ℓmin=1e-4, ℓmax=1e-2, nbins=100, ℓconf=0.95)
        N = length(grid)
        widths = [cps[2] - cps[1] for cps in grid.cutPoints]
        min_vals = [cps[1] for cps in grid.cutPoints]
        max_vals = [cps[end] for cps in grid.cutPoints]
        ℓs, Ks = get_Ks(grid; ℓmin=ℓmin, ℓmax=ℓmax, nbins=nbins)
        Kmat = cat(Ks..., dims=1)
        q = convert(Int64, floor((1 - ℓconf) * nbins))
        curr_pℓs = ones(nbins) / nbins
        return new(grid, 1, Vector{Int64}(), Vector{Bool}(), ones(N), ones(N),
            ones(N, nbins), ones(N, nbins), ones(N), ones(N), widths, min_vals,
            max_vals, curr_pℓs, ℓs, Kmat, ℓconf, Vector{Float64}())
    end
end

function reset!(model::LearningBanditModel)
    model.eval_inds = Vector{Int64}()
    model.eval_res = Vector{Bool}()
    N = length(model.grid)
    nbins = length(model.ℓs)
    model.α = ones(N)
    model.β = ones(N)
    model.αₖ = ones(N, nbins)
    model.βₖ = ones(N, nbins)
    model.curr_pℓs = ones(nbins) / nbins
    model.ℓests = Vector{Float64}()
end

"""
Logging
"""
function log!(model::LearningBanditModel, sample_ind, res)
    push!(model.eval_inds, sample_ind)
    push!(model.eval_res, res[1])
    push!(model.ℓests, dot(model.curr_pℓs, model.ℓs))

    nfail = sum(res)
    model.α[sample_ind] += nfail
    model.β[sample_ind] += 1 - nfail

    # Kernel estimates
    model.αₖ = reshape(1 .+ model.Ks * (model.α .- 1), length(model.grid), length(model.ℓs))
    model.βₖ = reshape(1 .+ model.Ks * (model.β .- 1), length(model.grid), length(model.ℓs))
    update_kernel!(model)
    model.αest = model.αₖ * model.curr_pℓs
    model.βest = model.βₖ * model.curr_pℓs
end

"""
Acquisition Functions
"""
to_params(model::LearningBanditModel, sample_ind) = ind2x(model.grid, sample_ind)

function dkwucb_acquisition(model::LearningBanditModel, pfail_threshold, conf_threshold; δ=1.0,
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

function kernel_dkwucb_acquisition(model::LearningBanditModel, pfail_threshold, conf_threshold; δ=1.0,
    rand_argmax=false, buffer=0.0)

    αest, βest = model.αest, model.βest
    pvec = [cdf(Beta(α, β), pfail_threshold) for (α, β) in zip(αest, βest)]
    N = αest .+ βest .- 2

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
function estimate_from_counts!(problem::GriddedProblem, model::LearningBanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α, β = predict_beta(model, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end

function estimate_from_est_counts!(problem::GriddedProblem, model::LearningBanditModel)
    for i = 1:length(problem.grid)
        problem.is_safe[i] = cdf(Beta(model.αest[i], model.βest[i]), problem.pfail_threshold) > problem.conf_threshold
    end
end

function safe_set_size(model::LearningBanditModel, pfail_threshold, conf_threshold)
    sz_nokernel = sum([cdf(Beta(α, β), pfail_threshold) > conf_threshold for (α, β) in zip(model.α, model.β)])
    sz_kernel = sum([cdf(Beta(α, β), pfail_threshold) > conf_threshold for (α, β) in zip(model.αest, model.βest)])
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

"""
Kernel Estimation Functions
"""
function logp_αβ(α, β, αₖ, βₖ)
    n, m = α .- 1, α + β .- 2
    nₖ, mₖ = αₖ .- 1, αₖ .+ βₖ .- 2
    numerator = loggamma.(mₖ .+ 2) .+ loggamma.(m .+ 1) .+ loggamma.(nₖ .+ n .+ 1) .+ loggamma.(mₖ .- nₖ .+ m .- n .+ 1)
    denominator = loggamma.(nₖ .+ 1) .+ loggamma.(n .+ 1) .+ loggamma.(mₖ .- nₖ .+ 1) + loggamma.(mₖ .+ m .+ 2) .+ loggamma.(m .- n .+ 1)
    logp = numerator .- denominator
    return logp
end

function log_ps(model::LearningBanditModel)
    return log_p(model, model.α, model.β, model.αₖ, model.βₖ)
end

function log_ps(αs, βs, αₖs, βₖs)
    return [sum(logp_αβ(αs, βs, αₖs[:, i], βₖs[:, i])) for i = 1:size(αₖs, 2)]
end

function pℓ(model::LearningBanditModel)
    return pℓ(model.α, model.β, model.αₖ, model.βₖ)
end

function pℓ(α, β, αₖ, βₖ)
    lps = log_ps(α, β, αₖ, βₖ)
    lsume = logsumexp(lps)
    lpℓs = lps .- lsume
    pℓs = exp.(lpℓs)
    return pℓs
end

function update_kernel!(model::LearningBanditModel)
    model.curr_pℓs = pℓ(model)
end