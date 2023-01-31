# Kernel Bandit Model
using GridInterpolations
using Distributions
using ProgressBars

"""
Set Estimation Model
"""
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
    curr_ℓ::Float64
    K::Matrix # Kernel matrix
    ℓs::Vector
    Ks::Vector # Kernel matrix for all possible ℓs
    ℓconf::Float64 # Confidence interval for ℓ
    ℓests::Vector # Estimate of ℓ at each evaluation
    function KernelBanditModel(grid, k, curr_ℓ; ℓmin=1e-4, ℓmax=1e-2, nbins=100, ℓconf=0.95)
        N = length(grid)
        widths = [cps[2] - cps[1] for cps in grid.cutPoints]
        min_vals = [cps[1] for cps in grid.cutPoints]
        max_vals = [cps[end] for cps in grid.cutPoints]
        X_pred = [X for X in grid]
        K = get_K(X_pred, X_pred, k)
        ℓs, Ks = get_Ks(grid; ℓmin=ℓmin, ℓmax=ℓmax, nbins=nbins)
        return new(grid, 1, Vector{Int64}(), Vector{Bool}(), ones(N), ones(N), widths,
            min_vals, max_vals, curr_ℓ, K, ℓs, Ks, ℓconf, Vector{Float64}())
    end
    function KernelBanditModel(grid; ℓmin=1e-4, ℓmax=1e-2, nbins=100, ℓconf=0.95)
        N = length(grid)
        widths = [cps[2] - cps[1] for cps in grid.cutPoints]
        min_vals = [cps[1] for cps in grid.cutPoints]
        max_vals = [cps[end] for cps in grid.cutPoints]
        ℓs, Ks = get_Ks(grid; ℓmin=ℓmin, ℓmax=ℓmax, nbins=nbins)
        q = convert(Int64, floor((1 - ℓconf) * nbins))
        K = Ks[q]
        curr_ℓ = ℓs[q]
        return new(grid, 1, Vector{Int64}(), Vector{Bool}(), ones(N), ones(N), widths,
            min_vals, max_vals, curr_ℓ, K, ℓs, Ks, ℓconf, Vector{Float64}())
    end
end

function reset!(model::KernelBanditModel)
    model.eval_inds = Vector{Int64}()
    model.eval_res = Vector{Bool}()
    N = length(model.grid)
    model.α = ones(N)
    model.β = ones(N)
    model.ℓests = Vector{Float64}()
    q = convert(Int64, floor((1 - model.ℓconf) * length(model.ℓs)))
    model.K = model.Ks[q]
    model.curr_ℓ = model.ℓs[q]
end

"""
Logging
"""
function log!(model::KernelBanditModel, sample_ind, res)
    push!(model.eval_inds, sample_ind)
    push!(model.eval_res, res[1])
    push!(model.ℓests, model.curr_ℓ)

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

function max_improve_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold; ρ=2.0,
    rand_argmax=false)

    α, β = model.α, model.β
    
    vals = zeros(length(α))
    for i = 1:length(vals)
        pfail = α[i] / (α[i] + β[i])
        αfail = copy(α)
        αfail[i] += 1
        scorefail = score(model, αfail, β, pfail_threshold, conf_threshold, ρ=ρ)
        
        psucceed = 1 - pfail
        βsucceed = copy(β)
        βsucceed[i] += 1
        scoresucceed = score(model, α, βsucceed, pfail_threshold, conf_threshold, ρ=ρ)
        
        vals[i] = pfail * scorefail + psucceed * scoresucceed
    end

    if rand_argmax
        val = maximum(vals)
        inds = findall(vals .== val)
        return rand(inds)
    else
        return argmax(vals)
    end
end

function faster_max_improve_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold; ρ=2.0,
    rand_argmax=false)

    α, β = model.α, model.β
    curr_α_est = 1 .+ model.K * (α .- 1)
    curr_β_est = 1 .+ model.K * (β .- 1)
    
    vals = zeros(length(α))
    for i = 1:length(vals)
        pfail = α[i] / (α[i] + β[i])
        scorefail = score(model, curr_α_est, curr_β_est, i, true, pfail_threshold, conf_threshold, ρ=ρ)
        
        psucceed = 1 - pfail

        scoresucceed = score(model, curr_α_est, curr_β_est, false, conf_threshold, ρ=ρ)
        
        vals[i] = pfail * scorefail + psucceed * scoresucceed
    end

    if rand_argmax
        val = maximum(vals)
        inds = findall(vals .== val)
        return rand(inds)
    else
        return argmax(vals)
    end
end

function optim_max_improve_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold; ρ=2.0,
    rand_argmax=false)

    α, β = model.α, model.β
    curr_α_est = 1 .+ model.K * (α .- 1)
    curr_β_est = 1 .+ model.K * (β .- 1)

    vals = zeros(length(α))
    for i = 1:length(vals)
        psucceed = curr_α_est[i] / (curr_α_est[i] + curr_β_est[i])
        scoresucceed = score(model, curr_α_est, curr_β_est, false, conf_threshold, ρ=ρ)
        vals[i] = psucceed * scoresucceed
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

function predict_beta(model::KernelBanditModel, params)
    s, p = interpolants(model.grid, params)
    ind = s[argmax(p)]
    α = model.α[ind]
    β = model.β[ind]
    return α, β
end

function score(model::KernelBanditModel, α, β, pfail_threshold, conf_threshold; ρ=2.0)
    α_est = 1 .+ model.K * (α .- 1)
    β_est = 1 .+ model.K * (β .- 1)

    scores = [cdf(Beta(α₀, β₀), pfail_threshold) for (α₀, β₀) in zip(α_est, β_est)]
    scores[scores .> conf_threshold] .= ρ

    return sum(scores)
end

function score(model::KernelBanditModel, curr_α_est, curr_β_est, n, fail, pfail_threshold, 
    conf_threshold; ρ=2.0)

    α_est = fail ? curr_α_est + model.K[:, n] : curr_α_est
    β_est = fail ? curr_β_est : curr_β_est + model.K[:, n]

    scores = [cdf(Beta(α₀, β₀), pfail_threshold) for (α₀, β₀) in zip(α_est, β_est)]
    scores[scores .> conf_threshold] .= ρ

    return sum(scores)
end

"""
Kernel Estimation Functions
"""
function p_αβ(α, β, αₖ, βₖ; nθ=100)
    dist = Beta(αₖ, βₖ)
    terms = [θ^α * (1 - θ)^β * pdf(dist, θ) for θ in range(0.0, stop=1.0, length=nθ)]
    return (1 / nθ) * sum(terms)
end

function p_αβ_exact(α, β, αₖ, βₖ)
    n, m = α, α + β
    nₖ, mₖ = αₖ - 1, αₖ + βₖ - 2
    numerator = gamma(mₖ + 2) * gamma(nₖ + n + 1) * gamma(mₖ - nₖ + m - n + 1)
    println(gamma(nₖ + n + 1))
    denominator = gamma(nₖ + 1) * gamma(mₖ - nₖ + 1) * gamma(mₖ + m + 2)
    println(denominator)
    p = numerator / denominator
    return p
end

function logp_αβ(α, β, αₖ, βₖ)
    n, m = α, α + β
    nₖ, mₖ = αₖ - 1, αₖ + βₖ - 2
    numerator = loggamma(mₖ + 2) + loggamma(nₖ + n + 1) + loggamma(mₖ - nₖ + m - n + 1)
    denominator = loggamma(nₖ + 1) + loggamma(mₖ - nₖ + 1) + loggamma(mₖ + m + 2)
    logp = numerator - denominator
    return logp
end

function log_p(model::KernelBanditModel, K)
    return log_p(model, K, model.α, model.β)
end

function log_p(model::KernelBanditModel, K, αs, βs)
    # Compute estimated pseudocounts
    αₖs = 1 .+ K * (αs .- 1)
    βₖs = 1 .+ K * (βs .- 1)

    # Compute probability of sucess/failure
    p_D = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

    return sum(p_D)
end

function pℓ(model::KernelBanditModel)
    return pℓ(model, model.α, model.β)
end

function pℓ(model::KernelBanditModel, α, β)
    log_ps = [log_p(model, K, α, β) for K in model.Ks]
    lsume = logsumexp(log_ps)
    log_pℓs = log_ps .- lsume
    pℓs = exp.(log_pℓs)
    return pℓs
end

function update_kernel!(model::KernelBanditModel)
    # Compute pℓ
    pℓs = pℓ(model)
    # Get quantile
    dist = Categorical(pℓs)
    q = quantile(dist, 1 - model.ℓconf)
    # Update K
    model.K = model.Ks[q]
    model.curr_ℓ = model.ℓs[q]
end