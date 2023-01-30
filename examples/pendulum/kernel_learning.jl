using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
include("controller.jl")
include("setup.jl")
include("pendulum_plotting.jl")

wsqe_kernel(r, W; ℓ=0.01) = exp(-(r' * W * r) / (2 * ℓ^2))

function pendulum_mc_model(nθ, nω, nsamps; σθ_max=0.3, σω_max=0.3)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return MonteCarloModel(grid, nsamps)
end

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0,
    ℓ=5e-3, w=[1.0, 0.04])
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)
    return KernelBanditModel(grid, k)
end

# Ground truth
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
problem_gt = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

# Small ground truth
nθ = 21
nω = 21
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

model_gt_small = pendulum_mc_model(nθ, nω, 10000; σθ_max=σθ_max, σω_max=σω_max)

function fill_in_small(model_gt, model_gt_small)
    for ind in 1:length(model_gt_small.grid)
        x = ind2x(model_gt_small.grid, ind)
        ind_orig = interpolants(model_gt.grid, x)[1][1]
        model_gt_small.α[ind] = model_gt.α[ind_orig]
        model_gt_small.β[ind] = model_gt.β[ind_orig]
        model_gt_small.pfail[ind] = model_gt.pfail[ind_orig]
    end
end

fill_in_small(model_gt, model_gt_small)
problem_gt_small = pendulum_problem(nθ, nω, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt_small, model_gt_small)

ℓ = 2e-2

model_kbrandom = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=ℓ)
set_sizes_kbrandom = run_estimation!(model_kbrandom, problem, random_acquisition, 5000, tuple_return=true)

# Kernel DKWUCB
model_kkb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=2e-2)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem.pfail_threshold,
    problem.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kkb = run_estimation!(model_kkb, problem, kernel_dkwucb_acquisition, 5000, tuple_return=true)

function compute_fpr(model::KernelBanditModel, problem_gt::GriddedProblem, ℓ; w=[1.0, 0.04])
    # Compute K
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    X_pred = [X for X in model.grid]
    K = get_K(X_pred, X_pred, k)

    αₖ = 1 .+ K * (model.α .- 1)
    βₖ = 1 .+ K * (model.β .- 1)

    is_safe = [cdf(Beta(α, β), problem_gt.pfail_threshold) > problem.conf_threshold for (α, β) in zip(αₖ, βₖ)]

    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    return isnothing(FP_inds) ? 0.0 : length(FP_inds) / length(is_safe)
end

@time compute_fpr(model_kkb, problem_gt_small, 1)

plot(1e-3:1e-3:1e-1, (x) -> compute_fpr(model_kbrandom, problem_gt_small, x),
    xlabel="ℓ", ylabel="FPR", legend=false, color=:teal, lw=3)

function plot_kernel_counts(model::KernelBanditModel, ℓ; w=[1.0, 0.04])
    # Compute K
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    X_pred = [X for X in model.grid]
    K = get_K(X_pred, X_pred, k)

    αₖ = 1 .+ K * (model.α .- 1)
    βₖ = 1 .+ K * (model.β .- 1)

    p1 = to_heatmap(model.grid, αₖ)
    p2 = to_heatmap(model.grid, βₖ)

    return plot(p1, p2)
end

plot_kernel_counts(model_kkb, 2e-2)

to_heatmap(model_kbrandom.grid, model_kbrandom.α)

function compute_size(model::KernelBanditModel, problem_gt::GriddedProblem, ℓ; w=[1.0, 0.04])
    # Compute K
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    X_pred = [X for X in model.grid]
    K = get_K(X_pred, X_pred, k)

    αₖ = 1 .+ K * (model.α .- 1)
    βₖ = 1 .+ K * (model.β .- 1)

    is_safe = [cdf(Beta(α, β), problem_gt.pfail_threshold) > problem.conf_threshold for (α, β) in zip(αₖ, βₖ)]
    return sum(is_safe)
end

plot(1e-3:1e-3:1e-1, (x) -> compute_size(model_kkb, problem_gt_small, x),
    xlabel="ℓ", ylabel="Safe Set Size", legend=false, color=:teal, lw=3)

function p_αβ(α, β, αₖ, βₖ; nθ=100)
    dist = Beta(αₖ, βₖ)
    terms = [θ^α * (1 - θ)^β * pdf(dist, θ) for θ in range(0.0, stop=1.0, length=nθ)]
    return (1 / nθ) * sum(terms)
end

function log_p(model::KernelBanditModel, K)
    # Compute estimated pseudocounts
    αₖs = 1 .+ K * (model.α .- 1)
    βₖs = 1 .+ K * (model.β .- 1)

    # Compute probability of sucess/failure
    p_D = [log(p_αβ(α, β, αₖ, βₖ) + eps()) for (α, β, αₖ, βₖ) in zip(model.α, model.β, αₖs, βₖs)]

    return sum(p_D)
end

function log_p(model::KernelBanditModel; ℓ=2e-2, w=[1.0, 0.04])
    # Compute K
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)
    X_pred = [X for X in model.grid]
    K = get_K(X_pred, X_pred, k)

    return log_p(model, K)
end

plot(1e-3:1e-3:1e-1, (x) -> log_p(model_kbrandom, ℓ=x),
    xlabel="ℓ", ylabel="log P(D ∣ ℓ)", legend=false, color=:teal, lw=3)

log_ps = [log_p(model_kkb, ℓ=x) for x in 1e-3:1e-3:1e-1]
lsume = logsumexp(log_ps)
N = length(1e-3:1e-3:1e-1)
log_pℓs = log_ps .- lsume
pℓs = exp.(log_pℓs)
sum(pℓs)
plot(1e-3:1e-3:1e-1, pℓs, legend=false, color=:teal, xlabel="ℓ", ylabel="P(ℓ | D)", lw=2)

function pℓ(model::KernelBanditModel; ℓmin=1e-3, ℓmax=1e-1, nbins=200)
    ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
    log_ps = [log_p(model, ℓ=ℓ) for ℓ in ℓs]
    lsume = logsumexp(log_ps)
    log_pℓs = log_ps .- lsume
    pℓs = exp.(log_pℓs)
    return ℓs, pℓs
end

function pℓ(model::KernelBanditModel, Ks; ℓmin=1e-3, ℓmax=1e-1, nbins=200)
    ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
    log_ps = [log_p(model, K) for K in Ks]
    lsume = logsumexp(log_ps)
    log_pℓs = log_ps .- lsume
    pℓs = exp.(log_pℓs)
    return ℓs, pℓs
end

@time ℓs, pℓs = pℓ(model_kbrandom, ℓmin=1e-4, ℓmax=1e-2, nbins=200)
plot(ℓs, pℓs, legend=false, color=:teal, xlabel="ℓ", ylabel="P(ℓ | D)", lw=2)
bar(ℓs, pℓs, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)")

function get_Ks(model::KernelBanditModel; w=[1.0, 0.04], ℓmin=1e-4, ℓmax=1e-2, nbins=200)
    X_pred = [X for X in model.grid]
    W = diagm(w ./ norm(w))

    ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
    Ks = [get_K(X_pred, X_pred, (x, x′) -> wsqe_kernel(x - x′, W, ℓ=ℓ)) for ℓ in ℓs]
    return Ks
end

@time Ks = get_Ks(model_kbrandom, ℓmin=1e-4, ℓmax=1e-2, nbins=100)
@time ℓs, pℓs = pℓ(model_kbrandom, Ks, ℓmin=1e-4, ℓmax=1e-2, nbins=100)
bar(ℓs, pℓs, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)")
dist = Categorical(pℓs)
ℓs[quantile(dist, 0.05)]

function log_p(model::KernelBanditModel, K, iter)
    # Get α and β at this iteration
    eval_inds = model.eval_inds[1:iter]
    eval_res = model.eval_res[1:iter]

    αs = zeros(length(model.grid))
    βs = zeros(length(model.grid))
    for i = 1:length(model.grid)
        eval_inds_inds = findall(eval_inds .== i)
        neval = length(eval_inds_inds)
        if neval > 0
            αs[i] = 1 + sum(eval_res[eval_inds_inds])
            βs[i] = 2 + neval - αs[i]
        else
            αs[i] = 1
            βs[i] = 1
        end
    end

    # Compute estimated pseudocounts
    αₖs = 1 .+ K * (αs .- 1)
    βₖs = 1 .+ K * (βs .- 1)

    # Compute probability of sucess/failure
    p_D = [log(p_αβ(α, β, αₖ, βₖ)) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

    return sum(p_D)
end

function pℓ(model::KernelBanditModel, Ks, iter; ℓmin=1e-3, ℓmax=1e-1, nbins=200)
    ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
    log_ps = [log_p(model, K, iter) for K in Ks]
    lsume = logsumexp(log_ps)
    log_pℓs = log_ps .- lsume
    pℓs = exp.(log_pℓs)
    return ℓs, pℓs
end

iter = 10
@time ℓs, pℓs = pℓ(model_kbrandom, Ks, iter, ℓmin=1e-4, ℓmax=1e-2, nbins=100)
bar(ℓs, pℓs, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)",
    ylims=(0, 0.15), xlims=(0, 0.01))
dist = Categorical(pℓs)
q = ℓs[quantile(dist, 0.05)]
plot!([q, q], [0, 0.15], lw=2)

anim = @animate for iter in 1:10:1000
    println(iter)
    ℓs, pℓs = pℓ(model_kkb, Ks, iter, ℓmin=1e-4, ℓmax=1e-2, nbins=100)
    bar(ℓs, pℓs, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)",
        ylims=(0, 0.15), xlims=(0, 0.01), title="Number of Episodes: $iter")
    dist = Categorical(pℓs)
    q = ℓs[quantile(dist, 0.05)]
    plot!([q, q], [0, 0.15], lw=2)
end
Plots.gif(anim, "figs/length_dist_kkba.gif", fps=10)

nothing

# function get_posterior(model::KernelBanditModel; ℓmin=1e-3, ℓmax=1e-1, nbins=200)
#     ℓs = collect(range(ℓmin, stop=ℓmax, length=nbins))
#     ps = [p(model, ℓ=ℓ) for ℓ in ℓs]
#     ps ./= sum(ps)
#     return ℓs, ps
# end

# ℓs, ps = get_posterior(model_kkb, ℓmax=5e-2)
# dist = Categorical(ps)
# ℓs[quantile(dist, 0.05)]
# compute_size(model_kkb, problem_gt_small, ℓs[quantile(dist, 0.05)])
# bar(ℓs, ps, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)")
# # plot(ps)





# function p_old(model::KernelBanditModel; ℓ=2e-2, w=[1.0, 0.04])
#     # Compute K
#     W = diagm(w ./ norm(w))
#     k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

#     X_pred = [X for X in model.grid]
#     K = get_K(X_pred, X_pred, k)

#     αₖ = 1 .+ K * (model.α .- 1)
#     βₖ = 1 .+ K * (model.β .- 1)

#     p̂fail = model.α ./ (model.α + model.β)

#     p_D = [pdf(Beta(α, β), pfail) for (α, β, pfail) in zip(αₖ, βₖ, p̂fail)]

#     return sum(p_D)
# end

# function p_o(o, αₖ, βₖ; nθ=100)
#     dist = Beta(αₖ, βₖ)
#     terms = [θ^o * (1 - θ)^(1 - o) * pdf(dist, θ) for θ in range(0.0, stop=1.0, length=nθ)]
#     return (1 / nθ) * sum(terms)
# end

# p_αβ(2, 4, 50, 100)
# ps = p_o(1, 50, 100; nθ=100)
# pf = p_o(0, 50, 100; nθ=100)
# ps^2 * pf^4

# p_αβ(2, 0, 50, 100)
# ps = p_o(1, 50, 100; nθ=100)
# ps^2