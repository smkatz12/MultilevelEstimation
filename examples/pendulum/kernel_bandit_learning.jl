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

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0, w=[1.0, 0.04])
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return KernelBanditModel(grid)
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

# Actual estimation
model_kbrandom = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_kbrandom)
set_sizes_kbrandom = run_estimation!(model_kbrandom, problem, random_acquisition, 20000,
    tuple_return=true, update_kernel_every=500)

plot(collect(1:20000), model_kbrandom.ℓests, xlabel="Number of Episodes", ylabel="ℓ",
    color=:magenta, lw=2, legend=false)

set_sizes_nk = [s[1] for s in set_sizes_kbrandom]
set_sizes_k = [s[2] for s in set_sizes_kbrandom]

iter = 20000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Random", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Random", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 150))
plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

model_kkb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_kkb)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem.pfail_threshold,
    problem.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kkb = run_estimation!(model_kkb, problem, kernel_dkwucb_acquisition, 20000,
    tuple_return=true, update_kernel_every=500)

plot(collect(1:20000), model_kkb.ℓests, xlabel="Number of Episodes", ylabel="ℓ",
    color=:magenta, lw=2, legend=false)

set_sizes_nk = [s[1] for s in set_sizes_kkb]
set_sizes_k = [s[2] for s in set_sizes_kkb]

iter = 20000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 150))
plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

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

compute_fpr(model_kbrandom, problem_gt_small, model_kbrandom.curr_ℓ)
compute_fpr(model_kkb, problem_gt_small, model_kkb.curr_ℓ)

