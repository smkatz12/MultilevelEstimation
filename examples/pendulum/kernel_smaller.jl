using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns
using SpecialFunctions

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

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0, w=[1.0, 0.04], ℓconf=0.95,
    ℓmin=1e-4, ℓmax=1e-2)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return KernelBanditModel(grid, ℓconf=ℓconf, ℓmin=ℓmin, ℓmax=ℓmax)
end

# Ground truth
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
problem_gt = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

pfail(model, params) = interpolate(model.grid, model.pfail, params)
heatmap(problem_gt.grid_points[:σθs][1:40], problem_gt.grid_points[:σωs][1:40], (x, y) -> pfail(model_gt, [x, y]), xlabel="σθ", ylabel="σω")

problem_gt_low = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95, threshold=0.001)
estimate_from_pfail!(problem_gt_low, model_gt)
to_heatmap(model_gt.grid, problem_gt_low.is_safe)

# Smaller ground truth
nθ = 26
nω = 26
σθ_max = 0.1
σω_max = 0.5
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
problem_gt_small = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95, threshold=0.001)
estimate_from_pfail!(problem_gt_small, model_gt_small)

to_heatmap(model_gt_small.grid, problem_gt_small.is_safe)
pfail(model, params) = interpolate(model.grid, model.pfail, params)
heatmap(problem_gt_small.grid_points[:σθs], problem_gt_small.grid_points[:σωs], (x, y) -> pfail(model_gt_small, [x, y]), xlabel="σθ", ylabel="σω")

sum(problem_gt_small.is_safe) # True safe set size

# Random acquisition
model_kbrandom = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_kbrandom)
set_sizes_kbrandom = run_estimation!(model_kbrandom, problem_gt_small, random_acquisition, 20000,
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
plot!(p1, [0.0, 20000.0], [281, 281], linestyle=:dash, lw=3, color=:black, label="True Size")

# DKWUCB acquisition
model_kkb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓmax=1e-1)
reset!(model_kkb)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem_gt_small.pfail_threshold,
    problem_gt_small.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kkb = run_estimation!(model_kkb, problem_gt_small, kernel_dkwucb_acquisition, 100000,
    tuple_return=true, update_kernel_every=10)

plot(collect(1:100000), model_kkb.ℓests, xlabel="Number of Episodes", ylabel="ℓ",
    color=:magenta, lw=2, legend=false)

set_sizes_nk = [s[1] for s in set_sizes_kkb]
set_sizes_k = [s[2] for s in set_sizes_kkb]

iter = 100000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 100000), ylims=(0, 300))
plot!(p1, [0.0, 100000.0], [281, 281], linestyle=:dash, lw=3, color=:black, label="True Size")

plot_kb_learning_summary(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, 20000,
    θmax=σθ_max, ωmax=σω_max)

create_kb_learning_gif(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, "kkb_100000_bugfix.gif",
    max_iter=100000, plt_every=500, fps=10, θmax=σθ_max, ωmax=σω_max)

# No kernel DKWUCB
model_kb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓmax=1e-1)
reset!(model_kb)
dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem_gt_small.pfail_threshold,
    problem_gt_small.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kb = run_estimation!(model_kb, problem_gt_small, dkwucb_acquisition, 100000,
    tuple_return=true, update_kernel_every=Inf)

set_sizes_baseline = [s[1] for s in set_sizes_kb]
set_sizes_kbaseline = [s[2] for s in set_sizes_kb]

iter = 100000
p1 = plot(collect(0:iter), set_sizes_baseline[1:iter+1],
    label="DKWUCB", legend=:bottomright, color=:gray, lw=4)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 100000), ylims=(0, 300))
plot!(p1, [0.0, 100000.0], [281, 281], linestyle=:dash, lw=3, color=:black, label="True Size")

@time ps = pℓ(model_kkb);
@time ps2 = pℓ_vec(model_kkb);

@time [1 .+ K * (model_kkb.α .- 1) for K in model_kkb.Ks];
Kmat = cat(model_kkb.Ks..., dims=1);
@time Kmat * model_kkb.α;