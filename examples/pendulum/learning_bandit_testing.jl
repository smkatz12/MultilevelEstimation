using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
include("../../src/learning_bandit.jl")
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

function pendulum_learning_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0, ℓconf=0.95,
    ℓmin=1e-4, ℓmax=1e-2)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return LearningBanditModel(grid, ℓconf=ℓconf, ℓmin=ℓmin, ℓmax=ℓmax)
end

# Ground truth
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
# Fix for bug when ground truth was generated
model_gt.β = model_gt.nsamps .+ model_gt.β .- 1
problem_gt = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

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

sum(problem_gt_small.is_safe) # True safe set size

# Random acquisition
model_random = pendulum_learning_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_random)
set_sizes_random = run_estimation!(model_random, problem_gt_small, random_acquisition, 20000,
    tuple_return=true)

# Kernel DKWUCB acquisition
model_kkb = pendulum_learning_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓmax=1e-1)
reset!(model_kkb)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem_gt_small.pfail_threshold,
    problem_gt_small.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kkb = run_estimation!(model_kkb, problem_gt_small, kernel_dkwucb_acquisition, 20000,
    tuple_return=true)

plot(collect(1:20000), model_kkb.ℓests, xlabel="Number of Episodes", ylabel="ℓ",
    color=:magenta, lw=2, legend=false, ylims=(1e-4, 1e-1))

set_sizes_nk = [s[1] for s in set_sizes_kkb]
set_sizes_k = [s[2] for s in set_sizes_kkb]

iter = 100000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 100000), ylims=(0, 320))
plot!(p1, [0.0, 100000.0], [281, 281], linestyle=:dash, lw=3, color=:black, label="True Size")

plot_learning_summary(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, 1000)

create_learning_gif(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, "kkb_100000_learning.gif",
    max_iter=100000, plt_every=500, fps=10, θmax=σθ_max, ωmax=σω_max)

# Checking pθ technique
function plot_pθ(model, pt; θs=range(0, stop=1.0, length=101))
    sps, ps = interpolants(model.grid, pt)
    ind = sps[argmax(ps)]
    p = plot(θs, (x) -> pdf(Beta(model.α[ind], model.β[ind]), x),
        xlabel="θ", ylabel="p(θ)", legend=false, ylims=(0,), xlims=(0,),
        title="p(θ)")
    return p
end

function plot_pθest(model, pt; θs=range(0, stop=1.0, length=101))
    sps, ps = interpolants(model.grid, pt)
    ind = sps[argmax(ps)]
    p = plot(θs, (x) -> pdf(Beta(model.αest[ind], model.βest[ind]), x),
        xlabel="θ", ylabel="p(θ)", legend=false, ylims=(0,), xlims=(0,),
        title="p_est(θ)")
    return p
end

plot_pθ(model_kkb, [0.01, 0.05])
plot_pθest(model_kkb, [0.01, 0.05])
plot_pθ(model_kkb, [0.05, 0.25])
plot_pθest(model_kkb, [0.05, 0.25])
plot_pθ(model_kkb, [0.09, 0.45])
plot_pθest(model_kkb, [0.09, 0.45])

function compute_θdist(model, pt; θs=range(0, stop=1.0, length=101))
    sps, ps = interpolants(model.grid, pt)
    ind = sps[argmax(ps)]

    αs, βs = model.αₖ[ind, :], model.βₖ[ind, :]

    ps = [sum([pℓ * pdf(Beta(α, β), θ) for (α, β, pℓ) in zip(αs, βs, model.curr_pℓs)]) for θ in θs]
    ps ./= sum(ps)
    return ps
end

function plot_θdist_v2(model, pt; θs=range(0, stop=1.0, length=101))
    ps = compute_θdist(model, pt, θs=θs)
    return bar(θs, ps, legend=false, color=:teal, lw=0.25, xlabel="θ",
        ylabel="P(θ)", ylims=(0,), xlims=(0,),
        title="p_est(θ) v2")
end

plot_pθ(model_random, [0.01, 0.05])
plot_pθest(model_random, [0.01, 0.05])
plot_θdist_v2(model_random, [0.01, 0.05])

function plot_comparison(model, pt; θs=range(0, stop=1.0, length=101))
    p1 = plot_pθ(model, pt, θs=θs)
    p2 = plot_pθest(model, pt, θs=θs)
    p3 = plot_θdist_v2(model, pt, θs=θs)

    return plot(p1, p2, p3)
end

plot_comparison(model_random, [0.01, 0.05])
plot_comparison(model_random, [0.09, 0.45])

# Messing around with point-specific length parameter
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

function plot_pℓ(model::LearningBanditModel, pspec_pℓs, pt)
    sps, ps = interpolants(model.grid, pt)
    ind = sps[argmax(ps)]

    p = bar(model.ℓs, pspec_pℓs[ind, :], legend=false, color=:teal, lw=0.25, xlabel="ℓ",
        ylabel="P(ℓ ∣ D)", ylims=(0, maximum(pspec_pℓs[ind, :]) + 0.02), xlims=(0, model.ℓs[end]))
    return p
end

# Trying on ground truth
αₖs = reshape(1 .+ model_random.Ks * (model_gt_small.α .- 1), length(model_gt_small.grid), length(model_random.ℓs))
βₖs = reshape(1 .+ model_random.Ks * (model_gt_small.β .- 1), length(model_gt_small.grid), length(model_random.ℓs))

pspec_pℓs = pspec_pℓ(model_gt_small.α, model_gt_small.β, αₖs, βₖs)

plot_pℓ(model_random, pspec_pℓs, [0.01, 0.05])
plot_pℓ(model_random, pspec_pℓs, [0.09, 0.45])
plot_pℓ(model_random, pspec_pℓs, [0.074, 0.4])

vals = [dot(model_random.ℓs, pspec_pℓs[i, :]) for i = 1:length(model_random.grid)]
to_heatmap(model_random.grid, vals, xlabel="σθ", ylabel="σω")

# Try on slightly larger ground truth
# model_full = pendulum_learning_bandit_model(36, 36, σθ_max=0.14, σω_max=0.7)
# model_full = pendulum_learning_bandit_model(26, 26, σθ_max=0.1, σω_max=0.5,
#     ℓmin=1e-3, ℓmax=5e-2) # Interesting results with this
model_full = pendulum_learning_bandit_model(26, 26, σθ_max=0.1, σω_max=0.5,
    ℓmin=1e-3, ℓmax=2e-2)
# model_gt_full = pendulum_mc_model(36, 36, 10000; σθ_max=0.14, σω_max=0.7)
model_gt_full = pendulum_mc_model(26, 26, 10000; σθ_max=0.1, σω_max=0.5)
fill_in_small(model_gt, model_gt_full)
αₖs = reshape(1 .+ model_full.Ks * (model_gt_full.α .- 1), length(model_full.grid), length(model_full.ℓs))
βₖs = reshape(1 .+ model_full.Ks * (model_gt_full.β .- 1), length(model_full.grid), length(model_full.ℓs))

pspec_pℓs = pspec_pℓ(model_gt_full.α, model_gt_full.β, αₖs, βₖs)

plot_pℓ(model_full, pspec_pℓs, [0.01, 0.05])
plot_pℓ(model_full, pspec_pℓs, [0.09, 0.45])
plot_pℓ(model_full, pspec_pℓs, [0.075, 0.4])

vals = [dot(model_full.ℓs, pspec_pℓs[i, :]) for i = 1:length(model_full.grid)]
to_heatmap(model_full.grid, vals, xlabel="σθ", ylabel="σω")

function plot_frame(model, model_gt, problem_gt, pspec_pℓs, pt)
    sps, ps = interpolants(model.grid, pt)
    curr_ind = sps[argmax(ps)]
    function get_heat(x, y)
        sps, ps = interpolants(model.grid, [x, y])
        ind = sps[argmax(ps)]
        if ind == curr_ind
            return 0.5
        else
            return interpolate(model_gt.grid, model_gt.pfail, [x, y])
        end
    end
    p1 = heatmap(problem_gt.grid_points[:σθs], problem_gt.grid_points[:σωs], (x, y) -> get_heat(x, y),
        xlabel="σθ", ylabel="σω")
    p2 = bar(model.ℓs, pspec_pℓs[curr_ind, :], legend=false, color=:teal, lw=0.25, xlabel="ℓ",
        ylabel="P(ℓ ∣ D)", ylims=(0, maximum(pspec_pℓs[curr_ind, :]) + 0.02), xlims=(0, model.ℓs[end]))
    return plot(p1, p2, size=(800, 300))
end

function plot_frame_ind(model, model_gt, problem_gt, pspec_pℓs, curr_ind)
    function get_heat(x, y)
        sps, ps = interpolants(model.grid, [x, y])
        ind = sps[argmax(ps)]
        if ind == curr_ind
            return 0.5
        else
            return interpolate(model_gt.grid, model_gt.pfail, [x, y])
        end
    end
    p1 = heatmap(problem_gt.grid_points[:σθs], problem_gt.grid_points[:σωs], (x, y) -> get_heat(x, y),
        xlabel="σθ", ylabel="σω")
    p2 = bar(model.ℓs, pspec_pℓs[curr_ind, :], legend=false, color=:teal, lw=0.25, xlabel="ℓ",
        ylabel="P(ℓ ∣ D)", ylims=(0, maximum(pspec_pℓs[curr_ind, :]) + 0.02), xlims=(0, model.ℓs[end]))
    return plot(p1, p2, size=(800, 300))
end

p = plot_frame(model_full, model_gt_full, problem_gt_small, pspec_pℓs, [0.01, 0.05])
p = plot_frame_ind(model_full, model_gt_full, problem_gt_small, pspec_pℓs, 420)


anim = @animate for ind in 417:442 #1:length(model_full.grid)
    println(ind)
    plot_frame_ind(model_full, model_gt_full, problem_gt_small, pspec_pℓs, ind)
end
Plots.gif(anim, "figs/pspec_dists_row.gif", fps=2)

# What happens to distribution as we change ℓ
function plot_from_ℓ(ℓs, αₖs, βₖs, ind, ℓind)
    α = αₖs[ind, ℓind]
    β = βₖs[ind, ℓind]
    p = plot(0:0.0001:1, (x) -> pdf(Beta(α, β), x), xlabel="θ", ylabel="p(θ)",
        title="ℓ=$(round(ℓs[ℓind], digits=4))", label="Kernel Dist", lw=2)
    plot!(p, 0:0.0001:1, (x) -> pdf(Beta(model_gt_full.α[ind], model_gt_full.β[ind]), x),
        label="True Dist", lw=2)
    return p
end

function plot_from_ℓpspec(ℓs, αₖs, βₖs, ind, ℓind, pspec_pℓs)
    α = αₖs[ind, ℓind]
    β = βₖs[ind, ℓind]
    p1 = plot(0:0.0001:1, (x) -> pdf(Beta(α, β), x), xlabel="θ", ylabel="p(θ)",
        title="ℓ=$(round(ℓs[ℓind], digits=4))", label="Kernel Dist", lw=2)
    plot!(p1, 0:0.0001:1, (x) -> pdf(Beta(model_gt_full.α[ind], model_gt_full.β[ind]), x),
        label="True Dist", lw=2)
    p2 = bar(ℓs[1:ℓind], pspec_pℓs[ind, 1:ℓind], legend=false, color=:teal, lw=0.25, xlabel="ℓ",
        ylabel="P(ℓ ∣ D)", ylims=(0, maximum(pspec_pℓs[ind, :]) + 0.02), xlims=(0, ℓs[end]))
    p3 = plot(ℓs[1:ℓind], αₖs[ind, 1:ℓind], xlabel="ℓ", ylabel="αest",
            xlims=(0, ℓs[end]), ylims=(0, maximum(αₖs[ind, :]) + 10), legend=false)
    p4 = plot(ℓs[1:ℓind], βₖs[ind, 1:ℓind], xlabel="ℓ", ylabel="βest",
            xlims=(0, ℓs[end]), ylims=(0, maximum(βₖs[ind, :]) + 10), legend=false)
    return plot(p1, p2, p3, p4, layout=(4, 1), size=(600, 700))
end

p = plot_from_ℓpspec(model_full.ℓs, αₖs, βₖs, 437, 75, pspec_pℓs)

anim = @animate for ℓind in 2:100
    println(ℓind)
    plot_from_ℓpspec(model_full.ℓs, αₖs, βₖs, 420, ℓind, pspec_pℓs)
end
Plots.gif(anim, "figs/ptheta_change_420.gif", fps=4)

# Messing around with timing
n = 121
α, β = model_gt_full.α, model_gt_full.β
αnew = copy(α)
αnew[n] += 1
@time αₖsnew = reshape(1 .+ model_full.Ks * (αnew .- 1), length(model_full.grid), length(model_full.ℓs))
@time αₖsnew_v2 = reshape(model_full.Ks[:, n], length(model_full.grid), length(model_full.ℓs)) .+ αₖs
sum(abs.(αₖsnew .- αₖsnew_v2))

# Random acq
model_random.αₖ = reshape(1 .+ model_random.Ks * (model_random.α .- 1), length(model_random.grid), length(model_random.ℓs))
model_random.βₖ = reshape(1 .+ model_random.Ks * (model_random.β .- 1), length(model_random.grid), length(model_random.ℓs))

pspec_pℓs = pspec_pℓ(model_random.α, model_random.β, model_random.αₖ, model_random.βₖ)

pfail(model, params) = interpolate(model.grid, model.pfail, params)
heatmap(problem_gt_small.grid_points[:σθs], problem_gt_small.grid_points[:σωs],
    (x, y) -> pfail(model_gt_small, [x, y]), xlabel="σθ", ylabel="σω")

plot_pℓ(model_random, pspec_pℓs, [0.09, 0.45])
plot_ℓdist(model_random, 100000)

vals = [dot(model_random.ℓs, pspec_pℓs[i, :]) for i = 1:length(model_random.grid)]
to_heatmap(model_random.grid, vals, xlabel="σθ", ylabel="σω")

pspec_pℓs = pspec_pℓ(model_kkb.α, model_kkb.β, model_kkb.αₖ, model_kkb.βₖ)

plot_pℓ(model_kkb, pspec_pℓs, [0.01, 0.05])
plot_ℓdist(model_kkb, 100000)

plot_eval_points(model_kkb, 100000, θmax=σθ_max, ωmax=σω_max, include_grid=false)

vals = [dot(model_kkb.ℓs, pspec_pℓs[i, :]) for i = 1:length(model_kkb.grid)]
to_heatmap(model_kkb.grid, vals, xlabel="σθ", ylabel="σω")