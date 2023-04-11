using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/bandit.jl")

function daa_mc_model(nx₀, ny₀, nf, nsamps; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0)
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    return MonteCarloModel(grid, nsamps)
end

function daa_bandit_model(nx₀, ny₀, nf; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0)
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    return BanditModel(grid)
end

# Ground truth
model_gt = BSON.load("examples/daa/results/ground_truth.bson")[:model]

# Small ground truth
nx₀ = 25
ny₀ = 25
nf = 8

model_gt_small = daa_mc_model(nx₀, ny₀, nf, 10000)

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
problem_gt_small = daa_problem(nx₀, ny₀, nf, conf_threshold=0.95)
estimate_from_pfail!(problem_gt_small, model_gt_small)

# Actual safe set size
sum(problem_gt_small.is_safe)

nsamps_tot = 50000

# Random acquisition
model_bandit = daa_bandit_model(nx₀, ny₀, nf)
ss_random = run_estimation!(model_bandit, problem_gt_small, random_acquisition, nsamps_tot)

# DKWUCB acquisition
model_dkwucb = daa_bandit_model(nx₀, ny₀, nf)
dkwucb_acquisition(model) = dkwucb_acquisition(model, problem_gt_small.pfail_threshold, problem_gt_small.conf_threshold)
ss_dkwucb = run_estimation!(model_dkwucb, problem_gt_small, dkwucb_acquisition, nsamps_tot)

p = plot(collect(0:50000), ss_random,
    label="Random", legend=:topleft, color=:gray, lw=2)
plot!(p, collect(0:50000), ss_dkwucb,
    label="DKWUCB", legend=:topleft, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), ylims=(0, 1100),
    xticks=[10000, 30000, 50000])
plot!(p, [0.0, 50000], [999, 999], linestyle=:dash, lw=3, color=:black, label="True Size")