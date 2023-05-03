using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/pspec_bandit.jl")
include("controller.jl")
include("setup.jl")

wsqe_kernel(r, W; ℓ=0.01) = exp(-(r' * W * r) / (2 * ℓ^2))

function daa_mc_model(nx₀, ny₀, nf, nsamps; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0)
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    return MonteCarloModel(grid, nsamps)
end

function daa_pspec_bandit_model(nx₀, ny₀, nf; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0,
    ℓmin=1e-3, ℓmax=1e-1, nbins=50)
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    return PSpecBanditModel(grid, ℓmin=ℓmin, ℓmax=ℓmax, w=[4e-8, 1.0, 3.265e-5], nbins=nbins)
end

# # Ground truth
model_gt_small = BSON.load("results/ground_truth_small.bson")[:model]

nx₀ = 25
ny₀ = 25
nf = 8

problem_gt_small = daa_problem(nx₀, ny₀, nf, conf_threshold=0.95)
estimate_from_pfail!(problem_gt_small, model_gt_small)

# # Actual safe set size
sum(problem_gt_small.is_safe)

println("Creating model...")
kb_model = daa_pspec_bandit_model(nx₀, ny₀, nf, ℓmin=1e-3, ℓmax=1e-1, nbins=50)

kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem_gt_small.pfail_threshold,
    problem_gt_small.conf_threshold, rand_argmax=true, buffer=0.0)
reset!(kb_model)

println("Running estimation...")
ss_kb = run_estimation!(kb_model, problem_gt_small, kernel_dkwucb_acquisition, 100000,
    tuple_return=true)

function get_rates(model)
    is_safe = falses(length(model.grid))
    for i = 1:length(model.grid)
        conf_ind = findfirst(cumsum(model.θdists[i, :]) .> 0.95)
        pfail_ind = findfirst(model.θs .>= 0.3)
        is_safe[i] = conf_ind ≤ pfail_ind
    end

    FN_inds = findall(.!is_safe .& problem_gt_small.is_safe)
    FP_inds = findall(is_safe .& .!problem_gt_small.is_safe)

    return length(FN_inds), length(FP_inds)
end

FNs, FPs = get_rates(kb_model)