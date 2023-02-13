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

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0, w=[1.0, 0.04], ℓconf=0.95)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return KernelBanditModel(grid, ℓconf=ℓconf)
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
