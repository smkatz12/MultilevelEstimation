using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("controller.jl")
include("setup.jl")

# Ground Truth Monte Carlo Estimator
function pendulum_mc_model(nθ, nω, nsamps; σθ_max=0.3, σω_max=0.3)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid_points = Dict(:σθs => σθs, :σωs => σωs)
    grid = RectangleGrid(σθs, σωs)

    return MonteCarloModel(grid, nsamps, zeros(length(grid)), ones(length(grid)), ones(length(grid)))
end

# nθ = 100
# nω = 100
# σθ_max = 0.2
# σω_max = 1.0
# nsamps = 10000
# problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
# model = pendulum_mc_model(nθ, nω, nsamps, σθ_max=σθ_max, σω_max=σω_max)

# @time run_estimation!(model, problem)

# @save "results/ground_truth.bson" model

# model = BSON.load("examples/pendulum/results/ground_truth_0p2.bson")[:model]
# problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)

# pfail(model, params) = interpolate(model.grid, model.pfail, params)
# heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> pfail(model, [x, y]), xlabel="σθ", ylabel="σω")

# issafe(problem, params) = interpolate(problem.grid, problem.is_safe, params)

# estimate_from_pfail!(problem, model)
# heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> issafe(problem, [x, y]), xlabel="σθ", ylabel="σω")

# estimate_from_counts!(problem, model)
# heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> issafe(problem, [x, y]), xlabel="σθ", ylabel="σω")