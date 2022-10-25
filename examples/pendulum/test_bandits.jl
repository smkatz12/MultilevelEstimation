using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save

include("../../src/multilevel_estimation.jl")
include("../../src/bandit.jl")
include("../../src/acquisition.jl")
include("controller.jl")
include("setup.jl")

function pendulum_bandit_model(nθ, nω; σθ_max=0.3, σω_max=0.3)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return BanditModel(grid, zeros(length(grid)), ones(length(grid)), ones(length(grid)))
end

# Set up the problem
nθ = 100
nω = 100
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.5)

# Random acquisition
nθ = 20
nω = 20
model_random = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
set_sizes_random = run_estimation!(model_random, problem, random_acquisition, 10000, log_every=100)

pfail(model, params) = interpolate(model.grid, model.pfail, params)
heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> pfail(model_random, [x, y]), xlabel="σθ", ylabel="σω")

# Max improvement acquisition
model_mi = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
max_improvement_acquisition(model) = max_improvement_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_mi = run_estimation!(model_mi, problem, max_improvement_acquisition, 10000, log_every=100)

heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> pfail(model_mi, [x, y]), xlabel="σθ", ylabel="σω")

p = plot(collect(0:100:10000), set_sizes_random, xlabel="Number of Simulations", ylabel="Safe Set Size", 
        label="random", legend=:bottomright)
plot!(p, collect(0:100:10000), set_sizes_mi, label="max improve")