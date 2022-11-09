using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors

mycmap = ColorScheme([RGB{Float64}(0.5, 1.5 * 0.5, 2.0 * 0.5),
    RGB{Float64}(0.25, 1.5 * 0.25, 2.0 * 0.25),
    RGB{Float64}(227 / 255, 27 / 255, 59 / 255),
    RGB{Float64}(0.0, 0.0, 0.0)])
mycmap_small = ColorScheme([RGB{Float64}(0.25, 1.5 * 0.25, 2.0 * 0.25),
    RGB{Float64}(0.0, 0.0, 0.0)])

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/bandit.jl")
include("controller.jl")
include("setup.jl")

function pendulum_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return BanditModel(grid)
end

# Ground truth
# model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
# problem_gt = pendulum_problem(100, 100, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
# estimate_from_pfail!(problem_gt, model_gt)

# Set up the problem
nθ = 101
nω = 101
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

# Random acquisition
nsamps = 50000
model_random = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
set_sizes_random = run_estimation!(model_random, problem, random_acquisition, nsamps)

p = plot(collect(0:nsamps), set_sizes_random, label="random", legend=:topleft, linetype=:steppre)

# Max improvement acquisition
nsamps = 50000
model_mi = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
max_improvement_acquisition(model) = max_improvement_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_mi = run_estimation!(model_mi, problem, max_improvement_acquisition, nsamps)

plot!(p, collect(0:nsamps), set_sizes_mi, label="Max Improvement", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

function to_heatmap(grid::RectangleGrid, vals; kwargs...)
    vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]))
    return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat'; kwargs...)
end

to_heatmap(model_random.grid, model_random.α .+ model_random.β, c=:thermal)
to_heatmap(model_mi.grid, model_mi.α .+ model_mi.β, c=:thermal)

function plot_eval_points(model::BanditModel)
    xs = [pt[1] for pt in model.grid]
    ys = [pt[2] for pt in model.grid]
    p = scatter(xs, ys, legend=false,
        markersize=0.5, markercolor=:black, markerstrokecolor=:black)

    xs_eval = [ind2x(model.grid, i)[1] for i in unique(model.eval_inds)]
    ys_eval = [ind2x(model.grid, i)[2] for i in unique(model.eval_inds)]
    scatter!(p, xs_eval, ys_eval,
        markersize=2.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω")
    return p
end

plot_eval_points(model_random)
plot_eval_points(model_mi)

plot(model_mi.eval_inds)