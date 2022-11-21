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
include("../../src/gittens.jl")
include("controller.jl")
include("setup.jl")

function pendulum_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return BanditModel(grid)
end

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

function to_heatmap(grid::RectangleGrid, vals; kwargs...)
    vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]))
    return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat'; kwargs...)
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

# Gittens Allocation Index Acquistion
nsamps = 50000
model_gi = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
gi = BSON.load("src/gittens_data/gi_1000pulls_beta99.bson")[:gi]
gi_acquisition(model) = gittens_acquisition(model, problem.pfail_threshold, problem.conf_threshold, gi)
set_sizes_gi = run_estimation!(model_gi, problem, gi_acquisition, nsamps)

model_gie = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
gie = BSON.load("src/gittens_data/gi_1000pulls_beta9999.bson")[:gi]
gi_acquisition(model) = gittens_acquisition(model, problem.pfail_threshold, problem.conf_threshold, gie)
set_sizes_gie = run_estimation!(model_gie, problem, gi_acquisition, nsamps)

model_gier = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
gie = BSON.load("src/gittens_data/gi_1000pulls_beta9999.bson")[:gi]
gi_acquisition(model) = gittens_acquisition(model, problem.pfail_threshold, problem.conf_threshold, gie, rand_argmax=true)
set_sizes_gie = run_estimation!(model_gie, problem, gi_acquisition, nsamps)

plot!(p, collect(0:nsamps), set_sizes_gi, label="Gittens 0.99", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

plot!(p, collect(0:nsamps), set_sizes_gie, label="Gittens 0.9999", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

plot_eval_points(model_random)
plot_eval_points(model_gi)
plot_eval_points(model_gie)

to_heatmap(model_random.grid, model_random.α .+ model_random.β, c=:thermal)
to_heatmap(model_gi.grid, model_gi.α .+ model_gi.β, c=:thermal)
to_heatmap(model_gi.grid, model_gie.α .+ model_gie.β, c=:thermal)

function get_heat(x, y)
    xi = convert(Int64, round(x))
    yi = convert(Int64, round(y))
    if xi + yi >= gi.npulls
        return 0.954
    elseif xi == 0 || yi == 0
        return 0.954
    else
        return gie(xi, yi)
    end
end

heatmap(0:999, 0:999, get_heat, xlabel="α", ylabel="β")
plot(1:999, (x) -> gi(x, 1), legend=false, xlabel="α", ylabel="gi(α, 1)")

plot(collect(0:49999), model_gi.eval_inds)
eval_xs = [ind2x(model_gi.grid, model_gi.eval_inds[i])[1] for i = 1:length(model_gi.eval_inds)]
plot(collect(0:49999), eval_xs)

nothing
# # LCB acquisition
# nsamps = 50000
# model_lcb = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
# lcb_acquisition(model) = lcb_acquisition(model, problem.pfail_threshold, problem.conf_threshold, c=0.1)
# set_sizes_lcb = run_estimation!(model_lcb, problem, lcb_acquisition, nsamps)

# plot!(p, collect(0:nsamps), set_sizes_lcb, label="LCB", legend=:topleft, linetype=:steppre,
#     xlabel="Number of Episodes", ylabel="Safe Set Size")

# function to_heatmap(grid::RectangleGrid, vals; kwargs...)
#     vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]))
#     return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat'; kwargs...)
# end

# to_heatmap(model_random.grid, model_random.α .+ model_random.β, c=:thermal)
# to_heatmap(model_lcb.grid, model_lcb.α .+ model_random.β, c=:thermal)

# function plot_eval_points(model::BanditModel)
#     xs = [pt[1] for pt in model.grid]
#     ys = [pt[2] for pt in model.grid]
#     p = scatter(xs, ys, legend=false,
#         markersize=0.5, markercolor=:black, markerstrokecolor=:black)

#     xs_eval = [ind2x(model.grid, i)[1] for i in unique(model.eval_inds)]
#     ys_eval = [ind2x(model.grid, i)[2] for i in unique(model.eval_inds)]
#     scatter!(p, xs_eval, ys_eval,
#         markersize=2.0, markercolor=:green, markerstrokecolor=:green,
#         xlabel="σθ", ylabel="σω")
#     return p
# end

# plot_eval_points(model_random)
# plot_eval_points(model_lcb)

# plot(model_lcb.eval_inds)