using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("controller.jl")
include("setup.jl")

# Ground Truth Monte Carlo Estimator
function daa_mc_model(nx₀, ny₀, nf, nsamps; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0)
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    return MonteCarloModel(grid, nsamps)
end

nx₀ = 25
ny₀ = 25
nf = 8
nsamps = 10000

problem = daa_problem(nx₀, ny₀, nf, fmin=30.0)
model = daa_mc_model(nx₀, ny₀, nf, nsamps, fmin=30.0)

@time run_estimation!(model, problem, mc_acquisition, nsamps * length(model.grid))

@save "examples/daa/results/ground_truth_small.bson" model

# Plotting

# pfail(model, params) = interpolate(model.grid, model.pfail, params)
# heatmap(problem.grid_points[:x₀s], problem.grid_points[:y₀s], (x, y) -> pfail(model, [x, y, 80.0]), xlabel="x₀", ylabel="y₀")
# heatmap(problem.grid_points[:x₀s], problem.grid_points[:fs], (x, y) -> pfail(model, [x, 1.0, y]), xlabel="x₀", ylabel="hfov")
# heatmap(problem.grid_points[:y₀s], problem.grid_points[:fs], (x, y) -> pfail(model, [2000.0, x, y]), xlabel="y₀", ylabel="hfov")

# xs = []
# ys = []
# zs = []
# for i in 1:100000 #1:length(model.grid)
#     x = rand(Uniform(1000, 3000))
#     y = rand(Uniform(0.8, 1.2))
#     f = rand(Uniform(40.0, 100.0))
#     # if model.pfail[i] < problem.pfail_threshold
#     if pfail(model, [x, y, f]) < problem.pfail_threshold
#         # push!(xs, model.grid[i][1])
#         # push!(ys, model.grid[i][2])
#         # push!(zs, model.grid[i][3])
#         push!(xs, x)
#         push!(ys, y)
#         push!(zs, f)
#     end
# end

# plotlyjs()
# scatter3d(xs, ys, zs, xlabel="x₀", ylabel="y₀", zlabel="hfov", markersize=1.0)
# gr()

# function plot_pfail_slices(model)
#     hfovs = collect(30:10:100)
#     ps = [heatmap(problem.grid_points[:x₀s], problem.grid_points[:y₀s], (x, y) -> pfail(model, [x, y, hfov]), 
#             xlabel="x₀", ylabel="y₀", clim=(0,1), colorbar=false, title="hfov=$hfov") for hfov in hfovs]
#     return plot(ps..., layout=(2, 4), size=(800, 450))
# end

# plot_pfail_slices(model)

# function plot_safe_slices(model)
#     hfovs = collect(30:10:100)
#     ps = [heatmap(problem.grid_points[:x₀s], problem.grid_points[:y₀s], (x, y) -> pfail(model, [x, y, hfov]) < 0.3,
#         xlabel="x₀", ylabel="y₀", clim=(0, 1), colorbar=false, title="hfov=$hfov") for hfov in hfovs]
#     return plot(ps..., layout=(2, 4), size=(800, 450))
# end
# plot_safe_slices(model)

# # Get an idea of what is needed to converge estimate

# x₀ = 1500
# y₀ = 1.0
# hfov = 80.0

# ns = collect(0:100:10000)
# ps = zeros(length(ns))

# for i in ProgressBar(1:length(ns))
#     nsamples = ns[i]
#     res = problem.sim([x₀, y₀, hfov], nsamples)
#     ps[i] = res / nsamples
# end

# plot(ns, ps, ylims=(0,1))