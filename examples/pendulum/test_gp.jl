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
include("../../src/kernel_bandit.jl")
include("../../src/gaussian_process.jl")
include("../../src/acquisition.jl")
include("controller.jl")
include("setup.jl")

wsqe_kernel(r, W; ℓ=0.01) = exp(-(r' * W * r) / (2 * ℓ^2))

function pendulum_gp_model(nθ, nω; σθ_max=0.2, σω_max=1.0,
    ℓ=5e-3, nsamps=500, ν=0.001, w=[1.0, 0.04])
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    # Set up the mean and kernel functions
    m(x) = zeros(length(x)) #0.5 * ones(length(x))
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    return GaussianProcessModel(grid, nsamps, m, k, ν)
end

# Plot where evaluated
function plot_eval_points(model::GaussianProcessModel)
    xs = [pt[1] for pt in model.grid]
    ys = [pt[2] for pt in model.grid]
    p = scatter(xs, ys, legend=false,
        markersize=0.5, markercolor=:black, markerstrokecolor=:black)

    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds]
    scatter!(p, xs_eval, ys_eval,
        markersize=2.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω")
    return p
end

function to_heatmap(grid::RectangleGrid, vals; kwargs...)
    vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]))
    return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat'; kwargs...)
end

# Plot predicted pfail mean and cov
function plot_predictions(model::GaussianProcessModel)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, all_X, all_inds, model.K)

    p1 = to_heatmap(model.grid, μ, title="μ", xlabel="σθ", ylabel="σω")
    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds]
    scatter!(p1, xs_eval, ys_eval,
        markersize=2.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    p2 = to_heatmap(model.grid, sqrt.(σ²), title="σ", xlabel="σθ", ylabel="σω")
    scatter!(p2, xs_eval, ys_eval,
        markersize=2.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    return plot(p1, p2, size=(1000, 400))
end

function plot_test_stats(model::GaussianProcessModel, conf_threshold)
    β = quantile(Normal(), conf_threshold)

    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, all_X, all_inds, model.K)

    p1 = to_heatmap(model.grid, μ .+ β .* sqrt.(σ²), title="Test Statistic", xlabel="σθ", ylabel="σω")
    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds]
    scatter!(p1, xs_eval, ys_eval,
        markersize=2.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    return p1
end

function plot_safe_set(model::GaussianProcessModel, problem_gt::GriddedProblem)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X, model.X_inds, model.y, all_X, all_inds, model.K)
    β = quantile(Normal(), problem.conf_threshold)
    is_safe = (μ .+ β .* sqrt.(σ²)) .< problem.pfail_threshold

    colors = zeros(length(is_safe))
    # TP_inds = findall(is_safe .& problem_gt.is_safe)
    FN_inds = findall(.!is_safe .& problem_gt.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.5 : nothing
    FP_inds = finalize(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.75 : nothing

    p = to_heatmap(model.grid, colors, c=cgrad(mycmap, 4, categorical=true),
        colorbar=:none)
    p
end

function plot_summary(model::GaussianProcessModel, problem::GriddedProblem, iter)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)

    p1 = to_heatmap(model.grid, μ, title="μ", xlabel="σθ", ylabel="σω")
    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
    scatter!(p1, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    p2 = to_heatmap(model.grid, sqrt.(σ²), title="σ", xlabel="σθ", ylabel="σω")
    scatter!(p2, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    β = quantile(Normal(), problem.conf_threshold)

    p3 = to_heatmap(model.grid, μ .+ β .* sqrt.(σ²), title="Test Statistic", xlabel="σθ", ylabel="σω")
    scatter!(p3, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    is_safe = (μ .+ β .* sqrt.(σ²)) .< problem.pfail_threshold
    p4 = to_heatmap(model.grid, is_safe, title="Estimated Safe Set", xlabel="σθ", ylabel="σω")
    scatter!(p4, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    return plot(p1, p2, p3, p4)
end

function plot_summary(model::GaussianProcessModel, problem::GriddedProblem)
    return plot_summary(model, problem, length(model.X))
end

function plot_summary_gt(model::GaussianProcessModel, problem::GriddedProblem, iter)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)

    p1 = to_heatmap(model.grid, μ, title="μ", xlabel="σθ", ylabel="σω", c=:thermal)
    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
    scatter!(p1, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    p2 = to_heatmap(model.grid, sqrt.(σ²), title="σ", xlabel="σθ", ylabel="σω", c=:thermal)
    scatter!(p2, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    β = quantile(Normal(), problem.conf_threshold)

    p3 = to_heatmap(model.grid, μ .+ β .* sqrt.(σ²), title="Test Statistic", xlabel="σθ", ylabel="σω", c=:thermal)
    scatter!(p3, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    is_safe = (μ .+ β .* sqrt.(σ²)) .< problem.pfail_threshold
    colors = zeros(length(is_safe))
    FN_inds = findall(.!is_safe .& problem.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.5 : nothing
    FP_inds = finalize(is_safe .& .!problem.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.75 : nothing

    if sum(is_safe) > 0
        p4 = to_heatmap(model.grid, colors, 
            c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
            title="Estimated Safe Set", xlabel="σθ", ylabel="σω")
    else
        p4 = to_heatmap(model.grid, colors, 
            c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
            title="Estimated Safe Set", xlabel="σθ", ylabel="σω")
    end
    scatter!(p4, xs_eval, ys_eval,
        markersize=1.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false)

    return plot(p1, p2, p3, p4)
end

function plot_summary_gt(model::GaussianProcessModel, problem::GriddedProblem)
    return plot_summary_gt(model, problem, length(model.X))
end

# Ground truth
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
problem_gt = pendulum_problem(100, 100, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

# Set up the problem
nθ = 100
nω = 100
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

# Random acquisition
nsamps = 500
nsamps_tot = 50000
model_random = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps, ℓ=1e-2)
set_sizes_random = run_estimation!(model_random, problem, random_acquisition, nsamps_tot, log_every=1)

p = plot_eval_points(model_random)
p = plot_predictions(model_random)
p = plot_test_stats(model_random, problem.conf_threshold)

p = plot(collect(0:nsamps:nsamps_tot), set_sizes_random, label="random", legend=:topleft, linetype=:steppre)

# MILE acquisition
nsamps = 500
nsamps_tot = 50000
model_MILE = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps, ℓ=1e-2)
MILE_acquisition(model) = MILE_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
reset!(model_MILE)
set_sizes_MILE = run_estimation!(model_MILE, problem, MILE_acquisition, nsamps_tot, log_every=1)

plot!(p, collect(0:nsamps:nsamps_tot), set_sizes_MILE, label="MILE", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

p = plot_eval_points(model_MILE)
p = plot_predictions(model_MILE)
p = plot_test_stats(model_MILE, problem.conf_threshold)

# Animations
anim = @animate for iter in 1:length(model_MILE.X)
    plot_summary_gt(model_MILE, problem_gt, iter)
end
Plots.gif(anim, "figs/MILE_example_gt.gif", fps=10)

anim = @animate for iter in 1:length(model_random.X)
    plot_summary_gt(model_random, problem_gt, iter)
end
Plots.gif(anim, "figs/random_GP_example_gt.gif", fps=10)