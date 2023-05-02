using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
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

function daa_gp_model(nx₀, ny₀, nf; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2,
    fmin=30.0, fmax=100.0, ℓ=1e-2, nsamps=500, w=[4e-8, 1.0, 3.265e-5])
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    # Set up the mean and kernel functions
    m(x) = zeros(length(x)) #0.5 * ones(length(x))
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    # Solve for variance based on coefficient of variation
    cv = √((1 - 0.3) / (0.3 * nsamps))
    ν = (0.3 * cv)^2

    return GaussianProcessModel(grid, nsamps, m, k, ν)
end

function get_rates(model, δ)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))

    μ, σ² = predict(model, model.X, model.X_inds, model.y, all_X, all_inds, model.K)

    β = quantile(Normal(), δ)
    test_stats = μ .+ β .* sqrt.(σ²)
    is_safe = test_stats .< 0.3

    FN_inds = findall(.!is_safe .& problem_gt_small.is_safe)
    FP_inds = findall(is_safe .& .!problem_gt_small.is_safe)

    return length(FN_inds), length(FP_inds)
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

nsamps_tot = 100000
nsamps_indiv = 100

MILE_acquisition(model) = MILE_acquisition(model, problem_gt_small.pfail_threshold, problem_gt_small.conf_threshold)

# Run trials
ss_gp1 = []
FNs_gp1 = zeros(5)
FPs_gp1 = zeros(5)
for i = 1:5
    model_gp1 = daa_gp_model(nx₀, ny₀, nf, ℓ=1e-1, nsamps=nsamps_indiv)
    set_sizes_gp1 = run_estimation!(model_gp1, problem_gt_small, MILE_acquisition, nsamps_tot)
    push!(ss_gp1, set_sizes_gp1)
    FNs_gp1[i], FPs_gp1[i] = get_rates(model_gp1, 0.95)
end

nsamps_tot = 5000000
nsamps_indiv = 5000

ss_gp2 = []
FNs_gp2 = zeros(5)
FPs_gp2 = zeros(5)
for i = 1:5
    model_gp2 = daa_gp_model(nx₀, ny₀, nf, ℓ=1e-1, nsamps=nsamps_indiv)
    set_sizes_gp2 = run_estimation!(model_gp2, problem_gt_small, MILE_acquisition, nsamps_tot)
    push!(ss_gp2, set_sizes_gp2)
    FNs_gp2[i], FPs_gp2[i] = get_rates(model_gp2, 0.95)
end

# # Random acquisition
# model_gp = daa_gp_model(nx₀, ny₀, nf, ℓ=5e-2, nsamps=nsamps_indiv)
# ss_random = run_estimation!(model_gp, problem_gt_small, random_acquisition, nsamps_tot)

# # MILE acquisition
# model_mile = daa_gp_model(nx₀, ny₀, nf, ℓ=1e-1, nsamps=nsamps_indiv)
# MILE_acquisition(model) = MILE_acquisition(model, problem_gt_small.pfail_threshold, problem_gt_small.conf_threshold)
# reset!(model_mile)
# ss_mile = run_estimation!(model_mile, problem_gt_small, MILE_acquisition, nsamps_tot)

# # Testing length parameters
# MILE_acquisition(model) = MILE_acquisition(model, problem_gt_small.pfail_threshold, problem_gt_small.conf_threshold)

# ℓs = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
# models = Matrix{GaussianProcessModel}(undef, length(ℓs), length(nsamps_indiv))
# set_sizes = Matrix{Vector}(undef, length(ℓs), length(nsamps_indiv))

# for (i, ℓ) in enumerate(ℓs)
#     for (j, nsamps) in enumerate(nsamps_indiv)
#         models[i, j] = daa_gp_model(nx₀, ny₀, nf, nsamps=nsamps_indiv, ℓ=ℓ)
#         set_sizes_MILE = run_estimation!(models[i, j], problem_gt_small, MILE_acquisition, nsamps_tot)
#         set_sizes[i, j] = set_sizes_MILE
#     end
# end

# res = BSON.load("examples/daa/results/gp_analysis_sz.bson")
# # models = res[:models]
# set_sizes = res[:set_sizes]

# p = plot(legend=:right)
# for (i, sz) in enumerate(set_sizes)
#     plot!(p, collect(0:100:50000), sz[1:501], label="ℓ=$(ℓs[i])")
# end
# p

# include("../../src/pspec_bandit.jl")
# ss_kb = BSON.load("examples/daa/results/sb_run1.bson")[:ss_kb]
# ss = [s[2] for s in ss_kb]

# plot!(p, collect(0:50000), ss, label="ours")

# # Best model
# # model = BSON.load("examples/daa/results/gp_analysis_best.bson")[:model]


# # nothing



# function to_heatmap(grid::RectangleGrid, ind3, vals; kwargs...)
#     vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]), length(grid.cutPoints[3]))
#     return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat[:, :, ind3]'; kwargs...)
# end

# # Some Plotting
# function plot_test_stats(model::GaussianProcessModel, conf_threshold, iter; kwargs...)
#     β = quantile(Normal(), conf_threshold)

#     all_X = [X for X in model.grid]
#     all_inds = collect(1:length(model.grid))
#     μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)

#     xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
#     ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
#     zs_eval = [ind2x(model.grid, i)[3] for i in model.X_inds[1:iter]]
#     ps = []

#     for i = 1:length(model.grid.cutPoints[3])
#         p = to_heatmap(model.grid, i, μ .+ β .* sqrt.(σ²),
#             colorbar=false, title="hfov = $(model.grid.cutPoints[3][i])", clim=(0, 2.0),
#             xlabel="x₀", ylabel="y₀"; kwargs...)
#         inds_slice = findall(x -> x == model.grid.cutPoints[3][i], zs_eval)
#         if !isnothing(inds_slice)
#             scatter!(p, xs_eval[inds_slice], ys_eval[inds_slice],
#                 markersize=1.0, markercolor=:aqua, markerstrokecolor=:aqua, legend=false)
#         end
#         push!(ps, p)
#     end

#     return plot(ps..., layout=(2, 4), size=(800, 450))
# end

# p = plot_test_stats(model_gp, problem_gt_small.conf_threshold, 500)
# p = plot_test_stats(model_mile, problem_gt_small.conf_threshold, 500)

# mycmap = ColorScheme([RGB{Float64}(0.5, 1.5 * 0.5, 2.0 * 0.5),
#     RGB{Float64}(0.25, 1.5 * 0.25, 2.0 * 0.25),
#     RGB{Float64}(227 / 255, 27 / 255, 59 / 255),
#     RGB{Float64}(0.0, 0.0, 0.0)])
# mycmap_small = ColorScheme([RGB{Float64}(0.25, 1.5 * 0.25, 2.0 * 0.25),
#     RGB{Float64}(0.0, 0.0, 0.0)])

# function plot_safe_set(model::GaussianProcessModel, problem_gt::GriddedProblem, iter; kwargs...)
#     all_X = [X for X in model.grid]
#     all_inds = collect(1:length(model.grid))
#     μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)
#     β = quantile(Normal(), problem_gt.conf_threshold)
#     is_safe = (μ .+ β .* sqrt.(σ²)) .< problem_gt.pfail_threshold
#     is_safe_slices = reshape(is_safe, length(model.grid.cutPoints[1]), length(model.grid.cutPoints[2]), length(model.grid.cutPoints[3]))

#     colors = zeros(length(is_safe))
#     FN_inds = findall(.!is_safe .& problem_gt.is_safe)
#     !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
#     TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
#     !isnothing(TN_inds) ? colors[TN_inds] .= 0.75 : nothing
#     FP_inds = findall(is_safe .& .!problem_gt.is_safe)
#     !isnothing(FP_inds) ? colors[FP_inds] .= 0.5 : nothing

#     ps = []

#     for i = 1:length(model.grid.cutPoints[3])
#         if sum(is_safe_slices[:, :, i]) > 0
#             p = to_heatmap(model.grid, i, colors,
#                 c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
#                 title="hfov = $(model.grid.cutPoints[3][i])",
#                 xlabel="x₀", ylabel="y₀"; kwargs...)
#         else
#             p = to_heatmap(model.grid, i, colors,
#                 c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
#                 title="hfov = $(model.grid.cutPoints[3][i])",
#                 xlabel="x₀", ylabel="y₀"; kwargs...)
#         end
#         push!(ps, p)
#     end

#     return plot(ps..., layout=(2, 4), size=(800, 450))
# end

# p = plot_safe_set(model_gp, problem_gt_small, 1000)
# p = plot_safe_set(model_mile, problem_gt_small, 1000)

# function plot_all_summary(model::GaussianProcessModel, problem_gt::GriddedProblem, iter; kwargs...)
#     p1 = plot_test_stats(model, problem_gt.conf_threshold, iter; kwargs...)
#     p2 = plot_safe_set(model, problem_gt, iter; kwargs...)
#     return plot(p1, p2, layout=(2, 1), size=(800, 900))
# end

# plot_all_summary(model_gp, problem_gt_small, 1000)
# plot_all_summary(model_mile, problem_gt_small, 1000)

# # anim = @animate for iter in 1:5:500
# #     println(iter)
# #     plot_all_summary(model_mile, problem_gt_small, iter)
# # end
# # Plots.gif(anim, "figs/daa_gp.gif", fps=5)

# p = plot(collect(0:1000:100000), ss_mile[1:101],
#     label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2)


# p = plot(collect(0:100:50000), ss_random,
#     label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2)
# plot!(p, collect(0:100:50000), ss_mile,
#     label="MILE", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), ylims=(0, 1100),
#     xticks=[10000, 30000, 50000])
# plot!(p, [0.0, 50000], [999, 999], linestyle=:dash, lw=3, color=:black, label="True Size")

# nothing

# # Plot kernel example
# w = [4e-8, 1.0, 3.265e-5]
# W = diagm(w ./ norm(w))
# k(x, x′) = wsqe_kernel(x - x′, W, ℓ=1e-2)

# function plot_kernel_slices(k, x₀=[2000.0, 1.0, 70.0])
#     hfovs = collect(30:10:100)
#     ps = [heatmap(problem.grid_points[:x₀s], problem.grid_points[:y₀s], (x, y) -> k([x, y, hfov], x₀),
#         xlabel="x₀", ylabel="y₀", clim=(0, 1), colorbar=false, title="hfov=$hfov") for hfov in hfovs]
#     return plot(ps..., layout=(2, 4), size=(800, 450))
# end

# k(x, x′) = wsqe_kernel(x - x′, W, ℓ=1e-1)
# plot_kernel_slices(k)