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
include("../../src/gittens_faster.jl")
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

    xs_eval = [GridInterpolations.ind2x(model.grid, i)[1] for i in unique(model.eval_inds)]
    ys_eval = [GridInterpolations.ind2x(model.grid, i)[2] for i in unique(model.eval_inds)]
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
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
problem_gt = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

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
gi = BSON.load("src/gittens_data/gibugfix_100pullL200_beta9999.bson")[:gi]
gi_acquisition(model) = gittens_acquisition(model, problem.pfail_threshold, problem.conf_threshold, gi,
    rand_argmax=true)
set_sizes_gi = run_estimation!(model_gi, problem, gi_acquisition, nsamps)

plot!(p, collect(0:nsamps), set_sizes_gi, label="Gittens", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

plot_eval_points(model_gi)
to_heatmap(model_gi.grid, model_gi.α .+ model_gi.β, c=:thermal)
plot(model_gi.eval_inds)

# Thompson Sampling Acquistion
nsamps = 50000
model_thompson = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
thompson_acquisition(model) = thompson_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_thompson = run_estimation!(model_thompson, problem, thompson_acquisition, nsamps)

plot!(p, collect(0:nsamps), set_sizes_thompson, label="Thompson", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

plot_eval_points(model_random)
plot_eval_points(model_thompson)

to_heatmap(model_random.grid, model_random.α .+ model_random.β, c=:thermal)
to_heatmap(model_thompson.grid, model_thompson.α .+ model_thompson.β, c=:thermal)

# DKWLCB
nsamps = 50000
model_dkwucb = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
dkwucb_acquisition(model) = dkwucb_acquisition(model, problem.pfail_threshold, problem.conf_threshold,
                                               δ=1.0, rand_argmax=true)
set_sizes_dkwucb = run_estimation!(model_dkwucb, problem, dkwucb_acquisition, nsamps)

plot_eval_points(model_dkwucb)
to_heatmap(model_dkwucb.grid, model_dkwucb.α .+ model_dkwucb.β, c=:thermal)

plot!(p, collect(0:nsamps), set_sizes_dkwucb, label="DKWUCB", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

function plot_summary_gt(model::BanditModel, problem::GriddedProblem, iter)
    eval_inds = model.eval_inds[1:iter]
    neval = [length(findall(eval_inds .== i)) for i = 1:length(model.grid)]
    p1 = to_heatmap(model.grid, neval, c=:thermal, clims=(0, 125),
        xlabel="σθ", ylabel="σω", title="Number of Episodes")

    # is_safe = falses(length(model.grid))
    # for i = 1:length(model.grid)
    #     params = ind2x(model.grid, i)
    #     α, β = predict_beta(model, params)
    #     is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    # end

    # eval_set = unique(eval_inds)
    # xs_eval = [ind2x(model.grid, i)[1] for i in eval_set]
    # ys_eval = [ind2x(model.grid, i)[2] for i in eval_set]

    # colors = zeros(length(is_safe))
    # FN_inds = findall(.!is_safe .& problem.is_safe)
    # !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    # TN_inds = findall(.!is_safe .& .!problem.is_safe)
    # !isnothing(TN_inds) ? colors[TN_inds] .= 0.5 : nothing
    # FP_inds = finalize(is_safe .& .!problem.is_safe)
    # !isnothing(FP_inds) ? colors[FP_inds] .= 0.75 : nothing

    # if sum(is_safe) > 0
    #     p2 = to_heatmap(model.grid, colors,
    #         c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
    #         title="Estimated Safe Set", xlabel="σθ", ylabel="σω")
    # else
    #     p2 = to_heatmap(model.grid, colors,
    #         c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
    #         title="Estimated Safe Set", xlabel="σθ", ylabel="σω")
    # end
    # scatter!(p2, xs_eval, ys_eval,
    #     markersize=1.0, markercolor=:green, markerstrokecolor=:green,
    #     xlabel="σθ", ylabel="σω", legend=false)

    return p1 #plot(p1, p2, size=(800, 400))
end

plot_summary_gt(model_gi, problem_gt, 100)

anim = @animate for iter in 1:500:50000
    plot_summary_gt(model_gi, problem_gt, iter)
end
Plots.gif(anim, "figs/gi_norand_example.gif", fps=10)

# Plotting Gittens Index
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

# Gittens Allocation Index Acquistion
# nsamps = 50000
# model_gi = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
# gi = BSON.load("src/gittens_data/gi_1000pulls_beta9999.bson")[:gi]
# gi_acquisition(model) = gittens_acquisition(model, problem.pfail_threshold, problem.conf_threshold, gi,
#     rand_argmax=true)
# set_sizes_gi = run_estimation!(model_gi, problem, gi_acquisition, nsamps)