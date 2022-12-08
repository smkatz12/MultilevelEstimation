using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
include("../../src/bandit.jl")
include("controller.jl")
include("setup.jl")
include("pendulum_plotting.jl")

wsqe_kernel(r, W; ℓ=0.01) = exp(-(r' * W * r) / (2 * ℓ^2))

function pendulum_gp_model(nθ, nω; σθ_max=0.2, σω_max=1.0,
    ℓ=5e-3, nsamps=500, w=[1.0, 0.04])
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    # Set up the mean and kernel functions
    m(x) = zeros(length(x)) #0.5 * ones(length(x))
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    # Solve for variance based on coefficient of variation
    cv = √((1 - 0.1) / (0.1 * nsamps))
    ν = (0.1 * cv)^2

    return GaussianProcessModel(grid, nsamps, m, k, ν)
end

function pendulum_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return BanditModel(grid)
end

# Ground truth
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
problem_gt = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

# General problem setup
nθ = 101
nω = 101
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

nsamps_indiv = 100
nsamps_tot = 50000

# # GP random
# model_random = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps_indiv, ℓ=1e-2)
# set_sizes_random = run_estimation!(model_random, problem, random_acquisition, nsamps_tot)

# # GP MILE
# model_MILE = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps_indiv, ℓ=1e-2)
# MILE_acquisition(model) = MILE_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
# reset!(model_MILE)
# set_sizes_MILE = run_estimation!(model_MILE, problem, MILE_acquisition, nsamps_tot)

# # Bandit random
# model_brandom = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
# set_sizes_brandom = run_estimation!(model_brandom, problem, random_acquisition, nsamps_tot)

# # Bandit thompson
# model_thompson = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
# thompson_acquisition(model) = thompson_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
# set_sizes_thompson = run_estimation!(model_thompson, problem, thompson_acquisition, nsamps_tot)

res = BSON.load("/scratch/smkatz/AA275_data.bson")
model_random = res[:model_random]
model_MILE = res[:model_MILE]
model_brandom = res[:model_brandom]
model_thompson = res[:model_thompson]

# Summary plots
plot_method_compare(model_MILE, model_thompson,
    set_sizes_MILE, set_sizes_thompson,
    problem_gt, 50000)

anim = @animate for iter in 0:100:50000
    println(iter)
    plot_method_compare(model_MILE, model_thompson, 
        set_sizes_MILE, set_sizes_thompson,
        problem_gt, iter)
end
Plots.gif(anim, "figs/compare_output.gif", fps=30)

# Bandit only plots
plot_eval_points(model_brandom, include_grid=false)
plot_eval_points(model_thompson, include_grid=false)

plot_safe_set(model_brandom, problem_gt, 5000)
plot_safe_set(model_thompson, problem_gt, 20000)

plot_bandit_compare(model_brandom, model_thompson, set_sizes_brandom, set_sizes_thompson,
    problem_gt, 50000)

anim = @animate for iter in 1:100:50000
    println(iter)
    plot_bandit_compare(model_brandom, model_thompson, set_sizes_brandom, set_sizes_thompson,
        problem_gt, iter)
end
Plots.gif(anim, "figs/bandit_output.gif", fps=30)

# GP only plots
p = plot(collect(0:nsamps_indiv:nsamps_tot), set_sizes_random, label="random", legend=:topleft, linetype=:steppre)
plot!(p, collect(0:nsamps_indiv:nsamps_tot), set_sizes_MILE, label="MILE", legend=:topleft, linetype=:steppre,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

plot_dist(model_random, 100)

plot_eval_points(model_random)
plot_eval_points(model_MILE)

plot_test_stats(model_random, problem_gt.conf_threshold)
plot_test_stats(model_MILE, problem_gt.conf_threshold)

plot_safe_set(model_random, problem_gt)
plot_safe_set(model_MILE, problem_gt)

plot_GP_compare(model_random, model_MILE, set_sizes_random, set_sizes_MILE,
    nsamps_indiv, problem_gt, 100)
savefig("figs/MILEvRandom100.png")

anim = @animate for iter in 1:500
    println(iter)
    plot_GP_compare(model_random, model_MILE, set_sizes_random, set_sizes_MILE,
        nsamps_indiv, problem_gt, iter)
end
Plots.gif(anim, "figs/MILE_output_100.gif", fps=30)