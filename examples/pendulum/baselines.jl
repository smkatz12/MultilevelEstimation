using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
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

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0, 
    ℓ=5e-3, w=[1.0, 0.04])
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)
    return KernelBanditModel(grid, k)
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

# Kernel Bandit
model_kb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=1e-2)
dkwucb_acquisition(model) = dkwucb_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_kb = run_estimation!(model_kb, problem, dkwucb_acquisition, nsamps_tot, tuple_return=true)

# res = BSON.load("/scratch/smkatz/AA275_data.bson")
# model_random = res[:model_random]
# model_MILE = res[:model_MILE]
# model_brandom = res[:model_brandom]
# model_thompson = res[:model_thompson]

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

# AA275 plots
# GP Plot
# Recreate the set sizes
function get_GP_set_sizes(model::GaussianProcessModel)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))

    set_sizes = zeros(length(model.X_inds))
    for iter in ProgressBar(1:length(model.X_inds))
        μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)
        β = quantile(Normal(), problem.conf_threshold)
        is_safe = (μ .+ β .* sqrt.(σ²)) .< problem.pfail_threshold
        set_sizes[iter] = sum(is_safe)
    end

    return set_sizes
end

set_sizes_random = get_GP_set_sizes(model_random)
set_sizes_MILE = get_GP_set_sizes(model_MILE)

set_sizes_random = [0; set_sizes_random]
set_sizes_MILE = [0; set_sizes_MILE]

# iter = 500
# p1 = plot(collect(range(0, step=nsamps_indiv, length=iter + 1)), set_sizes_random[1:iter+1],
#     label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2,
#     xlims=(0, 50000), ylims=(0, 2200), xticks=[10000, 30000, 50000])
# plot!(p1, collect(range(0, step=nsamps_indiv, length=iter + 1)), set_sizes_MILE[1:iter+1],
#     label="MILE", legend=:topleft, linetype=:steppre, color=:magenta, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size")

# p2 = plot_test_stats(model_MILE, problem_gt.conf_threshold, 5, colorbar=false,
#     xlims=(0.0, 0.2), ylims=(0.0, 1.0), xticks=false, title="500 Episodes",
#     titlefontsize=12)
# p3 = plot_test_stats(model_MILE, problem_gt.conf_threshold, 150, colorbar=false,
#     xlims=(0.0, 0.2), ylims=(0.0, 1.0), xticks=false, title="15,000 Episodes",
#     titlefontsize=12)
# p4 = plot_test_stats(model_MILE, problem_gt.conf_threshold, 500, colorbar=false,
#     xlims=(0.0, 0.2), ylims=(0.0, 1.0), xticks=false, title="50,000 Episodes",
#     titlefontsize=12)

# p5 = plot_safe_set(model_MILE, problem_gt, 5)
# p6 = plot_safe_set(model_MILE, problem_gt, 150)
# p7 = plot_safe_set(model_MILE, problem_gt, 500)

# l = @layout [
#     a{0.3w} [grid(2, 3)]
# ]
# p = plot(p1, p2, p3, p4, p5, p6, p7, layout=l, size=(800, 300),
#     left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)
# savefig("figs/AA275_GP.pdf")

# Bandit Plot
function get_set_sizes_bandit(model::BanditModel)
    set_sizes = zeros(length(model.eval_inds))

    for iter in ProgressBar(1:length(model.eval_inds))
        eval_inds = model.eval_inds[1:iter]
        eval_res = model.eval_res[1:iter]

        is_safe = falses(length(model.grid))
        for i = 1:length(model.grid)
            eval_inds_inds = findall(eval_inds .== i)
            neval = length(eval_inds_inds)
            if neval > 0
                α = 1 + sum(eval_res[eval_inds_inds])
                β = 2 + neval - α
            else
                α = 1
                β = 1
            end
            is_safe[i] = cdf(Beta(α, β), problem_gt.pfail_threshold) > problem.conf_threshold
        end

        set_sizes[iter] = sum(is_safe)
    end

    return set_sizes
end

function get_set_sizes_bandit(model::BanditModel)
    set_sizes = zeros(length(model.eval_inds))
    curr_size = 0

    is_safe_vec = falses(length(model.grid))

    for iter in ProgressBar(1:length(model.eval_inds))
        eval_ind = model.eval_inds[iter]
        eval_res = model.eval_res[1:iter]
        if !is_safe_vec[eval_ind]
            eval_inds_inds = findall(model.eval_inds[1:iter] .== eval_ind)
            neval = length(eval_inds_inds)
            if neval > 0
                α = 1 + sum(eval_res[eval_inds_inds])
                β = 2 + neval - α
            else
                α = 1
                β = 1
            end
            is_safe = cdf(Beta(α, β), problem_gt.pfail_threshold) > problem.conf_threshold
            if is_safe
                curr_size += 1
                is_safe_vec[eval_ind] = true
            end
        end
        set_sizes[iter] = curr_size
    end

    return set_sizes
end

set_sizes_brandom = get_set_sizes_bandit(model_brandom)
set_sizes_thompson = get_set_sizes_bandit(model_thompson)

set_sizes_brandom = [0; set_sizes_brandom]
set_sizes_thompson = [0; set_sizes_thompson]

# iter = 50000
# p1 = plot(collect(0:iter), set_sizes_brandom,
#     label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=4)
# plot!(p1, collect(0:iter), set_sizes_thompson,
#     label="Thompson", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), ylims=(0, 700),
#     xticks=[10000, 30000, 50000])

# p2 = plot_eval_points(model_thompson, 500, include_grid=false, xticks=false,
#                       title="500 Episodes")
# p3 = plot_eval_points(model_thompson, 15000, include_grid=false, xticks=false,
#                       title="15,000 Episodes")
# p4 = plot_eval_points(model_thompson, 50000, include_grid=false, xticks=false,
#                       title="50,000 Episodes")

# p5 = plot_safe_set(model_thompson, problem_gt, 500)
# p6 = plot_safe_set(model_thompson, problem_gt, 15000)
# p7 = plot_safe_set(model_thompson, problem_gt, 50000)

# l = @layout [
#     a{0.3w} [grid(2, 3)]
# ]
# p = plot(p1, p2, p3, p4, p5, p6, p7, layout=l, size=(800, 300),
#     left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)
# # savefig("figs/AA275_bandit.pdf")

# # Comparison
# p1 = plot(collect(range(0, step=nsamps_indiv, length=501)), set_sizes_MILE[1:501],
#     label="LSE-GP", legend=:topleft, linetype=:steppre, color=:magenta, lw=2)
# plot!(p1, collect(0:50000), set_sizes_thompson[1:50001],
#     label="Threshold Bandit", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), size=(380, 380),
#     ylims=(0, 2200), right_margin=8mm)
# # savefig("figs/AA275_comparison.pdf")

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

plot_method_compare_dark(model_MILE, model_thompson,
    set_sizes_MILE, set_sizes_thompson,
    problem_gt, 50000)

anim = @animate for iter in 0:100:50000
    println(iter)
    plot_method_compare_dark(model_MILE, model_thompson,
        set_sizes_MILE, set_sizes_thompson,
        problem_gt, iter)
end
Plots.gif(anim, "figs/compare_output_dark.gif", fps=30)

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