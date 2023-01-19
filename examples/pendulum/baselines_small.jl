using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
include("controller.jl")
include("setup.jl")
include("pendulum_plotting.jl")

wsqe_kernel(r, W; ℓ=0.01) = exp(-(r' * W * r) / (2 * ℓ^2))

function pendulum_mc_model(nθ, nω, nsamps; σθ_max=0.3, σω_max=0.3)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return MonteCarloModel(grid, nsamps)
end

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

# Small ground truth
nθ = 21
nω = 21
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

model_gt_small = pendulum_mc_model(nθ, nω, 10000; σθ_max=σθ_max, σω_max=σω_max)

function fill_in_small(model_gt, model_gt_small)
    for ind in 1:length(model_gt_small.grid)
        x = ind2x(model_gt_small.grid, ind)
        ind_orig = interpolants(model_gt.grid, x)[1][1]
        model_gt_small.α[ind] = model_gt.α[ind_orig]
        model_gt_small.β[ind] = model_gt.β[ind_orig]
        model_gt_small.pfail[ind] = model_gt.pfail[ind_orig]
    end
end

fill_in_small(model_gt, model_gt_small)
problem_gt_small = pendulum_problem(nθ, nω, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt_small, model_gt_small)

# Actual safe set size
sum(problem_gt_small.is_safe)

nsamps_indiv = 100
nsamps_tot = 20000

ℓ = 2e-2

# GP random
model_random = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps_indiv, ℓ=ℓ)
set_sizes_random = run_estimation!(model_random, problem, random_acquisition, nsamps_tot)

# GP MILE
model_MILE = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps_indiv, ℓ=ℓ)
MILE_acquisition(model) = MILE_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
reset!(model_MILE)
set_sizes_MILE = run_estimation!(model_MILE, problem, MILE_acquisition, nsamps_tot)

iter = 200
p1 = plot(collect(range(0, step=nsamps_indiv, length=iter + 1)), set_sizes_random[1:iter+1],
    label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2,
    xlims=(0, 20000), ylims=(0, 108))
plot!(p1, collect(range(0, step=nsamps_indiv, length=iter + 1)), set_sizes_MILE[1:iter+1],
    label="MILE", legend=:topleft, linetype=:steppre, color=:magenta, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size")

# p2 = plot_test_stats(model_MILE, problem_gt.conf_threshold, 5, colorbar=false,
#     xlims=(0.0, 0.2), ylims=(0.0, 1.0), xticks=false, title="500 Episodes",
#     titlefontsize=12)
# p3 = plot_test_stats(model_MILE, problem_gt.conf_threshold, 100, colorbar=false,
#     xlims=(0.0, 0.2), ylims=(0.0, 1.0), xticks=false, title="10,000 Episodes",
#     titlefontsize=12)
# p4 = plot_test_stats(model_MILE, problem_gt.conf_threshold, 200, colorbar=false,
#     xlims=(0.0, 0.2), ylims=(0.0, 1.0), xticks=false, title="20,000 Episodes",
#     titlefontsize=12)

# p5 = plot_safe_set(model_MILE, problem_gt_small, 5)
# p6 = plot_safe_set(model_MILE, problem_gt_small, 50)
# p7 = plot_safe_set(model_MILE, problem_gt_small, 100)

# l = @layout [
#     a{0.3w} [grid(2, 3)]
# ]
# p = plot(p1, p2, p3, p4, p5, p6, p7, layout=l, size=(800, 300),
#     left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)

# plot_GP_compare(model_random, model_MILE, set_sizes_random, set_sizes_MILE,
#     nsamps_indiv, problem_gt_small, 200)

# anim = @animate for iter in 1:200
#     println(iter)
#     plot_GP_compare(model_random, model_MILE, set_sizes_random, set_sizes_MILE,
#         nsamps_indiv, problem_gt_small, iter)
# end
# Plots.gif(anim, "figs/MILE_output_small_2em2.gif", fps=30)

# Kernel Bandit Random
model_kbrandom = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=ℓ)
set_sizes_kbrandom = run_estimation!(model_kbrandom, problem, random_acquisition, nsamps_tot, tuple_return=true)

set_sizes_nk = [s[1] for s in set_sizes_kbrandom]
set_sizes_k = [s[2] for s in set_sizes_kbrandom]

g = create_kb_gif(model_kbrandom, problem_gt_small, set_sizes_nk, set_sizes_k, 
                  "random_short.gif", max_iter=5000, plt_every=25, fps=10)

# iter = 20000
# p1 = plot(collect(0:iter), set_sizes_nk,
#     label="Random", legend=:bottomright, color=:gray, lw=2)
# plot!(p1, collect(0:iter), set_sizes_k,
#     label="Kernel Random", legend=:bottomright, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 130))
# plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

plot_eval_points(model_kbrandom)
# plot_test_stats(model_kbrandom, problem_gt_small, use_kernel=false)
# plot_test_stats(model_kbrandom, problem_gt_small, use_kernel=true)

# iter = 200
# plot_counts(model_kbrandom, problem_gt_small, model_kbrandom.K, iter, use_kernel=false)
# plot_counts(model_kbrandom, problem_gt_small, model_kbrandom.K, iter, use_kernel=true)

# plot_total_counts(model_kbrandom, problem_gt_small, model_kbrandom.K, iter, use_kernel=false)
# plot_total_counts(model_kbrandom, problem_gt_small, model_kbrandom.K, iter, use_kernel=true)

# plot_test_stats(model_kbrandom, problem_gt_small, model_kbrandom.K, iter, use_kernel=false)
# plot_test_stats(model_kbrandom, problem_gt_small, model_kbrandom.K, iter, use_kernel=true)

# w = [1.0, 0.04]
# W = diagm(w ./ norm(w))
# heatmap(collect(range(-0.2, stop=σθ_max, length=100)), collect(range(-1.0, stop=σω_max, length=100)),
#     (x, y) -> wsqe_kernel([x, y], W, ℓ=2e-2))

# Kernel Bandit DKWUCB
nsamps_tot = 20000
model_kb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=2e-2)
dkwucb_acquisition(model) = dkwucb_acquisition(model, problem.pfail_threshold, problem.conf_threshold,
    rand_argmax=true, buffer=0.05)
set_sizes_kb = run_estimation!(model_kb, problem, dkwucb_acquisition, nsamps_tot, tuple_return=true)

set_sizes_nk = [s[1] for s in set_sizes_kb]
set_sizes_k = [s[2] for s in set_sizes_kb]

g = create_kb_gif(model_kb, problem_gt_small, set_sizes_nk, set_sizes_k,
    "dkwucb_nofilter_short.gif", max_iter=5000, plt_every=25, fps=10)

# iter = 20000
# p1 = plot(collect(0:iter), set_sizes_nk,
#     label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
# plot!(p1, collect(0:iter), set_sizes_k,
#     label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 130))
# plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

# plot_eval_points(model_kb)
# plot_test_stats(model_kbrandom, problem_gt_small, use_kernel=false)
# plot_test_stats(model_kbrandom, problem_gt_small, use_kernel=true)

# Kernel Bandit Kernel DKWUCB
model_kkb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=2e-2)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem.pfail_threshold,
    problem.conf_threshold, rand_argmax=true, buffer=0.05)
set_sizes_kkb = run_estimation!(model_kkb, problem, kernel_dkwucb_acquisition, nsamps_tot, tuple_return=true)

set_sizes_nk = [s[1] for s in set_sizes_kkb]
set_sizes_k = [s[2] for s in set_sizes_kkb]

g = create_kb_gif(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k,
    "kdkwucb_nofilter_short.gif", max_iter=5000, plt_every=25, fps=10)

# iter = 20000
# p1 = plot(collect(0:iter), set_sizes_nk,
#     label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
# plot!(p1, collect(0:iter), set_sizes_k,
#     label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 130))
# plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

# plot_eval_points(model_kkb)
# plot_test_stats(model_kkb, problem_gt_small, use_kernel=false)
# plot_test_stats(model_kkb, problem_gt_small, use_kernel=true)

# plot(model_kkb.eval_inds)

# # Summary plot
# iter = 20000
# p1 = plot(collect(0:iter), set_sizes_nk,
#     label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
# plot!(p1, collect(0:iter), set_sizes_k,
#     label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 150))
# plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size",
#       legend=false)

# p2 = plot_eval_points(model_kkb, include_grid=false, xlabel="σθ")

# p3 = plot_total_counts(model_kkb, problem_gt_small, model_kkb.K, iter, use_kernel=false, title="No Kernel")
# p4 = plot_total_counts(model_kkb, problem_gt_small, model_kkb.K, iter, use_kernel=true, title="With Kernel")

# p5 = plot_test_stats(model_kkb, problem_gt_small, model_kkb.K, iter, use_kernel=false)
# p6 = plot_test_stats(model_kkb, problem_gt_small, model_kkb.K, iter, use_kernel=true)

# p7 = plot_safe_set(model_kkb, problem_gt_small, iter, use_kernel=false, colorbar=true)
# p8 = plot_safe_set(model_kkb, problem_gt_small, iter, use_kernel=true, colorbar=true)

# # l = @layout [
# #     a{0.5w} b{0.5w}
# #     grid(2, 3)
# # ]
# # p = plot(p2, p1, p3, p5, p7, p4, p6, p8, layout=l, size=(1000, 800),
# #     left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)
# p = plot(p2, p1, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(600, 800),
#     left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)

# function plot_kb_summary(model::KernelBanditModel, problem::GriddedProblem, 
#                          set_sizes_nk, set_sizes_k, iter; max_iter=20000)
#     p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
#         label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
#     plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
#         label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
#         xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, max_iter), ylims=(0, 150))
#     plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size",
#         legend=false)

#     p2 = plot_eval_points(model, iter, include_grid=false, xlabel="σθ")

#     p3 = plot_total_counts(model, problem, model.K, iter, use_kernel=false, title="Counts")
#     p4 = plot_total_counts(model, problem, model.K, iter, use_kernel=true, title="With Kernel")

#     p5 = plot_test_stats(model, problem, model.K, iter, use_kernel=false, title="Test Statistic")
#     p6 = plot_test_stats(model, problem, model.K, iter, use_kernel=true, title="With Kernel")

#     p7 = plot_safe_set(model, problem, iter, use_kernel=false, colorbar=true, title="Safe Set")
#     p8 = plot_safe_set(model, problem, iter, use_kernel=true, colorbar=true, title="With Kernel")

#     p = plot(p2, p1, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(600, 800),
#         left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)

#     return p
# end

# iter = 1000
# p = plot_kb_summary(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, iter)

# function create_kb_gif(model::KernelBanditModel, problem::GriddedProblem, 
#                        set_sizes_nk, set_sizes_k, filename; max_iter=20000, plt_every=100, fps=30)
#     anim = @animate for iter in 1:plt_every:max_iter
#         println(iter)
#         plot_kb_summary(model, problem, set_sizes_nk, set_sizes_k, iter, max_iter=max_iter)
#     end
#     Plots.gif(anim, "figs/$filename", fps=fps)
# end

# g = create_kb_gif(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, 
#                   "kdkwucb_nofilter_short.gif", max_iter=5000, plt_every=25, fps=10)