using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

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

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0, w=[1.0, 0.04], ℓconf=0.95)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return KernelBanditModel(grid, ℓconf=ℓconf)
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

# Actual estimation
model_kbrandom = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_kbrandom)
set_sizes_kbrandom = run_estimation!(model_kbrandom, problem, random_acquisition, 20000,
    tuple_return=true, update_kernel_every=500)

plot(collect(1:20000), model_kbrandom.ℓests, xlabel="Number of Episodes", ylabel="ℓ",
    color=:magenta, lw=2, legend=false)

set_sizes_nk = [s[1] for s in set_sizes_kbrandom]
set_sizes_k = [s[2] for s in set_sizes_kbrandom]

iter = 20000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Random", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Random", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 150))
plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

model_kkb = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_kkb)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem.pfail_threshold,
    problem.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kkb = run_estimation!(model_kkb, problem, kernel_dkwucb_acquisition, 20000,
    tuple_return=true, update_kernel_every=10)

plot(collect(1:20000), model_kkb.ℓests, xlabel="Number of Episodes", ylabel="ℓ",
    color=:magenta, lw=2, legend=false)

set_sizes_nk = [s[1] for s in set_sizes_kkb]
set_sizes_k = [s[2] for s in set_sizes_kkb]

iter = 20000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 20000), ylims=(0, 150))
plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

function compute_fpr(model::KernelBanditModel, problem_gt::GriddedProblem, ℓ; w=[1.0, 0.04])
    # Compute K
    W = diagm(w ./ norm(w))
    k(x, x′) = wsqe_kernel(x - x′, W, ℓ=ℓ)

    X_pred = [X for X in model.grid]
    K = get_K(X_pred, X_pred, k)

    αₖ = 1 .+ K * (model.α .- 1)
    βₖ = 1 .+ K * (model.β .- 1)

    is_safe = [cdf(Beta(α, β), problem_gt.pfail_threshold) > problem.conf_threshold for (α, β) in zip(αₖ, βₖ)]

    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    return isnothing(FP_inds) ? 0.0 : length(FP_inds) / length(is_safe)
end

compute_fpr(model_kkb, problem_gt_small, model_kkb.curr_ℓ)

create_kb_learning_gif(model_kbrandom, problem_gt_small, set_sizes_nk, set_sizes_k, "random_learning.gif",
    max_iter=15000, plt_every=200, fps=10)

create_kb_learning_gif(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, "kkb_ldrop_learning.gif",
    max_iter=18000, plt_every=200, fps=10)

plot_ℓdist(model_kkb, 16000)
plot_kb_learning_summary(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, 16000)

# Testing different ℓ confidences
confs = [0.2, 0.5, 0.75, 0.8, 0.95, 0.99]
models = [pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓconf=conf) for conf in confs]
set_sizes = []

for model in models
    println(model.ℓconf)
    ss = run_estimation!(model, problem, kernel_dkwucb_acquisition, 20000,
        tuple_return=true, update_kernel_every=10)
    push!(set_sizes, ss)
end

ss_nk = [[s[1] for s in ss] for ss in set_sizes]
ss_k = [[s[2] for s in ss] for ss in set_sizes]

iter = 20000
p1 = plot(collect(0:iter), ss_k[1][1:iter+1],
    label="ℓconf= $(confs[1])", legend=:bottomright, xlabel="Number of Episodes", ylabel="Safe Set Size",
    color=:teal, lw=2, opacity=0.3)
plot!(p1, collect(0:iter), ss_k[2][1:iter+1], label="ℓconf= $(confs[2])",
      color=:teal, opacity=0.4, lw=2)
plot!(p1, collect(0:iter), ss_k[3][1:iter+1], label="ℓconf= $(confs[3])",
    color=:teal, opacity=0.5, lw=2)
plot!(p1, collect(0:iter), ss_k[4][1:iter+1], label="ℓconf= $(confs[4])",
    color=:teal, opacity=0.6, lw=2)
plot!(p1, collect(0:iter), ss_k[5][1:iter+1], label="ℓconf= $(confs[5])",
    color=:teal, opacity=0.7, lw=2)
plot!(p1, collect(0:iter), ss_k[6][1:iter+1], label="ℓconf= $(confs[6])",
    color=:teal, opacity=0.8, lw=2)

p2 = plot(collect(1:20000), models[1].ℓests,
    label="ℓconf= $(confs[1])", legend=:topright, xlabel="Number of Episodes", ylabel="ℓ Estimate",
    color=:magenta, lw=2, opacity=0.3)
plot!(p2, collect(1:20000), models[2].ℓests, label="ℓconf= $(confs[2])",
    color=:magenta, opacity=0.4, lw=2)
plot!(p2, collect(1:20000), models[3].ℓests, label="ℓconf= $(confs[3])",
    color=:magenta, opacity=0.5, lw=2)
plot!(p2, collect(1:20000), models[4].ℓests, label="ℓconf= $(confs[4])",
    color=:magenta, opacity=0.6, lw=2)
plot!(p2, collect(1:20000), models[5].ℓests, label="ℓconf= $(confs[5])",
    color=:magenta, opacity=0.7, lw=2)
plot!(p2, collect(1:20000), models[6].ℓests, label="ℓconf= $(confs[6])",
    color=:magenta, opacity=0.8, lw=2)

nothing

# function plot_logp(model::KernelBanditModel, iter, Kind; kwargs...)
#     eval_inds = model.eval_inds[1:iter]
#     eval_res = model.eval_res[1:iter]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     p = plot_logp(model, model.Ks[Kind], αs, βs; kwargs...)

#     return p
# end

# function plot_logp(model::KernelBanditModel, K, αs, βs; kwargs...)
#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     return to_heatmap(model.grid, p_D; kwargs...)
# end

# p = plot_logp(model_kkb, 19500, 10, title="19500")

# model_kkb.ℓs[10]

# anim = @animate for iter in 1:200:18000
#     println(iter)
#     plot_logp(model_kkb, iter, 10, title="$iter")
# end
# Plots.gif(anim, "figs/ltesting.gif", fps=5)

# p1 = plot(model_kkb.eval_inds, legend=false)
# p2 = plot(model_kkb.ℓests, legend=false)
# plot(p1, p2, layout=(2, 1))

# model_kkb.eval_inds[16020]
# ind2x(model_kkb.grid, 216)

# anim = @animate for iter in 1:100
#     println(iter)
#     plot_logp(model_kkb, 16000, iter, title="ℓ = $(model_kkb.ℓs[iter])")
# end
# Plots.gif(anim, "figs/ltesting_l.gif", fps=5)

# function get_logp(model::KernelBanditModel, iter, Kind, sind)
#     eval_inds = model.eval_inds[1:iter]
#     eval_res = model.eval_res[1:iter]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     αₖs = 1 .+ model.Ks[Kind] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[Kind] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D = logp_αβ(αs[sind], βs[sind], αₖs[sind], βₖs[sind])
#     return p_D
# end

# get_logp(model_kkb, 16000, 1, 216)
# plot(1:100, (x) -> get_logp(model_kkb, 13000, x, 216), legend=false)
# plot(1:100, (x) -> get_logp(model_kkb, 17900, x, 217), legend=false)

# scatter(findall(model_kkb.eval_inds[17000:end] .== 217) .+ 17000)

# anim = @animate for iter in 13500:100:19000
#     println(iter)
#     plot(1:100, (x) -> get_logp(model_kkb, iter, x, 217), legend=false, title="$iter")
# end
# Plots.gif(anim, "figs/temp.gif", fps=4)

# function plot_Δlogp(model::KernelBanditModel, iter1, iter2, Kind; kwargs...)
#     K = model.Ks[Kind]

#     # Compute for iter1
#     eval_inds = model.eval_inds[1:iter1]
#     eval_res = model.eval_res[1:iter1]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute for iter2
#     eval_inds = model.eval_inds[1:iter2]
#     eval_res = model.eval_res[1:iter2]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_2 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     return to_heatmap(model.grid, max.(p_D_2 .- p_D_1, -1.0); kwargs...)
# end

# plot_Δlogp(model_kkb, 13448, 16001, 10)
# plot_Δlogp(model_kkb, 13400, 13440, 10)

# function plot_normlogp(model::KernelBanditModel, iter, Kind; kwargs...)
#     eval_inds = model.eval_inds[1:iter]
#     eval_res = model.eval_res[1:iter]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     K = model.Ks[Kind]
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]
#     norm_p_D = p_D ./ sum(p_D)

#     return to_heatmap(model.grid, norm_p_D; kwargs...)
# end

# plot_normlogp(model_kkb, 16001, 90)

# function plot_normΔlogp(model::KernelBanditModel, iter1, iter2, Kind; kwargs...)
#     K = model.Ks[Kind]

#     # Compute for iter1
#     eval_inds = model.eval_inds[1:iter1]
#     eval_res = model.eval_res[1:iter1]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute for iter2
#     eval_inds = model.eval_inds[1:iter2]
#     eval_res = model.eval_res[1:iter2]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_2 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     Δlogp = p_D_2 .- p_D_1
#     Δlogp[216] = 0
#     norm_Δlogp = Δlogp ./ maximum(abs.(Δlogp))

#     return to_heatmap(model.grid, norm_Δlogp; kwargs...)
# end

# plot_normΔlogp(model_kkb, 13448, 16001, 90)

# anim = @animate for iter in 13500:100:19000
#     println(iter)
#     plot_normΔlogp(model_kkb, 13448, iter, 90, title="$iter")
# end
# Plots.gif(anim, "figs/norm_deltalogp_90.gif", fps=2)

# function plot_Δpcontrib(model::KernelBanditModel, iter1, iter2, Kind; kwargs...)
#     K = model.Ks[Kind]

#     # Compute for iter1
#     eval_inds = model.eval_inds[1:iter1]
#     eval_res = model.eval_res[1:iter1]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute for iter2
#     eval_inds = model.eval_inds[1:iter2]
#     eval_res = model.eval_res[1:iter2]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ K * (αs .- 1)
#     βₖs = 1 .+ K * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_2 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     pcontrib_1 = p_D_1 .- sum(p_D_1)
#     pcontrib_2 = p_D_2 .- sum(p_D_2)
#     Δpcontrib = pcontrib_1 - pcontrib_2

#     return to_heatmap(model.grid, pcontrib_1; kwargs...)
# end

# plot_Δpcontrib(model_kkb, 13448, 16001, 10)

# function plot_Δrelative(model::KernelBanditModel, iter1, iter2; kwargs...)
#     # Compute for iter1
#     eval_inds = model.eval_inds[1:iter1]
#     eval_res = model.eval_res[1:iter1]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ model.Ks[1] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[1] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1_1 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ model.Ks[end] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[end] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1_end = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute for iter2
#     eval_inds = model.eval_inds[1:iter2]
#     eval_res = model.eval_res[1:iter2]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ model.Ks[1] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[1] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_2_1 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ model.Ks[end] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[end] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_2_end = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     Δ1 = (p_D_1_end .- p_D_1_1)
#     Δ2 = (p_D_2_end .- p_D_2_1)

#     return to_heatmap(model.grid, Δ2 - Δ1; kwargs...)
# end

# plot_Δrelative(model_kkb, 13448, 16001)

# function plot_Δ(model::KernelBanditModel, iter1; kwargs...)
#     # Compute for iter1
#     eval_inds = model.eval_inds[1:iter1]
#     eval_res = model.eval_res[1:iter1]

#     αs = zeros(length(model.grid))
#     βs = zeros(length(model.grid))
#     for i = 1:length(model.grid)
#         eval_inds_inds = findall(eval_inds .== i)
#         neval = length(eval_inds_inds)
#         if neval > 0
#             αs[i] = 1 + sum(eval_res[eval_inds_inds])
#             βs[i] = 2 + neval - αs[i]
#         else
#             αs[i] = 1
#             βs[i] = 1
#         end
#     end

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ model.Ks[1] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[1] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1_1 = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     # Compute estimated pseudocounts
#     αₖs = 1 .+ model.Ks[end] * (αs .- 1)
#     βₖs = 1 .+ model.Ks[end] * (βs .- 1)

#     # Compute probability of sucess/failure
#     p_D_1_end = [logp_αβ(α, β, αₖ, βₖ) for (α, β, αₖ, βₖ) in zip(αs, βs, αₖs, βₖs)]

#     Δ1 = (p_D_1_end .- p_D_1_1)

#     return to_heatmap(model.grid, Δ1; kwargs...)
# end

# plot_Δ(model_kkb, 13448)
# plot_Δ(model_kkb, 16000)
# plot_Δ(model_kkb, 13000)

# model_kkb.eval_inds[16294]
# nothing 

# @save "examples/pendulum/results/debugging_dip.bson" model_kkb set_sizes_kkb
# # Messing with integral computation
# function p_αβ_exact(α, β, αₖ, βₖ)
#     n, m = α, α + β
#     nₖ, mₖ = αₖ - 1, αₖ + βₖ - 2
#     numerator = gamma(mₖ + 2) * gamma(nₖ + n + 1) * gamma(mₖ - nₖ + m - n + 1)
#     denominator = gamma(nₖ + 1) * gamma(mₖ - nₖ + 1) * gamma(mₖ + m + 2)
#     return numerator / denominator
# end

# function p_αβ2(α, β, αₖ, βₖ; nθ=100)
#     dist = Beta(αₖ, βₖ)
#     terms = [θ^α * (1 - θ)^β * pdf(dist, θ) for θ in collect(range(0.0, stop=1.0, length=nθ))[1:end-1]]
#     return (1 / nθ) * sum(terms)
# end

# function p_αβ3(α, β, αₖ, βₖ; nθ=100)
#     function my_beta(α, β, θ)
#         num = gamma(α + β)
#         denom = gamma(α) * gamma(β)
#         return (num / denom) * θ^(α-1) * (1 - θ)^(β - 1)
#     end
#     terms = [θ^α * (1 - θ)^β * my_beta(αₖ, βₖ, θ) for θ in collect(range(0.0, stop=1.0, length=nθ))[1:end-1]]
#     return (1 / nθ) * sum(terms)
# end

# α, β, αₖ, βₖ = 2, 2, 4, 4
# p_αβ(α, β, αₖ, βₖ, nθ=1000)
# p_αβ2(α, β, αₖ, βₖ, nθ=1000)
# p_αβ3(α, β, αₖ, βₖ, nθ=100000)
# test = p_αβ_exact(α, β, αₖ, βₖ)
# log(test)
# logp_αβ(α, β, αₖ, βₖ)

# nothing

# compute_fpr(model_kbrandom, problem_gt_small, model_kbrandom.curr_ℓ)
# compute_fpr(model_kkb, problem_gt_small, model_kkb.curr_ℓ)

# @time plot_ℓdist(model_kbrandom, 200, title="")

# iter=18000
# max_iter=20000
# p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
#     label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
# plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
#     label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
#     xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, max_iter), ylims=(0, 150))
# plot!(p1, [0.0, 20000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size",
#     legend=false, title="Safe Set Size")

# p2 = plot_eval_points(model_kkb, iter, include_grid=false, xlabel="σθ")

# curr_K = model_kkb.Ks[findfirst(model_kkb.ℓs .== model_kkb.ℓests[iter])]
# p3 = plot_total_counts(model_kkb, problem_gt_small, curr_K, iter, use_kernel=true, title="Total Counts")
# p4 = plot_test_stats(model_kkb, problem_gt_small, curr_K, iter, use_kernel=true, title="Test Statistic")
# p5 = plot_ℓdist(model_kkb, iter, title="Distribution over Kernel")
# p6 = plot_safe_set(model_kkb, problem_gt_small, iter, use_kernel=true, title="Safe Set Estimate")

# p = plot(p1, p3, p5, p2, p4, p6, layout=(2, 3), size=(900, 500), left_margin=3mm)



# plot_kb_learning_summary(model_kkb, problem_gt_small, set_sizes_nk, set_sizes_k, 1800)

# p_D = log_p(model_kbrandom, model_kbrandom.Ks[end], model_kbrandom.α, model_kbrandom.β)
# minimum(p_D)
# findfirst(isnan.(p_D))

# αₖs = 1 .+ model_kbrandom.Ks[end] * (model_kbrandom.α .- 1)
# βₖs = 1 .+ model_kbrandom.Ks[end] * (model_kbrandom.β .- 1)

# αₖs[45]
# βₖs[45]

# model_kbrandom.α[45], model_kbrandom.β[45], αₖs[45], βₖs[45]
# p_αβ_exact(model_kbrandom.α[45], model_kbrandom.β[45], αₖs[45], βₖs[45])