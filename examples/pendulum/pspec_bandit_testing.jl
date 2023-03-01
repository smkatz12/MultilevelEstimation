using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/gaussian_process.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
include("../../src/learning_bandit.jl")
include("../../src/pspec_bandit.jl")
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

function pendulum_pspec_bandit_model(nθ, nω; σθ_max=0.2, σω_max=1.0,
    ℓmin=1e-4, ℓmax=1e-2)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return PSpecBanditModel(grid, ℓmin=ℓmin, ℓmax=ℓmax)
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
# Fix for bug when ground truth was generated
model_gt.β = model_gt.nsamps .+ model_gt.β .- 1
problem_gt = pendulum_problem(101, 101, σθ_max=0.2, σω_max=1.0, conf_threshold=0.95)
estimate_from_pfail!(problem_gt, model_gt)

# Smaller ground truth
# nθ = 26
# nω = 26
# σθ_max = 0.1
# σω_max = 0.5
# model_gt_small = pendulum_mc_model(nθ, nω, 10000; σθ_max=σθ_max, σω_max=σω_max)

# function fill_in_small(model_gt, model_gt_small)
#     for ind in 1:length(model_gt_small.grid)
#         x = ind2x(model_gt_small.grid, ind)
#         ind_orig = interpolants(model_gt.grid, x)[1][1]
#         model_gt_small.α[ind] = model_gt.α[ind_orig]
#         model_gt_small.β[ind] = model_gt.β[ind_orig]
#         model_gt_small.pfail[ind] = model_gt.pfail[ind_orig]
#     end
# end

# fill_in_small(model_gt, model_gt_small)
# problem_gt_small = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95, threshold=0.001)
# estimate_from_pfail!(problem_gt_small, model_gt_small)

# sum(problem_gt_small.is_safe) # True safe set size

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

sum(problem_gt_small.is_safe) # True safe set size

# Random acquisition
model_random = pendulum_pspec_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
reset!(model_random)
set_sizes_random = run_estimation!(model_random, problem_gt_small, random_acquisition, 1000,
    tuple_return=true)

model_kkb = pendulum_pspec_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓmax=1e-1)
reset!(model_kkb)
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem_gt_small.pfail_threshold,
    problem_gt_small.conf_threshold, rand_argmax=true, buffer=0.0)
set_sizes_kkb = run_estimation!(model_kkb, problem_gt_small, kernel_dkwucb_acquisition, 100000,
    tuple_return=true)

vals = [dot(model_kkb.ℓs, model_kkb.curr_pspecℓs[i, :]) for i = 1:length(model_kkb.grid)]
to_heatmap(model_kkb.grid, vals, xlabel="σθ", ylabel="σω")

vals = [dot(model_kkb.θs, model_kkb.θdists[i, :]) for i = 1:length(model_kkb.grid)]
to_heatmap(model_kkb.grid, vals, xlabel="σθ", ylabel="σω")
model_kkb.θdists

plot_eval_points(model_kkb, 500)

set_sizes_nk = [s[1] for s in set_sizes_kkb]
set_sizes_k = [s[2] for s in set_sizes_kkb]

iter = 100000
p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 100000), ylims=(0, 150))
plot!(p1, [0.0, 100000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")


# Baseline model
model_b = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
dkwucb_acquisition(model) = dkwucb_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_b = run_estimation!(model_b, problem, dkwucb_acquisition, 100000)

iter = 20000
p1 = plot(collect(0:iter), set_sizes_b[1:iter+1],
    label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
    label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, iter), ylims=(0, 120))
plot!(p1, [0.0, 100000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

# Summary plotting
function plot_pspec_summary(model, set_sizes_nk, set_sizes_b, set_sizes_k, problem_gt, iter;
    max_iter=20000)
    p1 = plot(collect(0:iter), set_sizes_b[1:iter+1],
        label="Baseline", color=:gray, lw=2)
    plot!(p1, collect(0:iter), set_sizes_nk[1:iter+1],
        label="No Kernel", color=:teal, linealpha=0.5, lw=2)
    plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
        label="Ours", legend=:right, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, max_iter), ylims=(0, 120))
    plot!(p1, [0.0, iter], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

    p2 = plot_eval_points(model, iter, xlabel="σθ")

    #### Get current counts ####
    eval_inds = model.eval_inds[1:iter]
    eval_res = model.eval_res[1:iter]

    αs = zeros(length(model.grid))
    βs = zeros(length(model.grid))
    for i = 1:length(model.grid)
        eval_inds_inds = findall(eval_inds .== i)
        neval = length(eval_inds_inds)
        if neval > 0
            αs[i] = 1 + sum(eval_res[eval_inds_inds])
            βs[i] = 2 + neval - αs[i]
        else
            αs[i] = 1
            βs[i] = 1
        end
    end
    ############################

    #### Get current estimates ####
    αₖ = reshape(1 .+ model.Ks * (αs .- 1), length(model.grid), length(model.ℓs))
    βₖ = reshape(1 .+ model.Ks * (βs .- 1), length(model.grid), length(model.ℓs))
    pspec_pℓs = pspec_pℓ(αs, βs, αₖ, βₖ)
    dists = Beta.(αₖ, βₖ) # N x nbins matrix of beta distributions
    nbins = size(dists, 2)
    θdists = zeros(size(dists))
    for (i, θ) in enumerate(model.θs)
        θdists[:, i] = sum(pdf.(dists, θ) .* pspec_pℓs, dims=2)
    end
    for i = 1:size(θdists, 1)
        θdists[i, :] = θdists[i, :] ./ sum(θdists[i, :])
    end
    ##############################

    vals = [dot(model.ℓs, pspec_pℓs[i, :]) for i = 1:length(model.grid)]
    p3 = to_heatmap(model.grid, vals, xlabel="σθ", ylabel="σω", title="E[ℓ]")

    vals = [dot(model.θs, θdists[i, :]) for i = 1:length(model.grid)]
    p4 = to_heatmap(model.grid, vals, xlabel="σθ", ylabel="σω", title="E[θ]")

    vals = [model.θs[findfirst(cumsum(θdists[i, :]) .> problem.conf_threshold)] for i = 1:length(model.grid)]
    p5 = to_heatmap(model.grid, vals, xlabel="σθ", ylabel="σω", title="Test Statistic")

    #### Safe set estimates ####
    N = length(model.grid)
    is_safe = falses(N)

    pfail_ind = findfirst(model.θs .>= problem_gt.pfail_threshold)
    for i = 1:length(model.grid)
        conf_ind = findfirst(cumsum(θdists[i, :]) .> problem.conf_threshold)
        if conf_ind ≤ pfail_ind
            is_safe[i] = true
        end
    end
    ############################

    colors = zeros(length(is_safe))
    FN_inds = findall(.!is_safe .& problem_gt.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.75 : nothing
    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.5 : nothing

    if sum(is_safe) > 0
        p6 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω", title="Safe Set Estimate")
    else
        p6 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω", title="Safe Set Estimate")
    end

    return plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(800, 800))
end

@time p = plot_pspec_summary(model_kkb, set_sizes_nk, set_sizes_b, set_sizes_k, problem_gt_small, 1000)

max_iter = 100000
anim = @animate for iter in 1:1000:max_iter
    println(iter)
    plot_pspec_summary(model_kkb, set_sizes_nk, set_sizes_b, set_sizes_k, problem_gt_small, iter;
                       max_iter=max_iter)
end
Plots.gif(anim, "figs/pspec_results_100000.gif", fps=5)

ℓs, Ks = get_Ks(model_kkb.grid, ℓmin=1e-3, ℓmax=1e-2, nbins=100)
model_kkb.ℓs = ℓs
model_kkb.Ks = cat(Ks..., dims=1)
model_kkb.αₖ = reshape(1 .+ model_kkb.Ks * (model_kkb.α .- 1), length(model_kkb.grid), length(model_kkb.ℓs))
model_kkb.βₖ = reshape(1 .+ model_kkb.Ks * (model_kkb.β .- 1), length(model_kkb.grid), length(model_kkb.ℓs))
update_kernel!(model_kkb)
vals = [dot(model_kkb.ℓs, model_kkb.curr_pspecℓs[i, :]) for i = 1:length(model_kkb.grid)]
to_heatmap(model_kkb.grid, vals, xlabel="σθ", ylabel="σω")

dists = Beta.(model_kkb.αₖ, model_kkb.βₖ) # N x nbins matrix of beta distributions
nbins = size(dists, 2)
for (i, θ) in enumerate(model_kkb.θs)
    model_kkb.θdists[:, i] = sum(pdf.(dists, θ) .* model_kkb.curr_pspecℓs, dims=2)
end
for i = 1:size(model_kkb.θdists, 1)
    model_kkb.θdists[i, :] = model_kkb.θdists[i, :] ./ sum(model_kkb.θdists[i, :])
end

vals = [dot(model_kkb.θs, model_kkb.θdists[i, :]) for i = 1:length(model_kkb.grid)]
to_heatmap(model_kkb.grid, vals, xlabel="σθ", ylabel="σω")
model_kkb.θdists

safe_set_size(model_kkb, problem_gt_small.pfail_threshold, problem_gt_small.conf_threshold)

plot_eval_points(model_kkb)

function data_replay(model, problem; update_every=50)
    N = length(model.grid)
    nbins = length(model.ℓs)
    α = ones(N)
    β = ones(N)
    αₖ = ones(N, nbins)
    βₖ = ones(N, nbins)
    θdists = ones(N, nbins) / nbins

    inds = []
    ss = []
    ssn = []
    fpr = []

    for i in ProgressBar(1:length(model.eval_inds))
        sample_ind = model.eval_inds[i]
        res = model.eval_res[i]

        nfail = sum(res)
        α[sample_ind] += nfail
        β[sample_ind] += 1 - nfail

        if nfail > 0
            αₖ = αₖ .+ reshape(model.Ks[:, sample_ind], length(model.grid), length(model.ℓs))
        else
            βₖ = βₖ .+ reshape(model.Ks[:, sample_ind], length(model.grid), length(model.ℓs))
        end

        if i % update_every == 1
            pspecℓs = pspec_pℓ(α, β, αₖ, βₖ)

            dists = Beta.(αₖ, βₖ) # N x nbins matrix of beta distributions
            nbins = size(dists, 2)
            for (i, θ) in enumerate(model.θs)
                θdists[:, i] = sum(pdf.(dists, θ) .* pspecℓs, dims=2)
            end
            for i = 1:size(model.θdists, 1)
                θdists[i, :] = θdists[i, :] ./ sum(θdists[i, :])
            end

            sz_nokernel = sum([cdf(Beta(αi, βi), problem.pfail_threshold) > problem.conf_threshold for (αi, βi) in zip(α, β)])

            is_safe = falses(N)

            pfail_ind = findfirst(model.θs .>= problem.pfail_threshold)
            # sz_kernel = 0
            for i = 1:length(model.grid)
                conf_ind = findfirst(cumsum(θdists[i, :]) .> problem.conf_threshold)
                if conf_ind ≤ pfail_ind
                    is_safe[i] = true
                end
            end
            sz_kernel = sum(is_safe)

            # print(problem.is_safe)
            FP_inds = findall(is_safe .& .!problem.is_safe)
            r = isnothing(FP_inds) ? 0.0 : length(FP_inds) / length(is_safe)

            push!(inds, i)
            push!(ss, sz_kernel)
            push!(ssn, sz_nokernel)
            push!(fpr, r)
        end
    end
    return inds, ssn, ss, fpr
end

inds, ssn, ss, fpr = data_replay(model_kkb, problem_gt_small, update_every=1000)

p1 = plot(inds, ssn,
    label="Kernel DKWUCB", legend=:bottomright, color=:gray, lw=2)
plot!(p1, inds, ss,
    label="Kernel Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
    xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 100000), ylims=(0, 150))
plot!(p1, [0.0, 100000.0], [108, 108], linestyle=:dash, lw=3, color=:black, label="True Size")

plot(inds, fpr)