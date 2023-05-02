using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/pspec_bandit.jl")
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

function daa_pspec_bandit_model(nx₀, ny₀, nf; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0,
    ℓmin=1e-3, ℓmax=1e-1, nbins=50)
    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid = RectangleGrid(x₀s, y₀s, fs)

    return PSpecBanditModel(grid, ℓmin=ℓmin, ℓmax=ℓmax, w=[4e-8, 1.0, 3.265e-5], nbins=nbins)
end

# Plot results
ss_kb = BSON.load("examples/daa/results/sb_run1.bson")[:ss_kb]
ss = [s[2] for s in ss_kb]
plot(collect(0:50000), ss, legend=false)

# Ground truth
model_gt = BSON.load("examples/daa/results/ground_truth.bson")[:model]

# Small ground truth
nx₀ = 25
ny₀ = 25
nf = 8

model_gt_small = daa_mc_model(nx₀, ny₀, nf, 10000)

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
problem_gt_small = daa_problem(nx₀, ny₀, nf, conf_threshold=0.95)
estimate_from_pfail!(problem_gt_small, model_gt_small)

# Actual safe set size
sum(problem_gt_small.is_safe)

# Try making a model
kb_model = daa_pspec_bandit_model(nx₀, ny₀, nf, ℓmin=1e-3, ℓmax=1e-1, nbins=50)

# Estimating ℓ from ground truth run
function fill_in_pspec(model_gt, model_pspec)
    for ind in 1:length(model_pspec.grid)
        x = ind2x(model_pspec.grid, ind)
        ind_orig = interpolants(model_gt.grid, x)[1][1]
        model_pspec.α[ind] = model_gt.α[ind_orig]
        model_pspec.β[ind] = model_gt.β[ind_orig]
    end
    model_pspec.αₖ = reshape(1 .+ model_pspec.Ks * (model_pspec.α .- 1), length(model_pspec.grid), length(model_pspec.ℓs))
    model_pspec.βₖ = reshape(1 .+ model_pspec.Ks * (model_pspec.β .- 1), length(model_pspec.grid), length(model_pspec.ℓs))
end

fill_in_pspec(model_gt, kb_model)
update_kernel!(kb_model)

function to_heatmap(grid::RectangleGrid, ind3, vals; kwargs...)
    vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]), length(grid.cutPoints[3]))
    return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat[:, :, ind3]'; kwargs...)
end

function plot_ℓ_means(model::PSpecBanditModel; kwargs...)
    vals = [dot(model.ℓs, model.curr_pspecℓs[i, :]) for i = 1:length(model.grid)]
    ps = []
    for i = 1:length(model.grid.cutPoints[3])
        p = to_heatmap(model.grid, i, vals,
            colorbar=false, title="hfov = $(model.grid.cutPoints[3][i])", #clim=(1e-3,1e-1),
            xlabel="x₀", ylabel="y₀"; kwargs...)
        push!(ps, p)
    end

    return plot(ps..., layout=(2, 4), size=(800, 450))
end

plot_ℓ_means(kb_model)

function plot_pℓ(model, pt)
    sps, ps = interpolants(model.grid, pt)
    ind = sps[argmax(ps)]

    p = bar(model.ℓs, model.curr_pspecℓs[ind, :], legend=false, color=:teal, lw=0.25, xlabel="ℓ",
        ylabel="P(ℓ ∣ D)", ylims=(0, maximum(model.curr_pspecℓs[ind, :]) + 0.02),
        xlims=(model.ℓs[1], model.ℓs[end]))
    return p
end

plot_pℓ(kb_model, [2800, 1.2, 80.0])
plot_pℓ(kb_model, [2700, 1.2, 80.0])
plot_pℓ(kb_model, [2700, 1.1, 80.0])
plot_pℓ(kb_model, [2500, 1.0, 80.0])
plot_pℓ(kb_model, [2550, 1.0, 80.0])
plot_pℓ(kb_model, [2000, 1.0, 80.0])

function plot_ℓ_means_lim(model::PSpecBanditModel; xmin=2000.0, ymin=1.0)
    vals = [dot(model.ℓs, model.curr_pspecℓs[i, :]) for i = 1:length(model.grid)]
    vals_mat = reshape(vals, length(model.grid.cutPoints[1]), length(model.grid.cutPoints[2]), length(model.grid.cutPoints[3]))
    xind = findfirst(x -> x >= xmin, model.grid.cutPoints[1])
    yind = findfirst(y -> y >= ymin, model.grid.cutPoints[2])
    ps = []
    for i = 1:length(model.grid.cutPoints[3])
        p = heatmap(model.grid.cutPoints[1][xind:end], model.grid.cutPoints[2][yind:end], vals_mat[xind:end, yind:end, i]',
            colorbar=false, title="hfov = $(model.grid.cutPoints[3][i])", #clim=(1e-3,1e-1),
            xlabel="x₀", ylabel="y₀")
        push!(ps, p)
    end

    return plot(ps..., layout=(2, 4), size=(800, 450))
end

p = plot_ℓ_means_lim(kb_model, xmin=2000.0, ymin=1.0)

pfail(model, params) = interpolate(model.grid, model.pfail, params)
plot(x -> pfail(model_gt, [x, 1.0, 80.0]), 1000, 3000)


nothing









# Try running it
kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem_gt_small.pfail_threshold,
    problem_gt_small.conf_threshold, rand_argmax=true, buffer=0.0)
reset!(kb_model)
ss_kb = run_estimation!(kb_model, problem_gt_small, kernel_dkwucb_acquisition, 10,
    tuple_return=true)

ss_k = [s[2] for s in ss_kb]

plot!(p, collect(0:5000), ss_k)

function to_heatmap(grid::RectangleGrid, ind3, vals; kwargs...)
    vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]), length(grid.cutPoints[3]))
    return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat[:, :, ind3]'; kwargs...)
end

function plot_ℓ_means(model::PSpecBanditModel; kwargs...)
    vals = [dot(model.ℓs, model.curr_pspecℓs[i, :]) for i = 1:length(model.grid)]
    ps = []
    for i = 1:length(model.grid.cutPoints[3])
        p = to_heatmap(model.grid, i, vals,
            colorbar=false, title="hfov = $(model.grid.cutPoints[3][i])", #clim=(1e-3,1e-1),
            xlabel="x₀", ylabel="y₀"; kwargs...)
        push!(ps, p)
    end

    return plot(ps..., layout=(2, 4), size=(800, 450))
end

plot_ℓ_means(kb_model)

function plot_pℓ(model, pt)
    sps, ps = interpolants(model.grid, pt)
    ind = sps[argmax(ps)]

    p = bar(model.ℓs, model.curr_pspecℓs[ind, :], legend=false, color=:teal, lw=0.25, xlabel="ℓ",
        ylabel="P(ℓ ∣ D)", ylims=(0, maximum(model.curr_pspecℓs[ind, :]) + 0.02),
        xlims=(model.ℓs[1], model.ℓs[end]))
    return p
end

plot_pℓ(kb_model, [2000, 1.0, 30.0])