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

nsamps_tot = 20000

nsamps_indiv = [100, 500, 1000]
ℓs = [1e-4, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1]
models = Matrix{GaussianProcessModel}(undef, length(ℓs), length(nsamps_indiv))
set_sizes = Matrix{Vector}(undef, length(ℓs), length(nsamps_indiv))

MILE_acquisition(model) = MILE_acquisition(model, problem_gt_small.pfail_threshold, problem_gt_small.conf_threshold)

for (i, ℓ) in enumerate(ℓs)
    for (j, nsamps) in enumerate(nsamps_indiv)
        models[i, j] = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps, ℓ=ℓ)
        set_sizes_MILE = run_estimation!(models[i, j], problem_gt_small, MILE_acquisition, nsamps_tot)
        set_sizes[i, j] = set_sizes_MILE
    end
end

samps_ind = 1
p = plot(legend=:bottomright, xlims=(0, 20000), ylims=(0, 120))
for (i, ℓ) in enumerate(ℓs)
    plot!(p, collect(range(0, step=nsamps_indiv[samps_ind], length=201)), set_sizes[i, samps_ind],
        label="ℓ=$(ℓ)", linetype=:steppre, color=:gray, lw=2, alpha=i / 6)
end
p

samps_ind = 2
p = plot(legend=:bottomright, xlims=(0, 20000), ylims=(0, 130))
for (i, ℓ) in enumerate(ℓs)
    plot!(p, collect(range(0, step=nsamps_indiv[samps_ind], length=41)), set_sizes[i, samps_ind],
        label="ℓ=$(ℓ)", linetype=:steppre, color=:gray, lw=2, alpha=i / 6)
end
p

samps_ind = 3
p = plot(legend=:bottomright, xlims=(0, 20000), ylims=(0, 130))
for (i, ℓ) in enumerate(ℓs)
    plot!(p, collect(range(0, step=nsamps_indiv[samps_ind], length=21)), set_sizes[i, samps_ind],
        label="ℓ=$(ℓ)", linetype=:steppre, color=:gray, lw=2, alpha=i / 6)
end
p

set_sizes[6, 1][end]
set_sizes[6, 2][end]
set_sizes[6, 3][end]

function compute_fpr(model::GaussianProcessModel, problem_gt::GriddedProblem)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X, model.X_inds, model.y, all_X, all_inds, model.K)
    β = quantile(Normal(), problem_gt.conf_threshold)
    is_safe = (μ .+ β .* sqrt.(σ²)) .< problem_gt.pfail_threshold

    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    return isnothing(FP_inds) ? 0.0 : length(FP_inds) / length(is_safe)
end

compute_fpr(models[6, 3], problem_gt_small)

plot_GP_summary(models[3, 1], problem_gt_small, 200)

mod = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=10, ℓ=0.01)
set_sizes_MILE = run_estimation!(mod, problem_gt_small, MILE_acquisition, 4000)
plot_GP_summary(mod, problem_gt_small, 40)
mod.X
plot(set_sizes_MILE)

qs = collect(1:100)
function get_cdf(q)
    dist = Beta(1 + 1 * q, 1 + 9 * q)
    return cdf(dist, 0.1)
end

plot(qs, (q) -> get_cdf(q))

function sim(neps)
    vals = zeros(neps)
    α, β = 1, 1
    for i = 1:neps
        succ = rand() < 0.05
        succ ? α += 1 : β += 1
        dist = Beta(α, β)
        vals[i] = cdf(dist, 0.1)
    end
    return vals
end

vals = sim(10000)
p = plot(vals, legend=false)
plot!(p, [0.0, 10000], [0.95, 0.95])

cdf(Beta(9, 93), 0.1)

function get_val(nwin)
    return cdf(Beta(1 + nwin, 1 + 100 - nwin), 0.1)
end

p = plot(1:100, get_val)
plot!(p, [0.0, 100.0], [0.95, 0.95])

vals = get_val.(1:100)
findfirst(vals .< 0.95)
# At most 4 failures
p = plot(models[6, 1].y, legend=false)
plot!(p, [0, 200], [0.96, 0.96])

model_kkb = BSON.load("/scratch/smkatz/multilevelest/pspec_run_100000_bugfix.bson")[:model_kkb]

function get_counts(model, iter)
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
    return αs + βs
end

to_heatmap(model_kkb.grid, get_counts(model_kkb, 5000))