using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns
using SpecialFunctions

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

#################################################################################################3
function run_ℓest(nθ, nω, σθ_min, σθ_max, σω_min, σω_max; nsamps=2500, ℓconf=0.95)
    σθs = collect(range(σθ_min, stop=σθ_max, length=nθ))
    σωs = collect(range(σω_min, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    model = KernelBanditModel(grid, ℓconf=ℓconf)
    run_estimation!(model, problem, random_acquisition, nsamps,
        tuple_return=true, update_kernel_every=10)
    return model
end

# σθs = collect(range(0, stop=0.2, length=6))
# σωs = collect(range(0, stop=1.0, length=6))
σθs = collect(range(0, stop=0.2, length=4))
σωs = collect(range(0, stop=1.0, length=4))

nθ = 11
nω = 11
models = Matrix{KernelBanditModel}(undef, length(σθs) - 1, length(σωs) - 1)
for i = 1:length(σθs)-1
    for j = 1:length(σωs)-1
        models[i, j] = run_ℓest(nθ, nω, σθs[i], σθs[i+1], σωs[j], σωs[j+1])
    end
end

plot_ℓdist(models[3, 3], 2500)

ℓmeans = zeros(length(σθs) - 1, length(σωs) - 1)
for i = 1:length(σθs)-1
    for j = 1:length(σωs)-1
        pℓs = pℓ(models[i, j])
        ℓmeans[i, j] = sum(pℓs .* models[i, j].ℓs)
    end
end

ℓmeans
points1 = [(σθs[i] + σθs[i+1]) / 2 for i = 1:length(σθs)-1]
points2 = [(σωs[i] + σωs[i+1]) / 2 for i = 1:length(σωs)-1]
heatmap(points1, points2, ℓmeans, xlabel="σθ", ylabel="σω", title="Estimate of ℓ")

to_heatmap(models[1, 1].grid, models[1, 1].α ./ (models[1, 1].α .+ models[1, 1].β),
    xlabel="σθ", ylabel="σω", title="Estimate of pfail")
to_heatmap(models[2, 2].grid, models[2, 2].α ./ (models[2, 2].α .+ models[2, 2].β),
    xlabel="σθ", ylabel="σω", title="Estimate of pfail")

models[1,1].α[1]
models[1,1].β[1]

to_heatmap(models[1, 1].grid, models[1, 1].β,
    xlabel="σθ", ylabel="σω", title="Estimate of pfail")

plot(1:100, (x)->logp_αβ(1, x, 1, 40), legend=false, xlabel="β", ylabel="log P(1, β, 1, 40)")
heatmap(1:20, 1:20, (x, y) -> logp_αβ(x, y, 1, 20))

plot(1:500, (x) -> logp_αβ(100, 1, 1, x), legend=false, xlabel="β", ylabel="log P(1, 20, 1, β)")

function p_αβ_new(α, β, αₖ, βₖ; nθ=100)
    dist = Beta(αₖ, βₖ)
    # dist1 = Binomial(α + β - 2, θ)
    terms = [pdf(Binomial(α + β - 2, θ), α-1) * pdf(dist, θ) for θ in range(0.0, stop=1.0, length=nθ)]
    return (1 / nθ) * sum(terms)
end

plot(1:100, (x) -> log(p_αβ_new(1, 20, 1, x)), legend=false, xlabel="β", ylabel="log P(1, 20, 1, β)")

res = BSON.load("examples/pendulum/results/debugging_dip.bson")
model_kkb = res[:model_kkb]

plot_ℓdist(model_kkb, 16000)

###################################################################################
model = run_ℓest(nθ, nω, σθs[i], σθs[i+1], σωs[j], σωs[j+1])