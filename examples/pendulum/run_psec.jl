using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save
using ColorSchemes, Colors, Measures
using StatsFuns
using Random

include("../../src/multilevel_estimation.jl")
include("../../src/pspec_bandit.jl")
include("controller.jl")
include("setup.jl")

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

nθ = 21
nω = 21
σθ_max = 0.2
σω_max = 1.0
ℓmin = 1e-4
ℓmax = 1e-1

problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

kernel_dkwucb_acquisition(model) = kernel_dkwucb_acquisition(model, problem.pfail_threshold,
                                    problem.conf_threshold, rand_argmax=true, buffer=0.0)

for i = 4:5
    println("Iteration: $(i)")
    Random.seed!(i)
    model_kkb = pendulum_pspec_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓmin=ℓmin, ℓmax=ℓmax)
    set_sizes_kkb = run_estimation!(model_kkb, problem, kernel_dkwucb_acquisition, 50000,
        tuple_return=true)
    @save "/scratch/smkatz/multilevelest/pendulum_results/run$(i).bson" model_kkb set_sizes_kkb
end