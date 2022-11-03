using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
include("../../src/gaussian_process.jl")
include("../../src/acquisition.jl")
include("controller.jl")
include("setup.jl")

sqe_kernel(r; ℓ=0.01) = exp(-(r' * r) / (2 * ℓ^2))

function pendulum_gp_model(nθ, nω; σθ_max=0.3, σω_max=0.3,
    ℓ = 0.1, nsamps=500, ν=0.01)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    # Set up the mean and kernel functions
    m(x) = zeros(length(x)) #0.5 * ones(length(x))
    k(x, x′) = sqe_kernel(norm(x - x′), ℓ=ℓ)

    return GaussianProcessModel(grid, nsamps, m, k, ν)
end

# Set up the problem
nθ = 101
nω = 101
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.95)

# Random acquisition
nsamps = 500
nsamps_tot = 10000
model_random = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps)
set_sizes_random = run_estimation!(model_random, problem, random_acquisition, nsamps_tot, log_every=1)

p = plot(collect(0:nsamps:nsamps_tot), set_sizes_random, label="random", legend=:topleft)

# MILE acquisition
nsamps = 500
nsamps_tot = 10000
model_MILE = pendulum_gp_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, nsamps=nsamps)
MILE_acquisition(model) = MILE_acquisition((model, problem.pfail_threshold, problem.conf_threshold))
set_sizes_MILE = run_estimation!(model_MILE, problem, MILE_acquisition, nsamps_tot, log_every=1)

# Timing analysis
@time predict(model_random)