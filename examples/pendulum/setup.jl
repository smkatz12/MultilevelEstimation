using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save

include("controller.jl")
include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")

# Problem
function pendulum_problem(nθ, nω; σθ_max=0.3, σω_max=0.3, threshold=0.1, eps_len=100)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid_points = Dict(:σθs => σθs, :σωs => σωs)
    grid = RectangleGrid(σθs, σωs)

    # Set up the rmdp
    env = InvertedPendulumMDP()
    π = FunPolicy(continuous_rule())

    # Create risk mdp which will allow simulating noisy perception
    tmax = eps_len * env.dt
    cost_fn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
    rmdp = RMDP(env, π, cost_fn=cost_fn, dt=env.dt, maxT=tmax, disturbance_type=:noise)

    # Set up simulation function
    function sim(params, nsamps)
        noise_distribution = MvNormal(zeros(2), I * params)
        noise_policy = DistributionPolicy(noise_distribution)

        D = episodes!(Sampler(rmdp, noise_policy), Neps=nsamps)
        num_failures = sum(D[:done])

        return num_failures
    end

    return GriddedProblem(grid_points, grid, sim, threshold, falses(length(grid)))
end

# Estimator
function pendulum_mc_model(nθ, nω, nsamps; σθ_max=0.3, σω_max=0.3)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid_points = Dict(:σθs => σθs, :σωs => σωs)
    grid = RectangleGrid(σθs, σωs)

    return MonteCarloModel(grid, nsamps, zeros(length(grid)))
end

nθ = 20
nω = 20
σθ_max = 0.25
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)

# nθ = 8
# nω = 8
nsamps = 100
model = pendulum_mc_model(nθ, nω, nsamps, σθ_max=σθ_max, σω_max=σω_max)

@time run_estimation!(model, problem)

@save "results/ground_truth.jl" problem model

# pfail(model, params) = interpolate(model.grid, model.pfail, params)
# heatmap(0:0.005:σθ_max, 0:0.005:σω_max, (x, y) -> pfail(model, [x, y]), xlabel="σθ", ylabel="σω")

# issafe(problem, params) = round(Int, interpolate(problem.grid, problem.is_safe, params))
# heatmap(0:0.005:σθ_max, 0:0.005:σω_max, (x, y) -> issafe(problem, [x, y]), xlabel="σθ", ylabel="σω")

# # Try to figure out how many samples are needed
# env = InvertedPendulumMDP()
# π = FunPolicy(continuous_rule())

# # Create risk mdp which will allow simulating noisy perception
# tmax = 100 * env.dt
# cost_fn(m, s, sp) = isterminal(m, sp) ? abs(s[2]) : 0
# rmdp = RMDP(env, π, cost_fn=cost_fn, dt=env.dt, maxT=tmax, disturbance_type=:noise)

# # Set up simulation function
# function sim(params, nsamps)
#     noise_distribution = MvNormal(zeros(2), I * params)
#     noise_policy = DistributionPolicy(noise_distribution)

#     D = episodes!(Sampler(rmdp, noise_policy), Neps=nsamps)
#     num_failures = sum(D[:done])

#     return num_failures
# end

# nsamps_test = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
# failures_test = [sim([0.125, 0.2], nsamps) for nsamps in nsamps_test]
# pfail_test = failures_test ./ nsamps_test
# plot(nsamps_test, pfail_test, legend=false, ylims=(0,1), xlabel="Number of Episodes", ylabel="Estimated Failure Probability")

# @time sim([0.125, 0.2], 10000)