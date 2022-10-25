using POMDPs, POMDPGym, Crux, Distributions, BSON, GridInterpolations, LinearAlgebra

include("controller.jl")
include("../../src/multilevel_estimation.jl")

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