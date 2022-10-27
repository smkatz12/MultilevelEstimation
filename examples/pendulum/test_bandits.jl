using POMDPs, POMDPGym, Crux, Distributions, Plots, BSON, GridInterpolations, LinearAlgebra
using BSON: @save

include("../../src/multilevel_estimation.jl")
include("../../src/montecarlo.jl")
include("../../src/bandit.jl")
include("../../src/kernel_bandit.jl")
include("../../src/acquisition.jl")
include("controller.jl")
include("setup.jl")

function pendulum_bandit_model(nθ, nω; σθ_max=0.3, σω_max=0.3)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    return BanditModel(grid, zeros(length(grid)), ones(length(grid)), ones(length(grid)))
end

sqe_kernel(r; ℓ=0.01) = exp(-(r' * r) / (2 * ℓ^2))

function pendulum_kernel_bandit_model(nθ, nω; σθ_max=0.3, σω_max=0.3, ℓ=0.01)
    # Set up grid
    σθs = collect(range(0, stop=σθ_max, length=nθ))
    σωs = collect(range(0, stop=σω_max, length=nω))
    grid = RectangleGrid(σθs, σωs)

    kernel(x, x′) = sqe_kernel(norm(x - x′), ℓ=ℓ)

    return KernelBanditModel(grid, ones(length(grid)), ones(length(grid)), kernel)
end

# Set up the problem
nθ = 101
nω = 101
σθ_max = 0.2
σω_max = 1.0
problem = pendulum_problem(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, conf_threshold=0.5)

# Random acquisition
nθ = 21
nω = 21
model_random = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
set_sizes_random = run_estimation!(model_random, problem, random_acquisition, 10000, log_every=100)

pfail(model, params) = interpolate(model.grid, model.pfail, params)
heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> pfail(model_random, [x, y]), xlabel="σθ", ylabel="σω")

# Max improvement acquisition
model_mi = pendulum_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max)
max_improvement_acquisition(model) = max_improvement_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_mi = run_estimation!(model_mi, problem, max_improvement_acquisition, 10000, log_every=100)

heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> pfail(model_mi, [x, y]), xlabel="σθ", ylabel="σω")

# Kernel max improvement acquisition
ℓ = 3e-2
model_kmi = pendulum_kernel_bandit_model(nθ, nω, σθ_max=σθ_max, σω_max=σω_max, ℓ=ℓ)
kernel_max_improvement_acquisition(model) = kernel_max_improvement_acquisition(model, problem.pfail_threshold, problem.conf_threshold)
set_sizes_kmi = run_estimation!(model_kmi, problem, kernel_max_improvement_acquisition, 10000, log_every=100)

points_model = [point for point in model_kmi.grid]
points_problem = [point for point in problem.grid]

p = scatter([point[1] for point in points_problem], [point[2] for point in points_problem])
scatter!(p, [point[1] for point in points_model], [point[2] for point in points_model])

# anim = @animate for ℓ in [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
#     K = get_K(model_kmi.grid, model_kmi.grid, (x, x′) -> sqe_kernel(norm(x - x′), ℓ=ℓ))
#     heatmap(K, title="ℓ = $ℓ")
# end
# Plots.gif(anim, "figs/ell_effect.gif", fps=1)

# test_model_kmi = pendulum_kernel_bandit_model(nθ, nω, σθ_max=0.2, σω_max=0.2, ℓ=ℓ)
# function get_heat(x, y, ℓ)
#     K = [sqe_kernel(norm([x, y] - point), ℓ=ℓ) for point in test_model_kmi.grid]
#     return sum(K)
# end
# heatmap(0.0:0.01:0.2, 0.0:0.01:0.2, (x, y) -> get_heat(x, y, 5e-3))

# K = [sqe_kernel(norm([0.0, 0.0] - point), ℓ=0.01) for point in model_kmi.grid]
# scatter(K)
# sum(K)

# heatmap(problem.grid_points[:σθs], problem.grid_points[:σωs], (x, y) -> get_heat(x, y, 0.01))

p = plot(collect(0:100:10000), set_sizes_random, xlabel="Number of Simulations", ylabel="Safe Set Size",
    label="random", legend=:bottomright)
plot!(p, collect(0:100:10000), set_sizes_mi, label="max improve")
plot!(p, collect(0:100:10000), set_sizes_kmi, label="kernel max improve")

# Actual safe set size
model_gt = BSON.load("examples/pendulum/results/ground_truth.bson")[:model]
estimate_from_counts!(problem, model_gt)
safe_set_size_gt = sum(problem.is_safe)