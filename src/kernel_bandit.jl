# Kernel Bandit Model
using GridInterpolations
using Distributions
using ProgressBars

get_K(X, X′, k) = [k(x, x′) for x in X, x′ in X′]

mutable struct KernelBanditModel
    grid::RectangleGrid # Grid to evaluate on
    α::Vector # Failure counts
    β::Vector # Success counts
    kernel::Function # Kernel function
    K::Matrix # Kernel matrix
    KernelBanditModel(grid, α, β, kernel) = new(grid, α, β, kernel, get_K(grid, grid, kernel))
end

function run_estimation!(model::KernelBanditModel, problem::GriddedProblem, acquisition, nsamps;
    log_every=Inf)
    set_sizes = [0]

    for i in ProgressBar(1:nsamps)
        sample_ind = acquisition(model)
        params = ind2x(model.grid, sample_ind)
        res = problem.sim(params, 1)
        nfail = sum(res)


        model.α[sample_ind] += nfail
        model.β[sample_ind] += 1 - nfail
        model.pfail[sample_ind] = model.α[sample_ind] / (model.α[sample_ind] + model.β[sample_ind])

        if (i % log_every) == 0
            estimate_from_counts!(problem, model)
            push!(set_sizes, sum(problem.is_safe))
        end
    end

    return set_sizes
end

function estimate_from_counts!(problem::GriddedProblem, model::KernelBanditModel)
    for i = 1:length(problem.grid)
        params = ind2x(problem.grid, i)
        α = interpolate(model.grid, model.α, params)
        β = interpolate(model.grid, model.β, params)
        problem.is_safe[i] = cdf(Beta(α, β), problem.pfail_threshold) > problem.conf_threshold
    end
end

function estimate_from_est_counts!(problem::GriddedProblem, model::KernelBanditModel)
    K = get_K(problem.grid, model.grid, model.kernel)
    α_est = K * model.α
    β_est = K * model.β

    for i = 1:length(problem.grid)
        problem.is_safe[i] = cdf(Beta(α_est[i], β_est[i]), problem.pfail_threshold) > problem.conf_t
    end
end