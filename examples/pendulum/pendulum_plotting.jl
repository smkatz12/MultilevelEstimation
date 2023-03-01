using Plots
default(fontfamily="Computer Modern", framestyle=:box)

mycmap = ColorScheme([RGB{Float64}(0.5, 1.5 * 0.5, 2.0 * 0.5),
    RGB{Float64}(0.25, 1.5 * 0.25, 2.0 * 0.25),
    RGB{Float64}(227 / 255, 27 / 255, 59 / 255),
    RGB{Float64}(0.0, 0.0, 0.0)])
mycmap_small = ColorScheme([RGB{Float64}(0.25, 1.5 * 0.25, 2.0 * 0.25),
    RGB{Float64}(0.0, 0.0, 0.0)])

function to_heatmap(grid::RectangleGrid, vals; kwargs...)
    vals_mat = reshape(vals, length(grid.cutPoints[1]), length(grid.cutPoints[2]))
    return heatmap(grid.cutPoints[1], grid.cutPoints[2], vals_mat'; kwargs...)
end


""" Plot where evaluated """
# GP
function plot_eval_points(model::GaussianProcessModel, iter; include_grid=true, kwargs...)
    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
    p = scatter(xs_eval, ys_eval, zcolor=30 * ones(length(xs_eval)),
        markersize=0.5, marker=:c, msw=0, c=:blues, clims=(1, 30),
        xlabel="σθ", ylabel="σω", legend=false,
        xlims=(0.0, 0.2), ylims=(0.0, 1.0); kwargs...)

    if include_grid
        xs = [pt[1] for pt in model.grid]
        ys = [pt[2] for pt in model.grid]
        scatter!(p, xs, ys, legend=false,
            markersize=0.5, markercolor=:black, markerstrokecolor=:black)
    end

    return p
end

function plot_eval_points(model::GaussianProcessModel; include_grid=true, kwargs...)
    return plot_eval_points(model, length(model.X_inds), include_grid=include_grid; kwargs...)
end

# Bandit
function plot_eval_points_size(model::Union{BanditModel,KernelBanditModel,LearningBanditModel, PSpecBanditModel}, iter; include_grid=true, kwargs...)
    eval_inds = model.eval_inds[1:iter]
    eval_set = unique(eval_inds)
    xs_eval = [ind2x(model.grid, i)[1] for i in eval_set]
    ys_eval = [ind2x(model.grid, i)[2] for i in eval_set]
    neval = [length(findall(eval_inds .== i)) for i in eval_set]
    szs = neval / 50.0

    p = scatter(xs_eval, ys_eval,
        markersize=szs, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false; kwargs...)

    if include_grid
        xs = [pt[1] for pt in model.grid]
        ys = [pt[2] for pt in model.grid]
        scatter!(p, xs, ys, legend=false,
            markersize=0.5, markercolor=:black, markerstrokecolor=:black)
    end
    return p
end

function plot_eval_points(model::Union{BanditModel,KernelBanditModel,LearningBanditModel, PSpecBanditModel}, iter; include_grid=true,
    θmax=0.2, ωmax=1.0, kwargs...)
    eval_inds = model.eval_inds[1:iter]
    eval_set = unique(eval_inds)
    xs_eval = [ind2x(model.grid, i)[1] for i in eval_set]
    ys_eval = [ind2x(model.grid, i)[2] for i in eval_set]
    neval = [length(findall(eval_inds .== i)) for i in eval_set]

    # p = scatter(xs_eval, ys_eval, zcolor=neval, c=:blues, clims=(1, 30),
    #     markersize=0.5, marker=:c, msw=0, #markerstrokecolor=:white,
    #     xlabel="σθ", ylabel="σω", legend=false,
    #     xlims=(0.0, 0.2), ylims=(0.0, 1.0); kwargs...)
    p = scatter(xs_eval, ys_eval, zcolor=neval, c=:thermal, clims=(1, 30),
        markersize=4.0, marker=:c, msw=0, #markerstrokecolor=:white,
        ylabel="σω", legend=false,
        xlims=(0.0, θmax), ylims=(0.0, ωmax); kwargs...)

    if include_grid
        xs = [pt[1] for pt in model.grid]
        ys = [pt[2] for pt in model.grid]
        scatter!(p, xs, ys, legend=false,
            markersize=0.5, markercolor=:black, markerstrokecolor=:black)
    end
    return p
end

function plot_eval_points(model::Union{BanditModel,KernelBanditModel,LearningBanditModel, PSpecBanditModel}; include_grid=true, kwargs...)
    return plot_eval_points(model, length(model.eval_inds), include_grid=include_grid; kwargs...)
end

""" Plot safeset estimate """
# GP
function plot_safe_set(model::GaussianProcessModel, problem_gt::GriddedProblem, iter; kwargs...)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)
    β = quantile(Normal(), problem.conf_threshold)
    is_safe = (μ .+ β .* sqrt.(σ²)) .< problem.pfail_threshold

    colors = zeros(length(is_safe))
    FN_inds = findall(.!is_safe .& problem_gt.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.75 : nothing
    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.5 : nothing

    if sum(is_safe) > 0
        p = to_heatmap(model.grid, colors,
            c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    else
        p = to_heatmap(model.grid, colors,
            c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    end

    return p
end

function plot_safe_set(model::GaussianProcessModel, problem_gt::GriddedProblem; kwargs...)
    return plot_safe_set(model, problem_gt, length(model.X_inds); kwargs...)
end

# Bandit
function plot_safe_set(model::BanditModel, problem_gt::GriddedProblem, iter; kwargs...)
    eval_inds = model.eval_inds[1:iter]
    eval_res = model.eval_res[1:iter]

    is_safe = falses(length(model.grid))
    for i = 1:length(model.grid)
        eval_inds_inds = findall(eval_inds .== i)
        neval = length(eval_inds_inds)
        if neval > 0
            α = 1 + sum(eval_res[eval_inds_inds])
            β = 2 + neval - α
        else
            α = 1
            β = 1
        end
        is_safe[i] = cdf(Beta(α, β), problem_gt.pfail_threshold) > problem.conf_threshold
    end

    colors = zeros(length(is_safe))
    FN_inds = findall(.!is_safe .& problem_gt.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.75 : nothing
    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.5 : nothing

    if sum(is_safe) > 0
        p2 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    else
        p2 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    end

    return p2
end

function plot_safe_set(model::BanditModel, problem_gt::GriddedProblem; kwargs...)
    return plot_safe_set(model, problem_gt, length(model.eval_inds); kwargs...)
end

# Kernel Bandit
function plot_safe_set(model::KernelBanditModel, problem_gt::GriddedProblem, K, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αs = 1 .+ K * (αs .- 1)
        βs = 1 .+ K * (βs .- 1)
    end

    is_safe = [cdf(Beta(α, β), problem_gt.pfail_threshold) > problem_gt.conf_threshold for (α, β) in zip(αs, βs)]

    colors = zeros(length(is_safe))
    FN_inds = findall(.!is_safe .& problem_gt.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.75 : nothing
    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.5 : nothing

    if sum(is_safe) > 0
        p2 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    else
        p2 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    end

    return p2
end

function plot_safe_set(model::KernelBanditModel, problem_gt::GriddedProblem; use_kernel=false, kwargs...)
    return plot_safe_set(model, problem_gt, model.K, length(model.eval_inds); use_kernel=use_kernel, kwargs...)
end

# Learning Bandit
function plot_safe_set(model::LearningBanditModel, problem_gt::GriddedProblem, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αₖ = reshape(1 .+ model.Ks * (αs .- 1), length(model.grid), length(model.ℓs))
        βₖ = reshape(1 .+ model.Ks * (βs .- 1), length(model.grid), length(model.ℓs))
        pℓs = pℓ(αs, βs, αₖ, βₖ)
        αs = αₖ * pℓs
        βs = βₖ * pℓs
    end

    is_safe = [cdf(Beta(α, β), problem_gt.pfail_threshold) > problem_gt.conf_threshold for (α, β) in zip(αs, βs)]

    colors = zeros(length(is_safe))
    FN_inds = findall(.!is_safe .& problem_gt.is_safe)
    !isnothing(FN_inds) ? colors[FN_inds] .= 0.25 : nothing
    TN_inds = findall(.!is_safe .& .!problem_gt.is_safe)
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.75 : nothing
    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.5 : nothing

    if sum(is_safe) > 0
        p2 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap, 4, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    else
        p2 = to_heatmap(model.grid, colors,
            c=cgrad(mycmap_small, 2, categorical=true), colorbar=:none,
            xlabel="σθ", ylabel="σω"; kwargs...)
    end

    return p2
end

function plot_safe_set(model::KernelBanditModel, problem_gt::GriddedProblem; use_kernel=false, kwargs...)
    return plot_safe_set(model, problem_gt, length(model.eval_inds); use_kernel=use_kernel, kwargs...)
end

""" GP specific plots """
function plot_test_stats(model::GaussianProcessModel, conf_threshold, iter; kwargs...)
    β = quantile(Normal(), conf_threshold)

    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)
    # p1 = to_heatmap(model.grid, μ .+ β .* sqrt.(σ²), xlabel="σθ", ylabel="σω"; kwargs...)
    p1 = to_heatmap(model.grid, μ .+ β .* sqrt.(σ²), ylabel="σω"; kwargs...)

    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
    scatter!(p1, xs_eval, ys_eval,
        markersize=1.0, markercolor=:aqua, markerstrokecolor=:aqua, ylabel="σω", legend=false)

    return p1
end

function plot_test_stats(model::GaussianProcessModel, conf_threshold; kwargs...)
    return plot_test_stats(model, conf_threshold, length(model.X_inds); kwargs...)
end

function plot_dist(model::GaussianProcessModel, iter; kwargs...)
    β = quantile(Normal(), conf_threshold)

    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)

    p1 = to_heatmap(model.grid, μ, xlabel="σθ", ylabel="σω", title="μ"; kwargs...)
    p2 = to_heatmap(model.grid, sqrt.(σ²), xlabel="σθ", ylabel="σω", title="σ"; kwargs...)

    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
    scatter!(p1, xs_eval, ys_eval,
        markersize=1.0, markercolor=:aqua, markerstrokecolor=:aqua,
        xlabel="σθ", ylabel="σω", legend=false)
    scatter!(p2, xs_eval, ys_eval,
        markersize=1.0, markercolor=:aqua, markerstrokecolor=:aqua,
        xlabel="σθ", ylabel="σω", legend=false)

    return plot(p1, p2)
end

function plot_test_stats(model::GaussianProcessModel, conf_threshold; kwargs...)
    return plot_test_stats(model, conf_threshold, length(model.X_inds); kwargs...)
end

function plot_GP_summary(model::GaussianProcessModel, problem_gt::GriddedProblem, iter)
    p1 = plot_eval_points(model, iter, title="Evaluated Points")
    p2 = plot_test_stats(model, problem_gt.conf_threshold, iter)
    p3 = plot_safe_set(model, problem_gt, iter)

    return plot(p1, p2, p3)
end

function plot_GP_compare(model_random::GaussianProcessModel, model_MILE::GaussianProcessModel,
    set_sizes_random, set_sizes_MILE, nsamps_indiv,
    problem_gt::GriddedProblem, iter)

    p1 = plot(collect(range(0, step=nsamps_indiv, length=iter + 1)), set_sizes_random[1:iter+1],
        label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2,
        #xlims=(0, 50000), ylims=(0, 2200))
        xlims=(0, 20000), ylims=(0, 108))
    plot!(p1, collect(range(0, step=nsamps_indiv, length=iter + 1)), set_sizes_MILE[1:iter+1],
        label="MILE", legend=:topleft, linetype=:steppre, color=:magenta, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size")

    p3 = plot_test_stats(model_random, problem_gt.conf_threshold, iter, colorbar=false, title="Test Statistic Random")
    p4 = plot_safe_set(model_random, problem_gt, iter, title="Safe Set Random")

    p6 = plot_test_stats(model_MILE, problem_gt.conf_threshold, iter, colorbar=false, title="Test Statistic MILE")
    p7 = plot_safe_set(model_MILE, problem_gt, iter, title="Safe Set MILE")

    l = @layout [
        a{0.4w,0.6h} [grid(2, 2)]
    ]
    p = plot(p1, p3, p4, p6, p7, layout=l, size=(800, 600))

    return p
end

""" Bandit Specific Plots """
function plot_bandit_compare(model_random::BanditModel, model_thompson::BanditModel,
    set_sizes_random, set_sizes_thompson,
    problem_gt::GriddedProblem, iter)

    p1 = plot(collect(0:iter), set_sizes_random[1:iter+1],
        label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2)
    plot!(p1, collect(0:iter), set_sizes_thompson[1:iter+1],
        label="Thompson", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), ylims=(0, 700))

    p2 = plot_eval_points(model_random, iter, include_grid=false, title="Evaluations Random")
    p3 = plot_safe_set(model_random, problem_gt, iter, title="Safe Set Random")

    p4 = plot_eval_points(model_thompson, iter, include_grid=false, title="Evaluations Thompson")
    p5 = plot_safe_set(model_thompson, problem_gt, iter, title="Safe Set Thompson")

    l = @layout [
        a{0.4w,0.6h} [grid(2, 2)]
    ]
    p = plot(p1, p2, p3, p4, p5, layout=l, size=(800, 600))

    return p
end

""" Kernel Bandit Specific Plots """
function plot_test_stats(model::KernelBanditModel, problem_gt::GriddedProblem, K, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αs = 1 .+ K * (αs .- 1)
        βs = 1 .+ K * (βs .- 1)
    end
    test_stats = [cdf(Beta(α, β), problem_gt.pfail_threshold) for (α, β) in zip(αs, βs)]

    p = to_heatmap(model.grid, test_stats, xlabel="σθ", ylabel="σω"; kwargs...)

    return p
end

function plot_test_stats(model::KernelBanditModel, problem_gt::GriddedProblem; use_kernel=false, kwargs...)
    return plot_test_stats(model, problem_gt, model.K, length(model.eval_inds); use_kernel=use_kernel, kwargs...)
end

function plot_counts(model::KernelBanditModel, problem_gt::GriddedProblem, K, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αs = 1 .+ K * (αs .- 1)
        βs = 1 .+ K * (βs .- 1)
    end

    p1 = to_heatmap(model.grid, αs, xlabel="σθ", ylabel="σω", title="α"; kwargs...)
    p2 = to_heatmap(model.grid, βs, xlabel="σθ", ylabel="σω", title="β"; kwargs...)

    return plot(p1, p2, size=(800, 400))
end

function plot_counts(model::KernelBanditModel, problem_gt::GriddedProblem; use_kernel=false, kwargs...)
    return plot_counts(model, problem_gt, model.K, length(model.eval_inds); use_kernel=use_kernel, kwargs...)
end

function plot_total_counts(model::KernelBanditModel, problem_gt::GriddedProblem, K, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αs = 1 .+ K * (αs .- 1)
        βs = 1 .+ K * (βs .- 1)
    end

    p1 = to_heatmap(model.grid, αs + βs, xlabel="σθ", ylabel="σω"; kwargs...)

    return p1
end

function plot_total_counts(model::KernelBanditModel, problem_gt::GriddedProblem; use_kernel=false, kwargs...)
    return plot_total_counts(model, problem_gt, model.K, length(model.eval_inds); use_kernel=use_kernel, kwargs...)
end

function plot_ℓdist(model::KernelBanditModel, iter; kwargs...)
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

    # curr_K = model.Ks[findfirst(model.ℓs .== model.ℓests[iter])]
    pℓs = pℓ(model, αs, βs)
    p = bar(model.ℓs, pℓs, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)",
        ylims=(0, 0.25), xlims=(0, model.ℓs[end]), title="Number of Episodes: $iter")
    dist = Categorical(pℓs)
    q = model.ℓs[quantile(dist, 1 - model.ℓconf)]
    plot!(p, [q, q], [0, 0.25], lw=2; kwargs...)

    return p
end

function plot_kb_summary(model::KernelBanditModel, problem::GriddedProblem,
    set_sizes_nk, set_sizes_k, iter; max_iter=20000, θmax=0.2, ωmax=1.0)
    true_size = sum(problem.is_safe)
    p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
        label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
    plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
        label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, max_iter), ylims=(0, true_size + 20))
    plot!(p1, [0.0, 20000.0], [true_size, true_size], linestyle=:dash, lw=3, color=:black, label="True Size",
        legend=false)

    p2 = plot_eval_points(model, iter, include_grid=false, θmax=0.2, ωmax=1.0, xlabel="σθ")

    p3 = plot_total_counts(model, problem, model.K, iter, use_kernel=false, title="Counts")
    p4 = plot_total_counts(model, problem, model.K, iter, use_kernel=true, title="With Kernel")

    p5 = plot_test_stats(model, problem, model.K, iter, use_kernel=false, title="Test Statistic")
    p6 = plot_test_stats(model, problem, model.K, iter, use_kernel=true, title="With Kernel")

    p7 = plot_safe_set(model, problem, model.K, iter, use_kernel=false, colorbar=true, title="Safe Set")
    p8 = plot_safe_set(model, problem, model.K, iter, use_kernel=true, colorbar=true, title="With Kernel")

    p = plot(p2, p1, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(600, 800),
        left_margin=3mm, bottom_margin=3.7mm, titlefontsize=10)

    return p
end

function create_kb_gif(model::KernelBanditModel, problem::GriddedProblem,
    set_sizes_nk, set_sizes_k, filename; max_iter=20000, plt_every=100, fps=30)
    anim = @animate for iter in 1:plt_every:max_iter
        println(iter)
        plot_kb_summary(model, problem, set_sizes_nk, set_sizes_k, iter, max_iter=max_iter)
    end
    Plots.gif(anim, "figs/$filename", fps=fps)
end

function plot_kb_learning_summary(model::KernelBanditModel, problem::GriddedProblem,
    set_sizes_nk, set_sizes_k, iter; max_iter=20000, θmax=0.2, ωmax=1.0)
    true_size = sum(problem.is_safe)
    p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
        label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
    plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
        label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, max_iter), ylims=(0, true_size + 20))
    plot!(p1, [0.0, max_iter], [true_size, true_size], linestyle=:dash, lw=3, color=:black, label="True Size",
        legend=false)

    p2 = plot_eval_points(model, iter, include_grid=false, xlabel="σθ", θmax=θmax, ωmax=ωmax)

    curr_K = model.Ks[findfirst(model.ℓs .== model.ℓests[iter])]
    p3 = plot_total_counts(model, problem, curr_K, iter, use_kernel=true, title="Total Counts")
    p4 = plot_test_stats(model, problem, curr_K, iter, use_kernel=true, title="Test Statistic")
    p5 = plot_ℓdist(model, iter, title="Distribution over Kernel")
    p6 = plot_safe_set(model, problem, curr_K, iter, use_kernel=true, title="Safe Set Estimate")

    p = plot(p1, p3, p5, p2, p4, p6, layout=(2, 3), size=(900, 500), left_margin=3mm)
    return p
end

function create_kb_learning_gif(model::KernelBanditModel, problem::GriddedProblem,
    set_sizes_nk, set_sizes_k, filename; max_iter=20000, plt_every=100, fps=30, θmax=0.2, ωmax=1.0)
    anim = @animate for iter in 1:plt_every:max_iter
        println(iter)
        plot_kb_learning_summary(model, problem, set_sizes_nk, set_sizes_k, iter, max_iter=max_iter,
            θmax=θmax, ωmax=ωmax)
    end
    Plots.gif(anim, "figs/$filename", fps=fps)
end

""" Learning Bandit Specific Plots """
function plot_total_counts(model::LearningBanditModel, problem_gt::GriddedProblem, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αₖ = reshape(1 .+ model.Ks * (αs .- 1), length(model.grid), length(model.ℓs))
        βₖ = reshape(1 .+ model.Ks * (βs .- 1), length(model.grid), length(model.ℓs))
        pℓs = pℓ(αs, βs, αₖ, βₖ)
        αs = αₖ * pℓs
        βs = βₖ * pℓs
    end

    p1 = to_heatmap(model.grid, αs + βs, xlabel="σθ", ylabel="σω"; kwargs...)

    return p1
end

function plot_test_stats(model::LearningBanditModel, problem_gt::GriddedProblem, iter; use_kernel=false, kwargs...)
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
    if use_kernel
        αₖ = reshape(1 .+ model.Ks * (αs .- 1), length(model.grid), length(model.ℓs))
        βₖ = reshape(1 .+ model.Ks * (βs .- 1), length(model.grid), length(model.ℓs))
        pℓs = pℓ(αs, βs, αₖ, βₖ)
        αs = αₖ * pℓs
        βs = βₖ * pℓs
    end
    test_stats = [cdf(Beta(α, β), problem_gt.pfail_threshold) for (α, β) in zip(αs, βs)]

    p = to_heatmap(model.grid, test_stats, xlabel="σθ", ylabel="σω"; kwargs...)

    return p
end

function plot_ℓdist(model::LearningBanditModel, iter; kwargs...)
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

    αₖ = reshape(1 .+ model.Ks * (αs .- 1), length(model.grid), length(model.ℓs))
    βₖ = reshape(1 .+ model.Ks * (βs .- 1), length(model.grid), length(model.ℓs))
    pℓs = pℓ(αs, βs, αₖ, βₖ)
    p = bar(model.ℓs, pℓs, legend=false, color=:teal, lw=0.25, xlabel="ℓ", ylabel="P(ℓ ∣ D)",
        ylims=(0, 0.3), xlims=(0, model.ℓs[end]), title="Number of Episodes: $iter")
    return p
end

function plot_learning_summary(model::LearningBanditModel, problem::GriddedProblem,
    set_sizes_nk, set_sizes_k, iter; max_iter=20000, θmax=0.2, ωmax=1.0)
    true_size = sum(problem.is_safe)
    p1 = plot(collect(0:iter), set_sizes_nk[1:iter+1],
        label="DKWUCB", legend=:bottomright, color=:gray, lw=2)
    plot!(p1, collect(0:iter), set_sizes_k[1:iter+1],
        label="Kernel DKWUCB", legend=:bottomright, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, max_iter), ylims=(0, true_size + 20))
    plot!(p1, [0.0, max_iter], [true_size, true_size], linestyle=:dash, lw=3, color=:black, label="True Size",
        legend=false)

    p2 = plot_eval_points(model, iter, include_grid=false, xlabel="σθ", θmax=θmax, ωmax=ωmax)

    p3 = plot_total_counts(model, problem, iter, use_kernel=true, title="Total Counts")
    p4 = plot_test_stats(model, problem, iter, use_kernel=true, title="Test Statistic")
    p5 = plot_ℓdist(model, iter, title="Distribution over Kernel")
    p6 = plot_safe_set(model, problem, iter, use_kernel=true, title="Safe Set Estimate")

    p = plot(p1, p3, p5, p2, p4, p6, layout=(2, 3), size=(900, 500), left_margin=3mm)
    return p
end

function create_learning_gif(model::LearningBanditModel, problem::GriddedProblem,
    set_sizes_nk, set_sizes_k, filename; max_iter=20000, plt_every=100, fps=30, θmax=0.2, ωmax=1.0)
    anim = @animate for iter in 1:plt_every:max_iter
        println(iter)
        plot_learning_summary(model, problem, set_sizes_nk, set_sizes_k, iter, max_iter=max_iter,
            θmax=θmax, ωmax=ωmax)
    end
    Plots.gif(anim, "figs/$filename", fps=fps)
end

""" GP Bandit Comparison """
function plot_method_compare(model_gp::GaussianProcessModel, model_bandit::BanditModel,
    set_sizes_gp, set_sizes_bandit,
    problem_gt::GriddedProblem, iter)

    iter = iter == 0 ? 1 : iter
    gp_iter = iter > 1 ? convert(Int64, iter / 100) : 1

    size_max = maximum(set_sizes_gp)
    p1 = plot(collect(range(0, step=nsamps_indiv, length=gp_iter + 1)), set_sizes_gp[1:gp_iter+1],
        label="GP", legend=:topleft, linetype=:steppre, color=:magenta, lw=2)
    plot!(p1, collect(0:iter), set_sizes_bandit[1:iter+1],
        label="Bandit", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), ylims=(0, size_max))

    p2 = plot_eval_points(model_gp, gp_iter, include_grid=false, title="Evaluations GP")
    p3 = plot_safe_set(model_gp, problem_gt, gp_iter, title="Safe Set GP")

    p4 = plot_eval_points(model_bandit, iter, include_grid=false, title="Evaluations Bandit")
    p5 = plot_safe_set(model_bandit, problem_gt, iter, title="Safe Set Bandit")

    l = @layout [
        a{0.4w,0.6h} [grid(2, 2)]
    ]
    p = plot(p1, p2, p3, p4, p5, layout=l, size=(800, 600))

    return p
end

function plot_method_compare_dark(model_gp::GaussianProcessModel, model_bandit::BanditModel,
    set_sizes_gp, set_sizes_bandit,
    problem_gt::GriddedProblem, iter)

    iter = iter == 0 ? 1 : iter
    gp_iter = iter > 1 ? convert(Int64, iter / 100) : 1

    size_max = maximum(set_sizes_gp)
    p1 = plot(collect(range(0, step=nsamps_indiv, length=gp_iter + 1)), set_sizes_gp[1:gp_iter+1],
        label="GP", legend=:topleft, linetype=:steppre, color=:magenta, lw=2,
        foreground_color_subplot="white",)
    plot!(p1, collect(0:iter), set_sizes_bandit[1:iter+1],
        label="Bandit", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
        xlabel="Number of Episodes", ylabel="Safe Set Size", xlims=(0, 50000), ylims=(0, size_max))

    p2 = plot_eval_points(model_gp, gp_iter, include_grid=false, title="Evaluations GP",
        foreground_color_subplot="white", c=:thermal)
    p3 = plot_safe_set(model_gp, problem_gt, gp_iter, title="Safe Set GP",
        foreground_color_subplot="white")

    p4 = plot_eval_points(model_bandit, iter, include_grid=false, title="Evaluations Bandit",
        foreground_color_subplot="white", c=:thermal)
    p5 = plot_safe_set(model_bandit, problem_gt, iter, title="Safe Set Bandit",
        foreground_color_subplot="white")

    l = @layout [
        a{0.4w,0.6h} [grid(2, 2)]
    ]
    p = plot(p1, p2, p3, p4, p5, layout=l, size=(800, 600), background_color=:black)

    return p
end