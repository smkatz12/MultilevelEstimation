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
    p = scatter(xs_eval, ys_eval,
        markersize=2.0, markercolor=:green, markerstrokecolor=:green,
        xlabel="σθ", ylabel="σω", legend=false; kwargs...)

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
function plot_eval_points(model::BanditModel, iter; include_grid=true, kwargs...)
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

function plot_eval_points(model::BanditModel; include_grid=true, kwargs...)
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
    !isnothing(TN_inds) ? colors[TN_inds] .= 0.5 : nothing
    FP_inds = findall(is_safe .& .!problem_gt.is_safe)
    !isnothing(FP_inds) ? colors[FP_inds] .= 0.75 : nothing

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
    return plot_safe_set(model, problem_gt, length(model.eval_inds))
end

""" GP specific plots """
function plot_test_stats(model::GaussianProcessModel, conf_threshold, iter; kwargs...)
    β = quantile(Normal(), conf_threshold)

    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, model.X[1:iter], model.X_inds[1:iter], model.y[1:iter], all_X, all_inds, model.K)
    p1 = to_heatmap(model.grid, μ .+ β .* sqrt.(σ²), xlabel="σθ", ylabel="σω"; kwargs...)

    xs_eval = [ind2x(model.grid, i)[1] for i in model.X_inds[1:iter]]
    ys_eval = [ind2x(model.grid, i)[2] for i in model.X_inds[1:iter]]
    scatter!(p1, xs_eval, ys_eval,
        markersize=1.0, markercolor=:aqua, markerstrokecolor=:aqua,
        xlabel="σθ", ylabel="σω", legend=false)

    return p1
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

    p1 = plot(collect(range(0, step=nsamps_indiv, length=iter+1)), set_sizes_random[1:iter+1], 
              label="Random", legend=:topleft, linetype=:steppre, color=:gray, lw=2)
    plot!(p1, collect(range(0, step=nsamps_indiv, length=iter+1)), set_sizes_MILE[1:iter+1], 
          label="MILE", legend=:topleft, linetype=:steppre, color=:teal, lw=2,
          xlabel="Number of Episodes", ylabel="Safe Set Size")

    p3 = plot_test_stats(model_random, problem_gt.conf_threshold, iter, colorbar=false, title="Test Statistic Random")
    p4 = plot_safe_set(model_random, problem_gt, iter, title="Safe Set Random")

    p6 = plot_test_stats(model_MILE, problem_gt.conf_threshold, iter, colorbar=false, title="Test Statistic MILE")
    p7 = plot_safe_set(model_MILE, problem_gt, iter, title="Safe Set MILE")

    l = @layout [
        a{0.4w, 0.6h} [grid(2, 2)]
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
          xlabel="Number of Episodes", ylabel="Safe Set Size", ylims=(0, 700))

    p2 = plot_eval_points(model_random, iter, include_grid=false, title="Evaluations Random")
    p3 = plot_safe_set(model_random, problem_gt, iter, title="Safe Set Random")

    p4 = plot_eval_points(model_thompson, iter, include_grid=false, title="Evaluations Thompson")
    p5 = plot_safe_set(model_thompson, problem_gt, iter, title="Safe Set Thompson")

    l = @layout [
        a{0.4w, 0.6h} [grid(2, 2)]
    ]
    p = plot(p1, p2, p3, p4, p5, layout=l, size=(800, 600))

    return p
end