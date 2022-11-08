# Acquisition functions for bandit algorithms
using Distributions

function random_acquisition(model::Union{BanditModel, KernelBanditModel, GaussianProcessModel})
    return rand(1:length(model.grid))
end

function expected_improvement(α, β, pfail_threshold)
    p_success = β / (α + β)
    new_prob_mass = cdf(Beta(α, β + 1), pfail_threshold)
    return p_success * new_prob_mass
end

function max_improvement_acquisition(model::BanditModel, pfail_threshold, conf_threshold)
    curr_best = 0.0
    curr_ind = 0

    for i = 1:length(model.grid)
        # Check if already safe
        safe = cdf(Beta(model.α[i], model.β[i]), pfail_threshold) > conf_threshold
        if !safe
            ei = expected_improvement(model.α[i], model.β[i], pfail_threshold)
            if ei > curr_best
                curr_best = ei
                curr_ind = i
            end
        end
    end

    return curr_ind
end

function kernel_max_improvement_acquisition(model::KernelBanditModel, pfail_threshold, conf_threshold)
    α_est = model.K * model.α
    β_est = model.K * model.β

    curr_best = 0.0
    curr_ind = 0

    for i = 1:length(model.grid)
        # Check if already safe
        safe = cdf(Beta(α_est[i], β_est[i]), pfail_threshold) > conf_threshold
        if !safe
            ei = expected_improvement(α_est[i], β_est[i], pfail_threshold)
            if ei > curr_best
                curr_best = ei
                curr_ind = i
            end
        end
    end

    return curr_ind
end

function MILE(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    β = quantile(Normal(), conf_threshold)

    neval = length(model.X)
    # println("neval: ", neval)
    npred = length(model.X_pred)

    max_ind_pred = 0
    max_pred = 0
    max_ind_eval = 0
    max_eval = 0
    
    if neval > 0
        if npred > 0
            objecs_pred = zeros(npred)
            μ, σ² = predict(model)
            for i = 1:npred
                x⁺ind = model.X_pred_inds[i]
                # x⁺ = model.X_pred[i]
                # println(i)
                # for j = 1:npred
                #     # println(j)
                #     #x = model.X_pred[j]
                #     # z = √(σ²[i] + model.ν) / model.K[i, j] * (pfail_threshold - μ[j] - β * √(σ²[j]))
                #     # objecs_pred[i] += cdf(Normal(), z)
                #     σGP⁺ = 
                #     zvec = (√(σ²[i] + model.ν) ./ model.K[:, i]) .* (pfail_threshold - μ)
                # end
                σ²GP⁺ = σ² .- (model.K[model.X_pred_inds, x⁺ind] .^ 2) ./ (σ²[i] + model.ν)
                # println(size(σ²GP⁺))
                zvec = (√(σ²[i] + model.ν) ./ model.K[model.X_pred_inds, x⁺ind]) .* (pfail_threshold .- μ .- β .* σ²GP⁺)
                # println(size(zvec))
                objecs_pred[i] = sum(cdf(Normal(), zvec))
                # println()
            end
        end
        max_ind_pred = argmax(objecs_pred)
        max_pred = maximum(objecs_pred)

        objecs_eval = zeros(neval)
        μ, σ² = predict(model, model.X, model.X_inds, model.K)
        for i = 1:neval
            # # x⁺ = model.X[i]
            # for j = 1:neval
            #     # x = model.X[j]
            #     z = √(σ²[i] + model.ν) / model.K[i, j] * (pfail_threshold - μ[j] - β * √(σ²[j]))
            #     objecs_eval[i] += cdf(Normal(), z)
            # end
            x⁺ind = model.X_inds[i]
            σ²GP⁺ = σ² .- (model.K[model.X_inds, x⁺ind] .^ 2) ./ (σ²[i] + model.ν)
            # println(size(σ²GP⁺))
            zvec = (√(σ²[i] + model.ν) ./ model.K[model.X_inds, x⁺ind]) .* (pfail_threshold .- μ .- β .* σ²GP⁺)
            # println(size(zvec))
            objecs_eval[i] = sum(cdf(Normal(), zvec))
            # println()
        end
        max_ind_eval = argmax(objecs_eval)
        max_eval = maximum(objecs_eval)

        if max_pred > max_eval
            return max_pred, model.X_pred_inds[max_ind_pred]
        else
            println("Picked an already evaluated point!")
            return max_eval, model.X_inds[max_ind_eval]
        end

    else
        return 0.0, rand(1:length(model.grid))
    end
end

function MILE2(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    β = quantile(Normal(), conf_threshold)

    neval = length(model.X)

    N = length(model.grid)
    all_X = [X for X in model.grid]
    all_inds = collect(1:length(model.grid))
    μ, σ² = predict(model, all_X, all_inds, model.K)

    # Debugging stuff (should be deleted later)
    zout = zeros(10)


    if neval > 0
        objecs = zeros(N)
        for i = 1:N
            σ²GP⁺ = σ² .- (model.K[:, i] .^ 2) ./ (σ²[i] + model.ν)
            zvec = (√(σ²[i] + model.ν) ./ model.K[:, i]) .* (pfail_threshold .- μ .- β .* σ²GP⁺)
            objecs[i] = sum(cdf(Normal(), zvec))
            if i == 4168
                zout = zvec
                println(sum(zvec))
                println(σ²[i])
                println(μ[i])
            end
        end

        max_ind = argmax(objecs)
        max_val = maximum(objecs)

        return max_val, max_ind, zout
    else
        return 0.0, rand(1:length(model.grid))
    end
end

function MILE3(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    β = quantile(Normal(), conf_threshold)

    neval = length(model.X)
    npred = length(model.X_pred)

    # Debugging stuff (should be deleted later)
    zout = zeros(10)

    if neval > 0
        objecs = zeros(npred)
        μ, σ² = predict(model)
        for i = 1:npred
            x⁺ind = model.X_pred_inds[i]
            σ²GP⁺ = σ² .- (model.K[model.X_pred_inds, x⁺ind] .^ 2) ./ (σ²[i] + model.ν)
            zvec = (√(σ²[i] + model.ν) ./ model.K[model.X_pred_inds, x⁺ind]) .* (pfail_threshold .- μ .- β .* σ²GP⁺)
            objecs[i] = sum(cdf(Normal(), zvec))
            if model.X_pred_inds[i] == 1593
                zout = σ²GP⁺ #zvec
                # println(sum(zvec))
                # println(σ²[i])
                # println(μ[i])
            end
        end

        max_ind = argmax(objecs)
        max_val = maximum(objecs)

        return max_val, model.X_pred_inds[max_ind], zout
    else
        return 0.0, rand(1:length(model.grid))
    end
end

function MILE4(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    β = quantile(Normal(), conf_threshold)

    neval = length(model.X)
    npred = length(model.X_pred)

    # Debugging stuff (should be deleted later)
    zout = zeros(10)

    if neval > 0
        objecs = zeros(npred)
        μ, S = predict_cov(model)
        σ² = diag(S)
        for i = 1:npred
            # inds⁺ = [model.X_inds; i]
            # tmp = model.K[model.X_pred_inds, inds⁺] / (model.K[inds⁺, inds⁺] + model.ν * I)
            # σ²GP⁺ = 1.0 .- dot.(eachrow(tmp), eachcol(model.K[inds⁺, model.X_pred_inds])) .+ eps()
            # x⁺ind = model.X_pred_inds[i]
            σ²GP⁺ = σ² .- (S[:, i] .^ 2) ./ (σ²[i] + model.ν)
            zvec = (√(σ²[i] + model.ν) ./ abs.(S[:, i])) .* (pfail_threshold .- μ .- β .* sqrt.(σ²GP⁺))
            objecs[i] = sum(cdf(Normal(), zvec))
            if model.X_pred_inds[i] == 8517
                zout = S[:, i] #cdf(Normal(), zvec) # σ²GP⁺ #zvec
                # println(sum(zvec))
                # println(σ²[i])
                # println(μ[i])
            end
        end

        max_ind = argmax(objecs)
        max_val = maximum(objecs)

        return max_val, model.X_pred_inds[max_ind] #, objecs #zout
    else
        return 0.0, rand(1:length(model.grid))
    end
end

function MILE_acquisition(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    _, ind = MILE4(model, pfail_threshold, conf_threshold)
    return ind #, objecs
end

function RMILE_acquisition(model::GaussianProcessModel, pfail_threshold, conf_threshold; γ=1e-5)
    new_size, ind = MILE(model, pfail_threshold, conf_threshold)
    old_size = safe_set_size(model, pfail_threshold, conf_threshold)

    _, σ²_pred = predict(model)
    max_pred = maximum(σ²_pred)
    _, σ²_eval = predict(model, model.X, model.X_inds, model.K)
    max_eval = maximum(σ²_eval)

    if γ * max(max_pred, max_eval) > new_size - old_size
        if max_pred > max_eval
            return model.X_pred_inds[argmax(σ²_pred)]
        else
            return model.X_inds[argmax(σ²_eval)]
        end
    else
        return ind
    end
end
