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
    """
    NOTE: this will not reevaluate an already selected point
    """
    β = quantile(Normal(), conf_threshold)

    neval = length(model.X)
    npred = length(model.X_pred)

    if neval > 0
        objecs = zeros(npred)
        μ, S = predict_cov(model)
        σ² = diag(S)
        for i = 1:npred
            σ²GP⁺ = σ² .- (S[:, i] .^ 2) ./ (σ²[i] + model.ν)
            zvec = (√(σ²[i] + model.ν) ./ abs.(S[:, i])) .* (pfail_threshold .- μ .- β .* sqrt.(σ²GP⁺))
            objecs[i] = sum(cdf(Normal(), zvec))
        end

        max_ind = argmax(objecs)
        max_val = maximum(objecs)

        return max_val, model.X_pred_inds[max_ind] #, objecs #zout
    else
        return 0.0, rand(1:length(model.grid))
    end
end

function MILE_acquisition(model::GaussianProcessModel, pfail_threshold, conf_threshold)
    """
    NOTE: this will not reevaluate an already selected point
    """
    _, ind = MILE(model, pfail_threshold, conf_threshold)
    return ind #, objecs
end

function RMILE_acquisition(model::GaussianProcessModel, pfail_threshold, conf_threshold; γ=1e-5)
    """
    NOTE: this will not reevaluate an already selected point
    """
    Δ_size, ind = MILE(model, pfail_threshold, conf_threshold)

    σ²_pred = predict(model)
    max_pred = maximum(σ²_pred)

    if γ * max_pred > Δ_size
        return model.X_pred_inds[argmax(σ²_pred)]
    else
        return ind
    end
end
