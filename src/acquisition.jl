# Acquisition functions for bandit algorithms
using Distributions

function random_acquisition(model::BanditModel)
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
        safe = cdf(Beta(α, β), pfail_threshold) > conf_threshold
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