# Uses the restart formulation (Katehakis and Derman)
# For approximation, uses Whittle's approximation
# Reference: https://www.jhelumch.com/wp-content/uploads/2018/01/Gittins_indices_survey2013.pdf

using ProgressBars
using Plots

mutable struct GittensIndex
    npulls::Int64
    β::Float64
    L::Int64
    v::Vector
    m::Vector
    p_success::Vector
    p_failure::Vector
    sp_success::Vector
    sp_failure::Vector
    not_terminal::Vector
    function GittensIndex(npulls, β, L)
        nsv = convert(Int64, (npulls) * (npulls + 1) / 2)
        nsm = convert(Int64, (L - 1) * L / 2)
        v = zeros(nsv)
        m = zeros(nsm)
        p_success, p_failure, sp_success, sp_failure, not_terminal = build_transitions(L)
        return new(npulls, β, L, v, m, p_success, p_failure, sp_success, sp_failure, not_terminal)
    end
end

function (gi::GittensIndex)(p, q)
    if p + q <= gi.npulls
        sind = pq2ind([p, q], gi.npulls + 1)
        return (1 - gi.β) * gi.v[sind]
    else
        return approx_gittens(gi, p, q)
    end
end

""" Utility Functions """
function pq2ind(x, tot)
    i, j = x
    return j + sum([tot - k for k = 1:i-1])
end

function ind2pq(ind, csum)
    i = findfirst(csum .≥ ind)
    j = i == 1 ? ind : ind - csum[i-1]
    return i, j
end

""" Dynamic Programming """

function calculate_gittens!(gi::GittensIndex; tol=1e-3, max_iter=100000)
    ns = length(gi.v)
    csum = cumsum(gi.npulls:-1:1)

    dp_iter = ProgressBar(1:max_iter)
    set_description(dp_iter, "DP: ")

    for sind in ProgressBar(1:ns)
        p₀, q₀ = ind2pq(sind, csum)
        sind_m = pq2ind([p₀, q₀], gi.L)
        calculate_gittens!(gi, sind_m, tol=tol, dp_iter=dp_iter)
        gi.v[sind] = gi.m[sind_m]
    end
end

function calculate_gittens!(gi::GittensIndex, sind; tol=1e-3, dp_iter=ProgressBar(1:100000))
    converged = false
    for i in 1:100000
        m_old = copy(gi.m)
        dp_step!(gi, sind)

        resid = maximum(abs.(m_old - gi.m))
        set_postfix(dp_iter, Resid="$resid")
        if resid < tol
            converged = true
            break
        end
    end
    !converged ? println("Warning: DP did not converge...") : nothing
end

function build_transitions(L)
    ns = convert(Int64, (L - 1) * L / 2)
    csum = cumsum(L-1:-1:1)

    p_success = zeros(ns)
    p_failure = zeros(ns)
    sp_success = zeros(Int64, ns)
    sp_failure = zeros(Int64, ns)
    not_terminal = ones(Int64, ns)

    iter = ProgressBar(1:ns)
    set_description(iter, "Computing Transitions...")

    for s = iter
        i, j = ind2pq(s, csum)
        if i + j == L
            sp_success[s] = s
            sp_failure[s] = s
            not_terminal[s] = 0
        else
            p_success[s] = i / (i + j)
            sp_success[s] = pq2ind([i + 1, j], L)
            p_failure[s] = 1.0 - p_success[s]
            sp_failure[s] = pq2ind([i, j + 1], L)
        end
    end

    return p_success, p_failure, sp_success, sp_failure, not_terminal
end

function dp_step!(gi, sind)
    w_pq = gi.p_success .* (1.0 .+ β .* gi.m[gi.sp_success]) +
           gi.p_failure .* β .* gi.m[gi.sp_failure]
    w_pq = w_pq .* gi.not_terminal # Zero out terminal 
    gi.m = max.(w_pq, w_pq[sind])
end

""" Approximation """
function approx_gittens(gi, p, q)
    # Whittle's approximation
    # https://www.jhelumch.com/wp-content/uploads/2018/01/Gittins_indices_survey2013.pdf
    n = p + q
    μ = p / n
    c = log(1 / gi.β)

    num = μ * (1 - μ)
    den = n * √((2c + (1 / n)) * μ * (1 - μ)) + μ - 0.5
    return μ + num / den
end

""" Code for calculating """
# function dp_step!(gi, sind)
#     gi.m = max.(gi.R + β * gi.T * gi.m, gi.m[sind])
# end

# npulls = 100
# β = 0.9999
# L = 200
# gi = GittensIndex(npulls, β, L)

# @time calculate_gittens!(gi)

# gi(2, 80)
# gi(2, 2)

# function get_heat(x, y)
#     xi = convert(Int64, round(x))
#     yi = convert(Int64, round(y))
#     return gi(xi, yi)
# end

# heatmap(1:99, 1:99, get_heat, xlabel="α", ylabel="β")
