# Code for computing Gittens Allocation Index with discount
# Using the restart formulation of Katehakis and Dermin
using ProgressBars
using LinearAlgebra
using Plots

mutable struct GittensIndex
    npulls::Int64
    β::Float64
    L::Int64
    v::AbstractMatrix
    m::AbstractMatrix
    function GittensIndex(npulls, β, L)
        v = zeros(npulls, npulls)
        m = zeros(L - 1, L - 1)
        return new(npulls, β, L, v, m)
    end
end

function (gi::GittensIndex)(p, q)
    if p + q <= gi.npulls
        return (1 - gi.β) * gi.v[p, q]
    else
        error("Exceeded max pulls")
    end
end

function calculate_gittens!(gi::GittensIndex; tol=1e-3, max_iter=1000)
    npulls = gi.npulls

    dp_iter = ProgressBar(1:max_iter)
    set_description(dp_iter, "DP: ")

    for ptot in npulls:-1:2
        # print(ptot)
        for p₀ in 1:ptot-1
            # print(p₀)
            if p₀ > ptot - 1
                break
            end
            q₀ = ptot - p₀
            calculate_gittens!(gi, p₀, q₀, tol=tol, dp_iter=dp_iter)
            gi.v[p₀, q₀] = gi.m[p₀, q₀]
        end
    end
end

function calculate_gittens!(gi::GittensIndex, p₀, q₀; tol=1e-3, dp_iter=ProgressBar(1:1000))
    for i in dp_iter
        m_old = copy(gi.m)
        dp_step_faster!(gi, p₀, q₀)
        #gi.m = max.(gi.m, gi.m[p₀, q₀])

        resid = maximum(abs.(m_old - gi.m))
        set_postfix(dp_iter, Resid="$resid")
        if resid < tol
            break
        end
    end
end

function dp_step!(gi, p₀, q₀)
    L = gi.L

    for p = 1:L
        for q = L-p:-1:1
            if p + q == gi.L # check this
                gi.m[p, q] = 0.0
            else
                p_success = p / (p + q)
                p_failure = 1 - p_success

                w_pq = p_success * (1 + β * gi.m[p+1, q]) +
                       p_failure * β * gi.m[p, q+1]

                gi.m[p, q] = max(w_pq, gi.m[p₀, q₀])
            end
        end
    end
end

function dp_step_faster!(gi, p₀, q₀)
    L = gi.L

    for ptot = L-1:-1:2
        for p = 1:ptot-1
            q = ptot - p
            # println("p: ", p, " q: ", q)

            p_success = p / (p + q)
            p_failure = 1 - p_success

            w_pq = p_success * (1 + β * gi.m[p+1, q]) +
                   p_failure * β * gi.m[p, q+1]

            if (p == p₀) && (q == q₀)
                gi.m[p, q] = w_pq 
            else
                gi.m[p, q] = max(w_pq, gi.m[p₀, q₀])
            end
            # println(w_pq)
        end
    end
end

npulls = 10
β = 0.99
L = 21
gi = GittensIndex(npulls, β, L)

# @time dp_step_faster!(gi, 1, 1)

@time calculate_gittens!(gi, max_iter=1000)

gi(2, 7)
gi(2, 2)

# function get_heat(x, y)
#     xi = convert(Int64, round(x))
#     yi = convert(Int64, round(y))
#     if xi + yi >= gi.npulls
#         return 0.99
#     elseif xi == 0 || yi == 0
#         return 0.99
#     else
#         return gi(xi, yi)
#     end
# end

# heatmap(1:99, 1:99, get_heat, xlabel="α", ylabel="β")

# gi(1, 1)
# gi(1, 100)
# gi(100, 1)