# Code for computing Gittens Allocation Index with discount
# Using the restart formulation of Katehakis and Dermin
using ProgressBars
using LinearAlgebra

mutable struct GittensIndex
    npulls::Int64
    β::Float64
    v::AbstractMatrix
    function GittensIndex(npulls, β)
        v = zeros(npulls - 1, npulls - 1)
        return new(npulls, β, v)
    end
end

(gi::GittensIndex)(p, q) = (1 - gi.β) * gi.v[p, q]

function calculate_gittens!(gi::GittensIndex; tol=1e-3, max_iter=1000)    
    iter = ProgressBar(1:max_iter)
    for i in iter
        v_old = copy(gi.v)
        dp_step!(gi)

        resid = maximum(abs.(v_old - gi.v))
        set_postfix(iter, Resid="$resid")
        if resid < tol
            break
        end
    end
end

function dp_step!(gi)
    L = gi.npulls
    
    for p = 1:L
        for q = L-p:-1:1
            if p + q == npulls
                gi.v[p, q] = 0.0
            else
                p_success = p / (p + q)
                p_failure = 1 - p_success

                w_pq = p_success * (1 + β * gi.v[p + 1, q]) + 
                       p_failure * β * gi.v[p, q + 1]
                
                gi.v[p, q] = max(w_pq, gi.v[1, 1])
            end
        end
    end
end

# npulls = 1000
# β = 0.99
# gi = GittensIndex(npulls, β)

# @time dp_step!(gi)

# @time calculate_gittens!(gi)

# gi(1, 1)
# gi(1, 100)
# gi(100, 1)