# Code for computing Gittens Allocation Index with discount
# Using the restart formulation of Katehakis and Dermin
using ProgressBars

mutable struct GittensIndex
    npulls::Int64
    β::Float64
    v::Vector
    function GittensIndex(npulls, β)
        nstates = convert(Int64, ((npulls - 1) * npulls) / 2)
        states = LinearIndices((1:npulls-1, 1:npulls-1))
        v = zeros(nstates)
        return new(npulls, β, nstates, states, v)
    end
end

# TODO: function that calls GittensIndex on a state

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
    
    for i = 1:L
        for j = L-i:-1:1
            si = gi.states[i, j]
            if i + j == npulls
                gi.v[si] = 0.0
            else
                si_success = gi.states[i+1, j]
                p_success = i / (i + j)

                si_failure = gi.states[i, j+1]
                p_failure = 1 - p_success

                w_pq = p_success * (1 + β * gi.v[si_success]) + 
                       p_failure * β * gi.v[si_failure]
                
                gi.v[si] = max(w_pq, states[1, 1])
            end
        end
    end
end

npulls = 10
β = 0.99
gi = GittensIndex(npulls, β)

@time dp_step!(gi)