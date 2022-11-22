using SparseArrays
using ProgressBars

mutable struct GittensIndex
    npulls::Int64
    β::Float64
    L::Int64
    v::Vector
    m::Vector
    T::SparseMatrixCSC
    R::SparseMatrixCSC
    function GittensIndex(npulls, β, L)
        nsv = convert(Int64, (npulls) * (npulls + 1) / 2)
        nsm = convert(Int64, (L - 1) * L / 2)
        v = zeros(nsv)
        m = zeros(nsm)
        T, R = build_T_and_R(L)
        return new(npulls, β, L, v, m, T, R)
    end
end

function x2ind(x, tot)
    i, j = x
    return j + sum([tot - k for k = 1:i-1])
end

function ind2x(ind, csum)
    i = findfirst(csum .≥ ind)
    j = i == 1 ? ind : ind - csum[i - 1]
    return i, j
end

function calculate_gittens!(gi::GittensIndex, sind; tol=1e-3, dp_iter=ProgressBar(1:1000))
    for i in dp_iter
        m_old = copy(gi.m)
        dp_step!(gi, sind)

        resid = maximum(abs.(m_old - gi.m))
        set_postfix(dp_iter, Resid="$resid")
        if resid < tol
            break
        end
    end
end

function build_T_and_R(L)
    ns = convert(Int64, (L - 1) * L / 2)
    csum = cumsum(L-1:-1:1)
    T = spzeros(ns, ns)
    R = spzeros(ns, ns)

    iter = ProgressBar(1:ns)
    set_description(iter, "Computing Transition Matrix...")

    for s = iter
        i, j = ind2x(s, csum)
        if i + j == L
            T[s, s] = 1.0
        else
            p_success = i / (i + j)
            sp_success = x2ind([i + 1, j], L)
            p_fail = 1 - p_success
            sp_fail = x2ind([i, j + 1], L)
            T[s, sp_success] = p_success
            R[s, sp_success] = 1.0
            T[s, sp_fail] = p_fail
        end
    end
    
    return T, R
end

function dp_step!(gi, sind)
    gi.m = max.(gi.R + β * gi.T * gi.m, gi.m[sind])
end

npulls = 100
β = 0.99
L = 1000
gi = GittensIndex(npulls, β, L)

@time calculate_gittens!(gi, 1)