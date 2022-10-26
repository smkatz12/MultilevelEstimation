using Distributions, Plots

dist1 = Beta(20, 80)
dist2 = Beta(80, 20)

p = plot(0:0.01:1.0, x -> pdf(dist1, x), legend=false, xlabel="P(failure)", lw=2, linealpha=0.5)
plot!(p, 0:0.01:1.0, x -> pdf(dist2, x), lw=2, linealpha=0.5)

function plot_interp_frame(point, dist1, dist2; endpoints=[0.0, 2.0])
    w_interp = (point - endpoints[1]) / (endpoints[2] - endpoints[1])
    α_interp = (1 - w_interp) * dist1.α + w_interp * dist2.α
    β_interp = (1 - w_interp) * dist1.β + w_interp * dist2.β
    dist_interp = Beta(α_interp, β_interp)

    p1 = plot(0:0.01:1.0, x -> pdf(dist1, x), legend=false, xlabel="P(failure)", lw=2, linealpha=0.5, linecolor=:blue)
    plot!(p1, 0:0.01:1.0, x -> pdf(dist2, x), lw=2, linealpha=0.5, linecolor=:red)
    plot!(p1, 0:0.01:1.0, x -> pdf(dist_interp, x), lw=2, linealpha=1.0, linecolor=:purple)

    p2 = plot([endpoints], [dist1.α, dist2.α], linecolor=:black, lw=2, linealpha=0.5, legend=false,
        xlabel="x", ylabel="pseudocount")
    plot!(p2, [endpoints], [dist1.β, dist2.β], linecolor=:black, lw=2, linealpha=0.5)
    scatter!(p2, [point, point], [α_interp, β_interp], markercolor=:purple, markerstrokecolor=:purple)

    p = plot(p1, p2, layout=(2, 1))

    return p
end

p = plot_interp_frame(0.1, dist1, dist2)

anim = @animate for point in 0.0:0.05:2.0
    plot_interp_frame(point, dist1, dist2)
end
Plots.gif(anim, "figs/count_interp.gif")

# Pseudocount decay
sqe_kernel(r; ℓ=0.3) = exp(-r^2 / (2 * ℓ^2))
function get_count(point, count1, count2; endpoints=[0.0, 2.0], ℓ=0.3)
    r1 = point - endpoints[1]
    r2 = point - endpoints[2]

    k1 = sqe_kernel(r1, ℓ=ℓ)
    k2 = sqe_kernel(r2, ℓ=ℓ)

    return 1 + k1 * count1 + k2 * count2
end

p = plot(0:0.05:2.0, x -> get_count(x, dist1.α, dist2.α), legend=false)
plot!(p, 0:0.05:2.0, x -> get_count(x, dist1.β, dist2.β))

function plot_decay_frame(point, dist1, dist2; endpoints=[0.0, 2.0], ℓ=ℓ)
    α_interp = get_count(point, dist1.α, dist2.α, endpoints=endpoints, ℓ=ℓ)
    β_interp = get_count(point, dist1.β, dist2.β, endpoints=endpoints, ℓ=ℓ)
    dist_interp = Beta(α_interp, β_interp)

    p1 = plot(0:0.01:1.0, x -> pdf(dist1, x), legend=false, xlabel="P(failure)", lw=2, linealpha=0.5, linecolor=:blue)
    plot!(p1, 0:0.01:1.0, x -> pdf(dist2, x), lw=2, linealpha=0.5, linecolor=:red)
    plot!(p1, 0:0.01:1.0, x -> pdf(dist_interp, x), lw=2, linealpha=1.0, linecolor=:purple)

    # p2 = plot(0:0.05:2.0, x -> get_count(x, dist1.α, dist2.α, ℓ=ℓ), linecolor=:black, lw=2, linealpha=0.5, legend=false,
    #     xlabel="x", ylabel="pseudocount")
    # plot!(p2, 0:0.05:2.0, x -> get_count(x, dist1.β, dist2.β, ℓ=ℓ), linecolor=:black, lw=2, linealpha=0.5)
    # scatter!(p2, [point, point], [α_interp, β_interp], markercolor=:purple, markerstrokecolor=:purple)

    # p = plot(p1, p2, layout=(2, 1))

    return p1
end

p = plot_decay_frame(0.1, dist1, dist2)

anim = @animate for point in 0.0:0.05:2.0
    plot_decay_frame(point, dist1, dist2)
end
Plots.gif(anim, "figs/count_decay.gif")

dist3 = Beta(2, 5)
dist4 = Beta(50, 20)
anim = @animate for point in 0.0:0.05:2.0
    plot_decay_frame(point, dist3, dist4, ℓ=0.7)
end
Plots.gif(anim, "figs/count_decay_v2.gif")