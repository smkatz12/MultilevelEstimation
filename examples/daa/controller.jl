# Simple collision avoidance controller
const Px = DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1])

env = CollisionAvoidanceMDP(px=Px, ddh_max=1.0, actions=[-8.0, 0.0, 8.0])
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

# Get optimal policy
policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)