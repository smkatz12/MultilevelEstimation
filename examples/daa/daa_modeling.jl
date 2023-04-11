using POMDPs, POMDPGym, Crux
using GridInterpolations
using PyCall
using BSON
using BSON: @save
using CSV, DataFrames
using LinearAlgebra
using Plots
using Random
using ProgressBars
using Distributions

include("encounter_model/straight_line_model.jl")

res = BSON.load("examples/daa/datafiles/model_detections.bson")
r = res[:r]
db = res[:db]
dr = res[:dr]

bin_edges = collect(0:100:4500)
bin_vals_b = zeros(length(bin_edges) - 1)
bin_vals_r = zeros(length(bin_edges) - 1)
bin_totals = zeros(length(bin_edges) - 1)

filtered_ranges = ranges[inds]
filtered_detects = detects_risk[inds]
for i = 1:length(filtered_ranges)
    # Find bin
    r = filtered_ranges[i]
    b = findfirst(bin_edges .> r)
    bin_totals[b] += 1
    if db[i] == 1
        bin_vals_b[b] += 1
    end
    if dr[i] == 1
        bin_vals_r[b] += 1
    end
end

p = bar(bin_edges[1:end-1], bin_vals_b ./ bin_totals)
p = bar(bin_edges[1:end-1], bin_vals_r ./ bin_totals)

function test_vals(x₀, y₀, bin_vals)
    dp(r) = -(y₀ / x₀) * r + y₀
    p = bar((bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2, bin_vals ./ bin_totals)
    plot!(p, 0:4000, (x) -> dp(x), lw=4, ylims=(0, 1), xlims=(200, 2000))
    return p
end

test_vals(1900, 1.15, bin_vals_b)
test_vals(2700, 1.05, bin_vals_r)

# Try simulating
function check_fov(s0, s1; hfov=80.0, vfov=49.5)
    # Make ownship be the origin
    x = s1.y - s0.y
    y = -(s1.x - s0.x)  # right-handed coordinates
    z = s1.z - s0.z

    # Rotate x and y according to ownship heading
    xrot = x * cosd(-s0.θ) - y * sind(-s0.θ)
    yrot = -(x * sind(-s0.θ) + y * cosd(-s0.θ))

    # https://www.youtube.com/watch?v=LhQ85bPCAJ8
    xp = yrot / (xrot * tand(hfov / 2))
    yp = z / (xrot * tand(vfov / 2))

    # Get xp and yp between 0 and 1
    xp = (xp + 1) / 2
    yp = (yp + 1) / 2

    return (0 < xp < 1) && (0 < yp < 1)
end

get_range(e0, n0, u0, e1, n1, u1) = norm([e0, n0, u0] - [e1, n1, u1])

# Simulate encounters with the model
const HNMAC = 100
const VNMAC = 50

const DDH_MAX = 1.0
const Px = DiscreteNonParametric([-0.5, 0.0, 0.5], [0.1, 0.8, 0.1])

function step_aircraft(s, a=0.0)
    h = s.z
    dh = s.dh

    h = h + dh
    if abs(a - dh) < DDH_MAX
        dh += a - dh
    else
        dh += sign(a - dh) * DDH_MAX
    end
    dh += rand(Px)
    h, dh
end

function mdp_state(s0, s1, a_prev)
    h = s0.z - s1.z
    dh = s0.dh - s1.dh

    dt = 0.1
    r0 = [s0.x, s0.y]
    r0_next = r0 + s0.v * dt * [-sind(s0.θ), cosd(s0.θ)]

    r1 = [s1.x, s1.y]
    r1_next = r1 + s1.v * dt * [-sind(s1.θ), cosd(s1.θ)]

    r = norm(r0 - r1)
    r_next = norm(r0_next - r1_next)

    ṙ = (r - r_next) / dt

    τ = r < HNMAC ? 0 : (r - HNMAC) / ṙ
    if τ < 0
        τ = Inf
    end

    [h, dh, a_prev, τ]
end

function simulate_encounter(enc::Encounter, policy, perception; seed=1)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """
    Random.seed!(seed)

    s0s = []
    s1s = []
    N = length(enc.x0)
    a_prev = 0
    s0 = get_ownship_state(enc, 1)
    s1 = get_intruder_state(enc, 1)
    z0, dh0 = s0.z, s0.dh
    z1, dh1 = s1.z, s1.dh
    as = []
    for t in 1:N
        s0 = get_ownship_state(enc, t)
        s0 = (x=s0.x, y=s0.y, z=z0, v=s0.v, dh=dh0, θ=s0.θ)
        s1 = get_intruder_state(enc, t)
        s1 = (x=s1.x, y=s1.y, z=z1, v=s1.v, dh=dh1, θ=s1.θ)

        # Store the state
        push!(s0s, s0)
        push!(s1s, s1)

        # Optionally call python to set state and take screenshot
        bb = perception(s0, s1)

        # compute the next state
        a = bb ? action(policy, mdp_state(s0, s1, a_prev)) : 0.0
        a_prev = a
        push!(as, a)

        z0, dh0 = step_aircraft(s0, a)
        z1, dh1 = step_aircraft(s1)
    end

    return Encounter(s0s, s1s, as)
end

perfect_perception(s0, s1) = true

function is_nmac(enc::Encounter)
    for i = 1:length(enc.x0)
        hsep = sqrt((enc.x0[i] - enc.x1[i])^2 + (enc.y0[i] - enc.y1[i])^2)
        vsep = abs(enc.z0[i] - enc.z1[i])
        if hsep < HNMAC && vsep < VNMAC
            return true
        end
    end
    return false
end

function perception_model(s0, s1; x₀=1500, y₀=1.1)
    detect = false

    # First check fov
    within_fov = check_fov(s0, s1)

    if within_fov
        # Get detection probability
        range = get_range(s0.x, s0.y, s0.z, s1.x, s1.y, s1.z)
        if 100 < range < 2000
            dprob = -(y₀ / x₀) * range + y₀
            if rand() < dprob
                detect = true
            end
        end
    end

    return detect
end

# Create environment
env = CollisionAvoidanceMDP(px=Px, ddh_max=1.0, actions=[-8.0, 0.0, 8.0])
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

# Get optimal policy
policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)

# Try with 200 encounters
Random.seed!(13)
encs = get_encounter_set(sampler, 200)
nmacs = sum([is_nmac(enc) for enc in encs])

# Rotate and shift
new_encs = rotate_and_shift_encs(encs)
nmacs = sum([is_nmac(enc) for enc in new_encs])

baseline_perception(s0, s1) = perception_model(s0, s1, x₀=1900, y₀=1.15)
risk_perception(s0, s1) = perception_model(s0, s1, x₀=2700, y₀=1.05)

baseline_encs = [simulate_encounter(enc, policy, baseline_perception; seed=i) for (i, enc) in enumerate(new_encs)];
baseline_nmacs = sum([is_nmac(enc) for enc in baseline_encs])

risk_encs = [simulate_encounter(enc, policy, risk_perception; seed=i) for (i, enc) in enumerate(new_encs)];
risk_nmacs = sum([is_nmac(enc) for enc in risk_encs])



nothing

# if pyimport("sys")."path"[1] != "/home/smkatz/Documents/MultilevelEstimation/examples/daa/"
#     pushfirst!(pyimport("sys")."path", "/home/smkatz/Documents/MultilevelEstimation/examples/daa/")
# end

# xplane_ctrl = pyimport("util")

# datafile = "/scratch/smkatz/state_uniform_data/state_data.csv"
# df = DataFrame(CSV.File(datafile))
# df[1, :]

# get_range(e0, n0, u0, e1, n1, u1) = norm([e0, n0, u0] - [e1, n1, u1])
# ranges = [get_range(r[2], r[3], r[4], r[6], r[7], r[8]) for r in eachrow(df)]

# model_baseline = xplane_ctrl.load_model("/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/models/uniform_v3.pt")
# model_risk = xplane_ctrl.load_model("/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/models/risk_v3_rl.pt")

# function bb_center(e0, n0, u0, h0, e1, n1, u1; hfov=80.0, vfov=49.5, sw=1920, sh=1056)
#     # Unshift
#     e1 -= e0
#     n1 -= n0
#     u1 -= u0

#     # Unrotate
#     yrot = e1 * cosd(h0) - n1 * sind(h0)
#     xrot = e1 * sind(h0) + n1 * cosd(h0)

#     # https://www.youtube.com/watch?v=LhQ85bPCAJ8
#     xp = yrot / (xrot * tand(hfov / 2))
#     yp = u1 / (xrot * tand(vfov / 2))

#     # Get xp and yp between 0 and 1
#     xp = (xp + 1) / 2
#     yp = (yp + 1) / 2

#     # Map to pixel location
#     xp = xp * sw
#     yp = (1 - yp) * sh

#     return xp, yp
# end

# detects_baseline = zeros(10000)
# for i = ProgressBar(0:9999)
#     filename = "/scratch/smkatz/state_uniform_data/imgs/$i.jpg"
#     bb, boxes = xplane_ctrl.bb_from_file(model_baseline, filename)
#     if bb
#         xp_gt, yp_gt = bb_center(df[i+1, 2], df[i+1, 3], df[i+1, 4], df[i+1, 5], df[i+1, 6], df[i+1, 7], df[i+1, 8])
#         min_error = Inf
#         for j = 1:size(boxes, 1)
#             xp, yp, _, _ = boxes[j, :]
#             e = norm([xp, yp] - [xp_gt, yp_gt])
#             if e < min_error
#                 min_error = e
#             end
#         end
#         if min_error < 100.0
#             detects_baseline[i+1] = true
#         end
#     end
# end

# detects_risk = zeros(10000)
# for i = ProgressBar(0:9999)
#     filename = "/scratch/smkatz/state_uniform_data/imgs/$i.jpg"
#     bb, boxes = xplane_ctrl.bb_from_file(model_risk, filename)
#     if bb
#         xp_gt, yp_gt = bb_center(df[i+1, 2], df[i+1, 3], df[i+1, 4], df[i+1, 5], df[i+1, 6], df[i+1, 7], df[i+1, 8])
#         min_error = Inf
#         for j = 1:size(boxes, 1)
#             xp, yp, _, _ = boxes[j, :]
#             e = norm([xp, yp] - [xp_gt, yp_gt])
#             if e < min_error
#                 min_error = e
#             end
#         end
#         if min_error < 100.0
#             detects_risk[i+1] = true
#         end
#     end
# end

# function check_fov(i; hfov=80.0, vfov=49.5)
#     e0, n0, u0, h0, e1, n1, u1 = df[i+1, 2], df[i+1, 3], df[i+1, 4], df[i+1, 5], df[i+1, 6], df[i+1, 7], df[i+1, 8]
#     # Unshift
#     e1 -= e0
#     n1 -= n0
#     u1 -= u0

#     # Unrotate
#     yrot = e1 * cosd(h0) - n1 * sind(h0)
#     xrot = e1 * sind(h0) + n1 * cosd(h0)

#     # https://www.youtube.com/watch?v=LhQ85bPCAJ8
#     xp = yrot / (xrot * tand(hfov / 2))
#     yp = u1 / (xrot * tand(vfov / 2))

#     # Get xp and yp between 0 and 1
#     xp = (xp + 1) / 2
#     yp = (yp + 1) / 2

#     return (0 < xp < 1) && (0 < yp < 1)
# end

# inds = findall([check_fov(i) for i in 0:9999])

# r = ranges[inds]
# db = detects_baseline[inds]
# dr = detects_risk[inds]

# @save "examples/daa/datafiles/model_detections.bson" r db dr