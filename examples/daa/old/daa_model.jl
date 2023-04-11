using POMDPs, POMDPGym, Crux
using GridInterpolations
using PyCall
using BSON
using CSV, DataFrames
using LinearAlgebra
using Flux
using Flux: update!, DataLoader
using Plots
using Random

# Load in detection data
detects = BSON.load("/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/data_files/detections_uniform_v3.bson")[:detects]
datafile = "/scratch/smkatz/state_uniform_data/state_data.csv"
df = DataFrame(CSV.File(datafile))
df[1, :]

get_range(e0, n0, u0, e1, n1, u1) = norm([e0, n0, u0] - [e1, n1, u1])
ranges = [get_range(r[2], r[3], r[4], r[6], r[7], r[8]) for r in eachrow(df)]

function get_angles(e0, n0, u0, h0, e1, n1, u1)
    # Unshift
    e1 -= e0
    n1 -= n0
    u1 -= u0

    # Unrotate
    e1rot = e1 * cosd(h0) - n1 * sind(h0)
    n1rot = e1 * sind(h0) + n1 * cosd(h0)

    z = n1rot
    hang = atand(e1rot / z)
    vang = atand(u1 / z)

    return hang, vang
end
hangs = [get_angles(r[2], r[3], r[4], r[5], r[6], r[7], r[8])[1] for r in eachrow(df)]
vangs = [get_angles(r[2], r[3], r[4], r[5], r[6], r[7], r[8])[2] for r in eachrow(df)]

hfov = 38
vfov = 25

inds = findall((abs.(hangs) .≤ hfov) .& (abs.(vangs) .≤ vfov))

x = reshape(ranges[inds], 1, :)
y = reshape(detects[inds], 1, :)

batch_size = 512
nepoch = 500
lr = 1e-3
data = DataLoader((x, y), batchsize=batch_size, shuffle=true, partial=false)
m = Chain(Dense(1, 1), sigmoid)
θ = Flux.params(m)
opt = ADAM(lr)

for e = 1:nepoch
    for (x, y) in data
        _, back = Flux.pullback(() -> Flux.binarycrossentropy(m(x), y), θ)
        update!(opt, θ, back(1.0f0))
    end
    loss_train = Flux.binarycrossentropy(m(x), y)
    println("Epoch: ", e, " Loss Train: ", loss_train)
end

θ # -0.002127558 * r + 1.7336898 # ̂p = -0.0010670999 * r + 0.46792078

get_detect_prob(range) = sigmoid(-0.002127558 * range + 1.7336898)
plot(0:4000, (x) -> get_detect_prob(x))
plot(0:4000, (x) -> m([x])[1])

include("encounter_model/straight_line_model.jl")

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

function perception_model(s0, s1)
    detect = false

    # First check fov
    within_fov = check_fov(s0, s1)

    if within_fov
        # Get detection probability
        range = get_range(s0.x, s0.y, s0.z, s1.x, s1.y, s1.z)
        dprob = get_detect_prob(range)
        if rand() < dprob
            detect = true
        end
    end

    return detect
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

# Create environment
env = CollisionAvoidanceMDP(px=Px, ddh_max=1.0, actions=[-8.0, 0.0, 8.0])
hmax = 500
hs_half = hmax .- (collect(range(0, stop=hmax^(1 / 0.5), length=21))) .^ 0.5
hs = [-hs_half[1:end-1]; reverse(hs_half)]
dhs = range(-10, 10, length=21)
τs = range(0, 40, length=41)

# Get optimal policy
policy = OptimalCollisionAvoidancePolicy(env, hs, dhs, τs)

# Get 100 encounters
Random.seed!(13)
encs = get_encounter_set(sampler, 100)
nmacs = sum([is_nmac(enc) for enc in encs])

# Rotate and shift
new_encs = rotate_and_shift_encs(encs)
nmacs = sum([is_nmac(enc) for enc in new_encs])

# Simulate
sim_encs = [simulate_encounter(enc, policy, perfect_perception; seed=i) for (i, enc) in enumerate(encs)]
sim_nmacs = sum([is_nmac(enc) for enc in sim_encs])

# Try with XPlane
struct XPlaneControl
    util
    client
end

function XPlaneControl()
    if pyimport("sys")."path"[1] != "/home/smkatz/Documents/MultilevelEstimation/examples/daa/"
        pushfirst!(pyimport("sys")."path", "/home/smkatz/Documents/MultilevelEstimation/examples/daa/")
    end
    xplane_ctrl = pyimport("util")
    xplane_ctrl = pyimport("importlib")["reload"](xplane_ctrl)
    xpc3 = pyimport("xpc3")
    xplane_client = xpc3.XPlaneConnect()
    xplane_client.pauseSim(true)
    xplane_client.sendDREF("sim/operation/override/override_joystick", 1)

    XPlaneControl(xplane_ctrl, xplane_client)
end

function bb_center(s0, s1; hfov=80.0, vfov=49.5, sw=1920, sh=1056)
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

    # Map to pixel location
    xp = xp * sw
    yp = (1 - yp) * sh

    return xp, yp
end

function perception_xplane(s0, s1; bb_error_tol=100.0)
    bb, boxes = xctrl.util.get_bounding_box(xctrl.client, uniform_v3, s0.x, s0.y, s0.z, s0.θ, s1.x, s1.y, s1.z, s1.θ, -1)
    if bb
        xp_gt, yp_gt = bb_center(s0, s1)
        min_error = Inf
        for i = 1:size(boxes, 1)
            xp, yp, _, _ = boxes[i, :]
            e = norm([xp, yp] - [xp_gt, yp_gt])
            if e < min_error
                min_error = e
            end
        end
        if min_error > bb_error_tol
            bb = false
        end
    end
    return bb
end

xctrl = XPlaneControl()
uniform_v3 = xctrl.util.load_model("/home/smkatz/Documents/RiskSensitivePerception/collision_avoidance/models/uniform_v3.pt")
xctrl.util.get_bounding_box(xctrl.client, uniform_v3, 0, 0, 0, 0, 0, 200, 0, 0, -1)

xplane_encs = [simulate_encounter(enc, policy, perception_xplane; seed=i) for (i, enc) in enumerate(new_encs)]
xplane_nmacs = sum([is_nmac(enc) for enc in xplane_encs])

function perception_model_v2(s0, s1)
    detect = false

    # First check fov
    within_fov = check_fov(s0, s1)

    if within_fov
        # Get detection probability
        range = get_range(s0.x, s0.y, s0.z, s1.x, s1.y, s1.z)
        a = -0.002
        b = 2.0
        c = 0.05
        dprob = sigmoid(a * range + b) + c
        if rand() < dprob
            detect = true
        end
    end

    return detect
end

model_encs = [simulate_encounter(enc, policy, perception_model_v2; seed=i) for (i, enc) in enumerate(new_encs)];
model_nmacs = sum([is_nmac(enc) for enc in model_encs])

histogram(ranges)
histogram(ranges[detects.==1])

bin_edges = collect(100:100:4500)
bin_vals = zeros(length(bin_edges) - 1)
bin_totals = zeros(length(bin_edges) - 1)

filtered_ranges = ranges[inds]
filtered_detects = detects[inds]
for i = 1:length(filtered_ranges)
    # Find bin
    r = filtered_ranges[i]
    b = findfirst(bin_edges .> r)
    bin_totals[b] += 1
    if filtered_detects[i] == 1
        bin_vals[b] += 1
    end
end

bin_vals ./ bin_totals
p = bar(bin_edges[1:end-1], bin_vals ./ bin_totals)
plot!(p, 0:4000, (x) -> get_detect_prob(x), lw=4)

function test_vals(a, b, c)
    dp(r) = sigmoid(a * r + b) + c
    p = bar(bin_edges[1:end-1], bin_vals ./ bin_totals)
    plot!(p, 0:4000, (x) -> dp(x), lw=4)
end

a = -0.002 #-0.002127558
b = 2.0 #1.7336898
c = 0.05
test_vals(a, b, c)

# Try with 200 encounters
Random.seed!(13)
encs = get_encounter_set(sampler, 200)
nmacs = sum([is_nmac(enc) for enc in encs])

# Rotate and shift
new_encs = rotate_and_shift_encs(encs)
nmacs = sum([is_nmac(enc) for enc in new_encs])

xplane_encs = [simulate_encounter(enc, policy, perception_xplane; seed=i) for (i, enc) in enumerate(new_encs)]
xplane_nmacs = sum([is_nmac(enc) for enc in xplane_encs])

model_encs = [simulate_encounter(enc, policy, perception_model_v2; seed=i) for (i, enc) in enumerate(new_encs)];
model_nmacs = sum([is_nmac(enc) for enc in model_encs])

model_old_encs = [simulate_encounter(enc, policy, perception_model; seed=i) for (i, enc) in enumerate(new_encs)];
model_old_nmacs = sum([is_nmac(enc) for enc in model_old_encs])