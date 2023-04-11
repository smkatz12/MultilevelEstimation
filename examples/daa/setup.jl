using POMDPs, POMDPGym, Crux, Distributions, BSON, GridInterpolations, LinearAlgebra

include("controller.jl")
include("encounter_model/straight_line_model.jl")
include("../../src/multilevel_estimation.jl")

# CAS parameters
const HNMAC = 100
const VNMAC = 50

const DDH_MAX = 1.0

function daa_problem(nx₀, ny₀, nf; x₀min=1000, x₀max=3000, y₀min=0.8, y₀max=1.2, fmin=30.0, fmax=100.0,
     threshold=0.3, conf_threshold=0.9)

    # Set up grid
    x₀s = collect(range(x₀min, stop=x₀max, length=nx₀))
    y₀s = collect(range(y₀min, stop=y₀max, length=ny₀))
    fs = collect(range(fmin, stop=fmax, length=nf))
    grid_points = Dict(:x₀s => x₀s, :y₀s => y₀s, :fs => fs)
    grid = RectangleGrid(x₀s, y₀s, fs)

    function sim(params, nsamps)
        # Generate the encounters
        encs = get_encounter_set(sampler, nsamps)
        perception(s0, s1) = perception_model(s0, s1, x₀=params[1], y₀=params[2], hfov=params[3])
        sim_encs = [simulate_encounter(enc, policy, perception) for enc in encs]
        nmacs = sum([is_nmac(sim_enc) for sim_enc in sim_encs])
        return nmacs
    end

    return GriddedProblem(grid_points, grid, sim, threshold, conf_threshold, falses(length(grid)))
end

function simulate_encounter(enc::Encounter, policy, perception; seed=1)
    """
    Inputs:
    - enc (Encounter): encounter to simulate (see encounter model file for details)
    - policy (OptimalCollisionAvoidancePolicy): policy for ownship to use
    Outputs:
    - sim_enc (Encounter): encounter object with simulated trajectories
    """
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

function perception_model(s0, s1; x₀=1500, y₀=1.1, hfov=80.0)
    detect = false

    # First check fov
    within_fov = check_fov(s0, s1, hfov=hfov)

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