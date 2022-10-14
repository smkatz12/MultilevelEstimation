struct FunPolicy <: Policy
    f
end

Crux.action_space(p::FunPolicy) = ContinuousSpace(1)

function POMDPs.action(p::FunPolicy, s)
    return p.f(s)
end

function discrete_rule(s)
    return (-8 / π) * s[1] < s[2] ? -1.0 : 1.0
end

function continuous_rule(; k1=0.0, k2=2.0, k3=-1.0)
     (s) -> begin
        ωtarget = sign(s[1])*sqrt(6*10*(1-cos(s[1])))
        -k1*s[1] - k2*s[2] + k3*(ωtarget - s[2])
    end
end