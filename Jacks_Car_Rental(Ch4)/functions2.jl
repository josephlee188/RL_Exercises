function expected_return(V,s,a,args...)
    # Current states
    n1, n2 = s

    # Cars moved (overnight)
    n1 -= a
    n2 += a

    n1 = max(min(n1, S), 0)
    n2 = max(min(n2, S), 0)

    rewards = 0.0

    rewards -= 2.0 * abs(a)

    # if a <= 0
    #     rewards -= 2.0 * abs(a)
    # else
    #     rewards -= 2.0 * (a - 1)    # One car can be moved for free
    # end

    p1, p2, p3, p4 = args

    for req1 = 0:length(p1)-1     # Request 1
        for ret1 = 0:length(p2)-1     # Return 1
            for req2 = 0:length(p3)-1     # Request 2
                for ret2 = 0:length(p4)-1     # Return 2

                    prob = p1[req1+1] * p2[ret1+1] * p3[req2+1] * p4[ret2+1]

                    valid_req_1 = min(req1, n1)
                    valid_req_2 = min(req2, n2)
                    # valid_req_1 = req1
                    # valid_req_2 = req2

                    new_r = 10.0 * (valid_req_1 + valid_req_2)

                    n1_new = min(n1 - valid_req_1 + ret1, S)
                    n2_new = min(n2 - valid_req_2 + ret2, S)

                    # Bellman's equation
                    rewards += prob * (new_r + γ * V[n1_new+1, n2_new+1])
                end
            end
        end
    end

    return rewards
end

function update_QV(V, Q, Π, args...)
    for a = -A:A
        for s₁ = 0:S
            for s₂ = 0:S
                Q[a+A+1, s₁+1, s₂+1] = expected_return(V, [s₁, s₂], a, args...)
            end
        end
    end
    V .= dropdims(sum(Q .* Π, dims=1), dims=1)
end

function policy_evaluation(V, Q, Π, args...)
    errs = zero(V)
    V_copy = zero(V)
    Δ = θ
    while Δ >= θ
        V_copy .= V
        update_QV(V, Q, Π, args...)
        errs .= abs.(V_copy .- V)
        Δ = maximum(errs)
    end
end

function policy_improvement(V, Q, Π)
    is_stable = true
    for i = 0:S
        for j = 0:S
            s = [i, j]
            # Action is valid for va_1:va_2
            va_1 = -min(j,A)    # valid action 1    (n2 -> n1)
            va_2 = min(i,A)     # valid action 2    (n1 -> n2)
            ind1 = va_1 + A + 1
            ind2 = va_2 + A + 1
            # ind_s = s .+ 1
            ind_old = findfirst(Π[:, i+1,j+1] .== 1.0)
            ind_new = argmax(Q[ind1:ind2, i+1,j+1]) + ind1 - 1
            Π[ind_old, i+1,j+1] = 0.0
            Π[ind_new, i+1,j+1] = 1.0

            if ind_old != ind_new
                is_stable = false
            end
        end
    end
    return is_stable
end

function initialize()
    V = zeros(S+1, S+1)         # zeros(States, States)
    Q = zeros(2A+1, S+1, S+1)    # zeros(Actions, States, States)
    Π = zeros(2A+1, S+1, S+1)    # zeros(Actions, States, States)
    # Policy initialized to at a = 0, the probability is 1
    Π[6, :, :] .= 1.0
    return V, Q, Π
end

function policy_iteration(V, Q, Π, args...)
    # Combine evalutation, improvement.
    is_stable = false
    trial = 0
    Policies = []
    while !is_stable
        policy_evaluation(V, Q, Π, args...)
        is_stable = policy_improvement(V, Q, Π)
        println("Day : " * string(trial))
        trial += 1
        push!(Policies, optimal_actions(Π))
    end
    # Policy = optimal_actions(Π)
    # return Policy
    return Policies
end

function optimal_actions(Π)
    Policy = zeros(Int, S+1, S+1)
    for i = 0:S
        for j = 0:S
            a = Π[:, i+1, j+1]
            f = findfirst(a .== 1.0)
            Policy[i+1, j+1] = f - A - 1
        end
    end
    return Policy
end

norm(vec) = begin
    if sum(vec) > 0.0
        vec .= vec ./ sum(vec)
    else
        vec .= 0.0
    end
    vec
end
