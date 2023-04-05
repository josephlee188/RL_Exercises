function init_wind()
    wind = zeros(Int, WIDTH)

    wind[4] = wind[5] = wind[6] = wind[9] = 1
    wind[7] = wind[8] = 2

    return wind
end

function take_step(s, a, wind, start, goal)
    # Play 'a' step in the game
    y0, x0 = s
    a_y, a_x = a

    x = x0 + a_x

    if x > WIDTH
        x = WIDTH
    elseif x < 1
        x = 1
    end

    y = y0 + a_y + wind[x0]

    if y > HEIGHT
        y = HEIGHT
    elseif y < 1
        y = 1
    end

    # if (y, x) == goal
    #     return start
    # else
    #     return (y, x)
    # end

    return (y, x)
end

function init_Q()
    Q = Dict()
    E = Dict()  # Eligibility traces
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for y = 1:HEIGHT
        for x = 1:WIDTH
            for (ay, ax) = actions
                Q[y,x,(ay,ax)] = 0.0
                E[y,x,(ay,ax)] = 0.0

                # if (y,x) == goal
                #     Q[y,x,ay,ax] = 0.0
                # else
                #     Q[y,x,ay,ax] = rand()
                # end
            end
        end
    end
    return Q, E, actions
end

function ϵ_greedy(Q, s, actions)
    Q_val = Q[s..., actions[1]]
    ay, ax = actions[1]

    if ϵ < rand()   # greedy
        for a = actions
            if Q[s..., a] >= Q_val
                Q_val = Q[s..., a]
                ay, ax = a
            end
        end

    else        # random policy
        ay, ax = rand(actions)
    end

    return (ay, ax)
end

function update_Q(Q,s,a,r,s′,a′)
    Q[s..., a] += α * (r + γ * Q[s′..., a′] - Q[s..., a])
end

function update_Q_E(Q,E,actions,s,a,r,s′,a′)
    δ = r + γ * Q[s′..., a′] - Q[s..., a]
    E[s..., a] += 1
    # for all states
    for (s1, s2, act) in collect(keys(Q))
        Q[s1, s2, act] += α * δ * E[s1, s2, act]
        E[s1, s2, act] = λ * γ * E[s1, s2, act]
    end
end

function sarsa_TD_control(Q, E, args...; return_path = false, use_et = false)
    start, goal, actions, wind = args

    s = start
    a = ϵ_greedy(Q, s, actions)

    nsteps = 0

    path = []
    push!(path, s)
    # One 'episode'
    while s != goal
        s′ = take_step(s, a, wind, start, goal)
        if s′ != goal
            r = -1
        else
            r = 0
        end
        a′ = ϵ_greedy(Q, s′, actions)

        if use_et   # use eligibility traces?
            update_Q_E(Q,E,actions,s,a,r,s′,a′)
        else
            update_Q(Q,s,a,r,s′,a′)
        end

        s = s′
        a = a′
        push!(path, s)
        nsteps += 1
    end

    if return_path
        return nsteps, path
    else
        return nsteps
    end
end

function run_episodes(Epi, use_et)
    Q, E, actions = init_Q()
    wind = init_wind()

    start = (4, 1)
    goal = (4, 8)

    args = (start, goal, actions, wind)

    nsteps_list = []
    path = []

    for epi = 1:Epi
        if epi != Epi
            nsteps = sarsa_TD_control(Q, E, args..., return_path = false, use_et = use_et)
        else
            nsteps, path = sarsa_TD_control(Q, E, args..., return_path = true, use_et = use_et)
        end
        push!(nsteps_list, nsteps)

        print("$epi ")
    end

    return nsteps_list, path
end
