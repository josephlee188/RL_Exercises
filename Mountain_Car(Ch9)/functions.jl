function model_update(x, v, a)
    x′ = x + v
    v′ = v + 0.001 * a - 0.0025 * cos(3*x)
    is_goal = false

    if x′ < XMIN
        v′ = 0.0
        x′ = XMIN
    elseif x > XMAX
        is_goal = true
    end

    v′ = max(min(v′, VMAX), VMIN)

    r = -1
    if is_goal
        r = 0
    end

    return x′, v′, r
end

rescale(x, xmin, xmax) = (x - xmin) / (xmax - xmin)

function F(x, v, a, offsets, feat_ind)
    """
    "Let Fₐ be a set of feature indices for every possible action a."

    Returns the indicies of features that are 1, provided (x, v)

    """
    x0 = rescale(x, XMIN, XMAX)     # 0 ~ 1
    v0 = rescale(v, VMIN, VMAX)     # 0 ~ 1

    x0_rem = mod(x0, 1/NX)      # 0 ~ 1/NX
    v0_rem = mod(v0, 1/NV)      # 0 ~ 1/NV

    x0_ind = round((x0 - x0_rem) / (1/NX))     # 0 ~ NX-1
    v0_ind = round((v0 - v0_rem) / (1/NV))     # 0 ~ NV-1

    if x0_ind == NX
        x0_ind -= 1.0
    end

    if v0_ind == NV
        v0_ind -= 1.0
    end

    for i = 1:N_TILES
        n_tile = (i-1) * NX * NV
        dx, dv = offsets[i, a+2]

        if x0_rem > dx
            ind_x = x0_ind + 1
        else
            if x0_ind == 0
                ind_x = NaN
            else
                ind_x = x0_ind
            end
        end

        if v0_rem > dv
            ind_v = v0_ind + 1
        else
            if v0_ind == 0
                ind_v = NaN
            else
                ind_v = v0_ind
            end
        end

        feat_ind[i] = n_tile + (ind_v - 1) * NX + ind_x
    end
end

function ϵ_greedy(Q, actions)
    # Return index of action
    if ϵ < rand()
        return actions[argmax(Q)]
    else
        return rand(actions)
    end
end

function calc_Q(w,x,v,a,offsets,feat_ind)
    """
    Q = wᵀF,
    where F is feature vector
    w is parameter vector.
    """

    Q = 0.0
    F(x,v,a,offsets,feat_ind)
    for i = 1:N_TILES
        if !isnan(feat_ind[i])
            f_i = Int(feat_ind[i])
            Q += w[f_i]
        end
    end
    return Q
end

function update_w(w, sᵢ, args...)
    """
    Gradient-descent Sarsa

    e : [N_FEATURES]
    w : [N_FEATURES]
    s : (x, v)
    a : -1 or 0 or 1

    indx : discretized x indicies (tile)
    indv : discretized v indicies (tile)

    main goal : update w vector.

    """
    Qₐ, e, feat_ind, actions, offsets = args

    x, v = sᵢ

    for a in actions  # For all possible actions at (x, v)
        a_ind = a + 2
        Qₐ[a_ind] = calc_Q(w,x,v,a,offsets,feat_ind)
    end

    a = ϵ_greedy(Qₐ, actions)

    e .= 0.0

    steps = 0
    ############## One episode ################
    while true
        if steps > MAXSTEPS
            break
        end

        F(x,v,a,offsets,feat_ind)

        w_sum = 0.0
        for i = 1:N_TILES
            if !isnan(feat_ind[i])
                f_i = Int(feat_ind[i])
                e[f_i] += 1
                w_sum += w[f_i]
            end
        end

        x′, v′, r = model_update(x, v, a)

        δ = r - w_sum

        if r == 0   # Reached terminal state (Q = 0)
            w .+= α * δ * e
            break
        end

        for act in actions  # For all possible actions at (x′, v′)
            a_ind = act + 2
            Qₐ[a_ind] = calc_Q(w,x′,v′,act,offsets,feat_ind)
        end

        a′ = ϵ_greedy(Qₐ, actions)  # Selecting action here (policy = "ϵ-greedy")

        # δ += γ * Qₐ[a′+2]           # Sarsa (on-policy)
        δ += γ * maximum(Qₐ)        # Q-Learning (off-policy)



        w .+= α * δ * e
        e .*= γ * λ

        x = x′
        v = v′
        a = a′

        steps += 1
    end
    ##########################################

    return steps
end

function run_episodes(Episodes, offsets)

    Qₐ, e, feat_ind, actions = init()

    args = (Qₐ, e, feat_ind, actions, offsets)

    w = zeros(N_FEATURES)

    # Q_temp = zeros(length(actions))
    # for act in actions
    #     a_ind = act + 2
    #     Q_temp[a_ind] = calc_Q(w,x,v,act,offsets,feat_ind)
    # end
    println("############################################")
    println("Gradient-descent Sarsa : Mountain Car")

    steps = []
    for episode = 1:Episodes

        x = (XMAX - XMIN) * rand() + XMIN
        # v = (VMAX - VMIN) * rand() + VMIN
        v = 0.0
        sᵢ = (x, v)

        step = update_w(w, sᵢ, args...)

        push!(steps, step)

        println("Episode $episode, Step = $step")
    end

    return w, steps
end

function init()
    actions = [-1, 0, 1]
    Qₐ = zeros(length(actions))     # Contains Q-values for specific action a (i.e. container).
    e = zeros(N_FEATURES)
    # offsets = [(rand()/NX, rand()/NV) for _ = 1:N_TILES, _ = 1:length(actions)]
    feat_ind = zeros(N_FEATURES)
    # return Qₐ, e, offsets, feat_ind, actions
    return Qₐ, e, feat_ind, actions

end

function Q_matrix(X, V, w, offsets)
    actions = [-1, 0, 1]
    feat_ind = zeros(N_FEATURES)
    Q = zeros(length(X), length(V))
    Q_temp = zeros(3)
    for i = 1:length(X)
        for j = 1:length(V)
            x = X[i]
            v = V[j]
            for k = 1:3
                a = actions[k]
                Q_temp[k] = calc_Q(w,x,v,a,offsets,feat_ind)
            end
            Q[i,j] = maximum(Q_temp)    # Greedy
        end
    end
    return Q
end

function simulate_car(w, offsets, maxsteps)
    actions = [-1, 0, 1]
    feat_ind = zeros(N_FEATURES)
    # x = (XMAX - XMIN) * rand() + XMIN
    # v = (VMAX - VMIN) * rand() + VMIN
    x = -0.5
    v = 0.0
    Q = zeros(length(actions))
    X = [x]
    V = [v]
    A = [0]
    step = 0
    while true
        for a_i = 1:3
            Q[a_i] = calc_Q(w,x,v,actions[a_i],offsets,feat_ind)
        end
        a = actions[argmax(Q)]      # Pick greedy actions.
        x′, v′, r = model_update(x, v, a)
        push!(X, x′)
        push!(V, v′)
        push!(A, a)

        x = x′
        v = v′

        if r == 0 || step > maxsteps
            break
        end
        step += 1
    end
    return X, V, A
end

# function make_scene(bgcolor, res, x_track, y_track, pos, act)
#     GLMakie.activate!()
#     scene = Scene(backgroundcolor = bgcolor, resolution=res)
#     colors = [:red, :black, :blue]
#     markers = ['←', '.', '→']
#     shiftx = 0.7
#     lines!(scene, x_track .- shiftx, y_track)
#     GLMakie.scatter!(scene, Point2f(x_track[end] - shiftx, y_track[end]),
#         markersize=30, color=:brown, marker=:diamond)
#     pos_obs = Observable(Point2f(pos[1,1] - shiftx, pos[1,2]))
#     color_obs = Observable(colors[act[1]+2])
#     time_obs = Observable('0')
#     marker_obs = Observable(markers[act[1]+2])
#     rotation_obs = Observable(atan(dir[1,2], dir[1,1]))
#     # meshscatter!(scene, pos_obs, markersize=0.03, color=color_obs)
#     GLMakie.scatter!(scene, pos_obs, markersize=0.15, color=color_obs,
#         marker = marker_obs, rotations=rotation_obs, linewidth=3)
#     text_obs = Observable("0")
#     GLMakie.text!(scene, text_obs, position = (-1,-1), textsize=40)
#
#     display(scene)
# end

# function animate_car()
#
# end
