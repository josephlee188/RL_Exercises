## Monte Carlo Methods
# Black Jack game
using StatsBase

function close_to_n(n, a, b)
    # Choose a or b closer to n but not exceed n.
    if n < a && n < b
        return a > b ? b : a    # If both exceed n, return smaller one.
    elseif n < a && n >= b
        return b
    elseif n >= a && n < b
        return a
    else
        return a > b ? a : b
    end
end

function pick_card(cards, deck, usable_ace)
    new_card = rand(deck)
    if new_card == 0
        new_card = close_to_n(21, sum(cards) + 1, sum(cards) + 11) - sum(cards)
    end

    # if new_card == 11
    #     usable_ace = true
    # else
    #     usable_ace = false
    # end


    push!(cards, new_card)
    # return usable_ace
    return new_card
end

function BlackJack(Π, Πₐ, deck, deck_ace)
    # Run an episode of BlackJack.
    # Π = Policy : vector of hit or stick from sum = 12 ~ 21
    # Πₐ = Policy with 'usable' Ace

    is_player_bust = false
    is_dealer_bust = false
    is_player_natural = false
    is_dealer_natural = false

    player = []
    dealer = []
    actions = []

    push!(player, rand(deck_ace))
    push!(player, rand(deck_ace))

    if sum(player) > 21   # if both are Aces
        player[1] = 1
    end
    # if 11 in player
    #     usable_ace = true
    # end
    push!(dealer, rand(deck_ace))

    ###### Dealer has fixed strategy (stick if sum is 17 or higher) ######
    while !is_dealer_bust
        if sum(dealer) > 21
            is_dealer_bust = true
            break;
        elseif sum(dealer) < 17
            _ = pick_card(dealer, deck)
        else
            break;
        end
    end

    dealer_sum = sum(dealer)

    # usable_ace = 11 in player
    # usable_ace = false
    usable_ace = false

    while !is_player_bust
        if sum(player) > 21
            is_player_bust = true
            # push!(actions, true)
            break
        elseif sum(player) < 12
            _ = pick_card(player, deck)
        else    # sum = 12 ~ 21. Follow policy.
            if sum(player) == 21    # Black Jack
                push!(actions, false)
                break
            end

            if (11 in player)
                hit = Πₐ[sum(player), dealer[1]]
            else
                hit = Π[sum(player), dealer[1]]
            end

            push!(actions, hit)

            if hit
                new_card = pick_card(player, deck)
                # usable_ace = usable_ace | (new_card == 11)
            else    # stick
                break;
            end
        end
    end

    player_sum = sum(player)

    if is_player_bust
        winner = "dealer"
    else
        if is_dealer_bust
            winner = "player"
        elseif player_sum == dealer_sum
            winner = "draw"
        elseif player_sum > dealer_sum
            winner = "player"
        else
            winner = "dealer"
        end
    end

    # if is_player_bust && is_dealer_bust
    #     winner = "draw"
    # elseif is_player_bust && !is_dealer_bust
    #     winner = "dealer"
    # elseif !is_player_bust && is_dealer_bust
    #     winner = "player"
    # elseif player_sum > dealer_sum
    #     winner = "player"
    # elseif player_sum < dealer_sum
    #     winner = "dealer"
    # else
    #     winner = "draw"
    # end

    reward = 0
    if winner == "dealer"
        reward = -1
    elseif winner == "player"
        reward = 1
    end

    # return player, dealer, winner

    return player, dealer, actions, reward, usable_ace
end

##
# avg(vec) = isempty(vec) ? (0, 0.0) : (length(vec), mean(vec))

# function sum_over(n, vec)
#     m = vec[1] + vec[2]
#     i = 3
#     while m < n
#         m += vec[i]
#         i += 1
#     end
#     return m
# end

#########################################################################################
function mc_control(Episodes, Π, Πₐ, args...)
    deck, deck_ace, ϵ = args

    V = Dict((n, m) => 0.0 for n=12:21, m=2:11)
    Vₐ = Dict((n, m) => 0.0 for n=12:21, m=2:11)
    # G = Dict((n, m)=>[] for n=12:21, m=2:11)
    # Gₐ = Dict((n, m)=>[] for n=12:21, m=2:11)

    Q = Dict((n, m, k) => 0.0 for n=12:21, m=2:11, k=[false,true])
    Qₐ = Dict((n, m, k) => 0.0 for n=12:21, m=2:11, k=[false,true])

    # G = Dict((n, m, k) => [] for n=12:21, m=2:11, k=[false,true])
    # Gₐ = Dict((n, m, k) => [] for n=12:21, m=2:11, k=[false,true])

    g = Dict((n, m, k) => 0.0 for n=12:21, m=2:11, k=[false,true])
    gₐ = Dict((n, m, k) => 0.0 for n=12:21, m=2:11, k=[false,true])
    N = Dict((n, m, k) => 0.0 for n=12:21, m=2:11, k=[false,true])
    Nₐ = Dict((n, m, k) => 0.0 for n=12:21, m=2:11, k=[false,true])

    for epi = 1:Episodes
        #### One 'episode'. ####
        # Run episode
        player, dealer, actions, reward, usable_ace = BlackJack(Π, Πₐ, deck, deck_ace)

        dealer_state = dealer[1]

        ################### Evaluation ###################
        # 'Busted' action always has 1 less element (because loop stops)
        player_sum = 0

        # First start MC
        for card in player
            player_sum += card
            if player_sum >= 12
                break
            end
        end

        player_state = player_sum

        action = actions[1]
        has_ace = (11 in player)
        # has_ace = usable_ace[1]
        # has_ace = usable_ace[end]

        if !has_ace
            g_val = g[player_state, dealer_state, action] += reward
            N_val = N[player_state, dealer_state, action] += 1.0
            g_val_0 = g[player_state, dealer_state, false]
            g_val_1 = g[player_state, dealer_state, true]
            N_val_0 = N[player_state, dealer_state, false]
            N_val_1 = N[player_state, dealer_state, true]

            Q[player_state, dealer_state, action] = g_val / N_val
            V[player_state, dealer_state] = (g_val_0 + g_val_1) / (N_val_0 + N_val_1)

            if ϵ < rand()
                Π[player_state, dealer_state] = g_val_0 / N_val_0 < g_val_1 / N_val_1
            else
                Π[player_state, dealer_state] = rand([true, false])
            end
        else
            g_val = gₐ[player_state, dealer_state, action] += reward
            N_val = Nₐ[player_state, dealer_state, action] += 1.0
            g_val_0 = gₐ[player_state, dealer_state, false]
            g_val_1 = gₐ[player_state, dealer_state, true]
            N_val_0 = Nₐ[player_state, dealer_state, false]
            N_val_1 = Nₐ[player_state, dealer_state, true]

            Qₐ[player_state, dealer_state, action] = g_val / N_val
            Vₐ[player_state, dealer_state] = (g_val_0 + g_val_1) / (N_val_0 + N_val_1)

            if ϵ < rand()
                Πₐ[player_state, dealer_state] = g_val_0 / N_val_0 < g_val_1 / N_val_1
            else
                Πₐ[player_state, dealer_state] = rand([true, false])
            end
        end


        # Every start MC
        # do_update = false
        # player_state = 0
        # action = false
        # action_ind = 1
        #
        # for card in player
        #     player_sum += card
        #     if player_sum > 11 && player_sum < 22
        #         player_state = player_sum
        #         action = actions[action_ind]
        #         action_ind += 1
        #         do_update = true
        #     # elseif player_sum > 21  # If got busted
        #     #     player_state = player_sum - card
        #     #     action = true
        #     end
        #
        #     if do_update
        #         if !usable_ace
        #             g_val = g[player_state, dealer_state, action] += reward
        #             N_val = N[player_state, dealer_state, action] += 1.0
        #             # g_val_! = g[player_state, dealer_state, !action] += 0.0
        #             # N_val_! = N[player_state, dealer_state, !action] += 1.0
        #             g_val_0 = g[player_state, dealer_state, false]
        #             g_val_1 = g[player_state, dealer_state, true]
        #             N_val_0 = N[player_state, dealer_state, false]
        #             N_val_1 = N[player_state, dealer_state, true]
        #
        #             Q[player_state, dealer_state, action] = g_val / N_val
        #             V[player_state, dealer_state] = (g_val_0 + g_val_1) / (N_val_0 + N_val_1)
        #
        #             # Q[player_state, dealer_state, !action] = g_val_! / N_val_!
        #
        #             # if g_val_0 / N_val_0 == g_val_1 / N_val_1
        #             #     Π[player_state, dealer_state] = rand([false, true])
        #             # elseif g_val_0 / N_val_0 < g_val_1 / N_val_1
        #             #     Π[player_state, dealer_state] = true
        #             # elseif g_val_0 / N_val_0 > g_val_1 / N_val_1
        #             #     Π[player_state, dealer_state] = false
        #             # end
        #
        #             if g_val_0 / N_val_0 < g_val_1 / N_val_1
        #                 Π[player_state, dealer_state] = true
        #             elseif g_val_0 / N_val_0 > g_val_1 / N_val_1
        #                 Π[player_state, dealer_state] = false
        #             else
        #                 Π[player_state, dealer_state] = false
        #             end
        #
        #         else
        #             g_val = gₐ[player_state, dealer_state, action] += reward
        #             N_val = Nₐ[player_state, dealer_state, action] += 1.0
        #             g_val_0 = gₐ[player_state, dealer_state, false]
        #             g_val_1 = gₐ[player_state, dealer_state, true]
        #             N_val_0 = Nₐ[player_state, dealer_state, false]
        #             N_val_1 = Nₐ[player_state, dealer_state, true]
        #
        #             Qₐ[player_state, dealer_state, action] = g_val / N_val
        #             Vₐ[player_state, dealer_state] = (g_val_0 + g_val_1) / (N_val_0 + N_val_1)
        #
        #             # if g_val_0 / N_val_0 == g_val_1 / N_val_1
        #             #     Πₐ[player_state, dealer_state] = rand([false, true])
        #             # elseif g_val_0 / N_val_0 < g_val_1 / N_val_1
        #             #     Πₐ[player_state, dealer_state] = true
        #             # elseif g_val_0 / N_val_0 > g_val_1 / N_val_1
        #             #     Πₐ[player_state, dealer_state] = false
        #             # end
        #
        #             if g_val_0 / N_val_0 < g_val_1 / N_val_1
        #                 Πₐ[player_state, dealer_state] = true
        #             elseif g_val_0 / N_val_0 > g_val_1 / N_val_1
        #                 Πₐ[player_state, dealer_state] = false
        #             end
        #         end
        #     end
        # end

        ################### Control ###################

        if epi % 100000 == 0
            println("Episode : $(epi)")
        end
    end

    return V, Vₐ, Q, Qₐ
    # return Q, Qₐ
end

function initialize()
    Π = Dict([(n, m) => n < 20 for n in 12:21, m in 2:11])
    Πₐ = Dict([(n, m) => n < 20 for n in 12:21, m in 2:11])    # Policy with usable Ace

    # Π = Dict([(n, m) => rand([true,false]) for n in 12:21, m in 2:11])
    # Πₐ = Dict([(n, m) => rand([true,false]) for n in 12:21, m in 2:11])    # Policy with usable Ace

    deck = [0, collect(2:9)..., 10, 10, 10, 10]         # There are four 10s, one Ace, each
    deck_ace = [collect(2:9)..., 11, 10, 10, 10, 10]     # 0 is usable Ace

    return Π, Πₐ, deck, deck_ace
end
