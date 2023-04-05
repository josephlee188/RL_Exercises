## Monte Carlo Methods
# Black Jack game
include("functions.jl")
using Plots
Π, Πₐ, deck, deck_ace = initialize()

## #################################################
player, dealer, actions, reward, usable_ace = BlackJack(Π, Πₐ, deck, deck_ace)

println("Player Ace : $(usable_ace)")
println("Actions : " * join(string.(actions), ", "))
println("Player sum : $(sum(player)) -> " * join(string.(player), ", "))
println("Dealer sum : $(sum(dealer)) -> " * join(string.(dealer), ", "))
println("Reward : $reward \n")

## Count number ####################################
dealer_n = 0
player_n = 0
N = 10000
for n = 1:N
    player, dealer, actions, reward, has_player_ace = BlackJack(Π, Πₐ, deck, deck_ace)
    if reward == -1
        dealer_n += 1
    elseif reward == 1
        player_n += 1
    end
end

println("Dealer winning rate : $(dealer_n/N*100)%")
println("Player winning rate : $(player_n/N*100)% \n")

## ###############################################
include("functions.jl")
using Plots
ϵ = 0.05

Π, Πₐ, deck, deck_ace = initialize()

args = (deck, deck_ace, ϵ)

Episodes = 200_000_000

# do_control = true

V, Vₐ, Q, Qₐ = mc_control(Episodes, Π, Πₐ, args...)

V_mat = zeros(10, 10)
Vₐ_mat = zeros(10, 10)
Q_mat = zeros(10, 10, 2)
Qₐ_mat = zeros(10, 10, 2)
policy = zeros(Int, 10, 10)
policyₐ = zeros(Int, 10, 10)

for (s1, s2) = collect(keys(V))
    if s2 == 11
        s2a = 1
    else
        s2a = s2
    end
    V_mat[s1-11, s2a] = V[s1,s2]
    Vₐ_mat[s1-11, s2a] = Vₐ[s1,s2]
    policy[s1-11, s2a] = Π[s1,s2]
    policyₐ[s1-11, s2a] = Πₐ[s1,s2]
    Q_mat[s1-11, s2a, 1] = Q[s1,s2,false]
    Q_mat[s1-11, s2a, 2] = Q[s1,s2,true]
end

plot(layout=(2,2))

heatmap!(V_mat, size=(600,500), subplot=1, title="V (No usable Ace)")

heatmap!(policy, size=(600,500), subplot=2, title="π (No usable Ace)")

heatmap!(Vₐ_mat, size=(600,500), subplot=3, title="Vₐ (Usable Ace)")

heatmap!(policyₐ, size=(600,500), subplot=4, title="πₐ (Usable Ace)")



##
