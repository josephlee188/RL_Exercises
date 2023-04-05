## Jack's car rental exercise (Chapter 4 - 4.3). Dynamic programming.
# p(s',r | s, a) --> Transition probability (environments)
# Π(iₐ,iₛ) --> Policy (\Pi). π(a|s) --> Given a state 's', probability of choosing action 'a'
# V(s) --> Value-function

## Define the system
const γ = 0.9           # Discount coefficient
const θ = 1e-6
const S = 20
const A = 5     # Maximum cars to be transferred
using Plots
const ϵ = 0.01
include("functions2.jl")
P(n, λ) = λ^n / factorial(n) * exp(-λ)      # Poisson probability :  Σₙ P(n,λ) = 1

p = P.(0:20, 3)
p1 = norm(p[p .> ϵ])
p2 = norm(p[p .> ϵ])
p = P.(0:20, 4)
p3 = norm(p[p .> ϵ])
p = P.(0:20, 2)
p4 = norm(p[p .> ϵ])

probs = Dict()

args = (p1, p2, p3, p4)

##
V, Q, Π = initialize()

# error = error_evalutation_test(V, Q, Π)
# plot(error)

# Policy = policy_iteration(V, Q, Π, args...)
Policies = policy_iteration(V, Q, Π, args...)

for i = 1:5
    heatmap(0:20, 0:20, Policies[i], size=(550,500), xlabel="n₂", ylabel="n₁")
    png("Day$i.png")
end


heatmap(0:20, 0:20, zeros(Int,21,21), size=(550,500), clims=(-4,5), xlabel="n₂", ylabel="n₁")

heatmap(0:20, 0:20, V, size=(550,500), xlabel="n₂", ylabel="n₁")

##
# plot(0)
# for i = 1:21
#     plot!(-5:5, Q[:,i,14], marker=:o, leg=false)
# end
# plot!(0)
#
# policy_evaluation(V, Q, Π)
#
# is_stable = policy_improvement(V, Q, Π)
#
# Policy = optimal_actions(Π)
#
# heatmap(Policy)

##
