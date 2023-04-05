include("functions.jl")
using Plots
const WIDTH = 10
const HEIGHT = 7
const ϵ = 0.1   # ϵ-soft control
const α = 0.5   # Learning rate
const γ = 1.0   # Discount rate
const λ = 0.9   # Eligibility coefficient

use_eligibility_traces = false

nsteps, path = run_episodes(1000, use_eligibility_traces)

##
gridworld = zeros(Int, HEIGHT, WIDTH)
for (i, pos) in enumerate(path)
    y, x = pos
    gridworld[y, x] = i + 4
end

println(sum(nsteps) / length(nsteps))

# Draw
plot(layout=(2,1), size=(450,600))

plot!(1:length(nsteps), nsteps, subplot=1, xlabel="episode", ylabel="steps")

heatmap!(gridworld, subplot=2)

##
