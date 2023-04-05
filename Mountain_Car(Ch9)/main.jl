##
"""
Mountain car problem

10-dimensional parameter vector, w.

Using linear function approximator;

    i.e., q(x,v,a,w) = wᵀF(x,v,a)

    F(x,v,a) is a feature vector of same size as w.

Using tiling method to determine the 'features'.

There are 10 tiles (9x9), and for a given state (x, v),

    one can determine the value of the feature by looking at the

    value of the tile at position (s, a).


"""

using Plots
using GLMakie, Makie
const XMAX = 0.5
const XMIN = -1.2
const VMAX = 0.07
const VMIN = -0.07
const NX = 9
const NV = 9
const N_TILES = 10
const N_FEATURES = NX * NV * N_TILES
const MAXSTEPS = 10000

const λ = 0.9
const α = 0.05
const γ = 1.0       # γ = 1.0 is important (e.g. γ=0.9 does not produce same result...)
const ϵ = 0.0       # ϵ = 0.0 is also important... (i.e. no exploration).

##
include("functions.jl")

offsets = [(rand()/NX, rand()/NV) for _ = 1:N_TILES, _ = 1:3]
# offsets = repeat([(rand()/NX, rand()/NV) for _ = 1:N_TILES], 1, 3)

w, steps = run_episodes(100, offsets)
Plots.plot(steps)

##
X = LinRange(XMIN, XMAX, 2000)
V = LinRange(VMIN, VMAX, 2000)
Q_mat = Q_matrix(X, V, w, offsets)
Plots.heatmap(Q_mat, size=(550,500))

## Track : Convert to x-y axis
N = length(X)
ΔX = (XMAX-XMIN) / N
δy = ΔX * 0.8 * cos.(3 * X)
δx = .√(ΔX^2 .- δy .^2)
y_track = cumsum(δy)
x_track = cumsum(δx)
Plots.plot(x_track, y_track,legend=false)

## Get matrix of car positions in x-y coord.
convert_ind(p, X) = argmin(abs.(p .- X))
p, vel, act = simulate_car(w, offsets, 1000)
T = length(p)
pos = zeros(T, 2)
dir = zeros(T, 2)
# vel = zeros(T, 2)
for i = 1:T
    ind = convert_ind(p[i], X)
    pos[i,1] = x_track[ind]
    pos[i,2] = y_track[ind]
    dir[i,1] = δx[ind] / ΔX
    dir[i,2] = δy[ind] / ΔX
end

## Makie 'scene'
GLMakie.activate!()
scene = Scene(backgroundcolor = :white, resolution=(600, 600))
# fig = Figure()
# scene = Axis(fig[1,1])
colors = [:red, :black, :blue]
markers = ['←', '.', '→']
shiftx = 0.7
lines!(scene, x_track .- shiftx, y_track)
# meshscatter!(scene, Point2f(x_track[end] - 0.5, y_track[end]), markersize=0.02, color=:black)
GLMakie.scatter!(scene, Point2f(x_track[end] - shiftx, y_track[end]),
    markersize=30, color=:brown, marker=:diamond)
# x = Observable(pos[1, 1])
# y = Observable(pos[1, 2])
pos_obs = Observable(Point2f(pos[1,1] - shiftx, pos[1,2]))
color_obs = Observable(colors[act[1]+2])
time_obs = Observable('0')
marker_obs = Observable(markers[act[1]+2])
rotation_obs = Observable(atan(dir[1,2], dir[1,1]))
# meshscatter!(scene, pos_obs, markersize=0.03, color=color_obs)
GLMakie.scatter!(scene, pos_obs, markersize=0.15, color=color_obs,
    marker = marker_obs, rotations=rotation_obs)
text_obs = Observable("0")
GLMakie.text!(scene, text_obs, position = (-1,-1), textsize=40)


# Δl = 0.05
# shift = (-0.9, 0.5)
# lw_obs = Observable(0.05)
# lw_table = [lw_obs for i=1:NX*NV*N_TILES, a_i=1:3]
# for tile = 1:N_TILES
#     for a_i = 1:3
#         for i = 0:NX-1
#             for j = 0:NV-1
#                 ind = (tile-1)*NX*NV + i*NX+j + 1
#                 x_rect = i * Δl + offsets[tile,a_i][1] * NX * Δl + shift[1]
#                 y_rect = j * Δl + offsets[tile,a_i][2] * NV * Δl + shift[2]
#                 lines!(scene, Rect(x_rect, y_rect ,Δl, Δl), linewidth=lw_table[ind,a_i])
#             end
#         end
#     end
# end

display(scene)

## Run animation
feat_ind = zeros(N_TILES)
for t = 1:T
    a = act[t]
    a_ind = a + 2
    pos_obs[] = Point2f(pos[t,1] - shiftx, pos[t,2])
    color_obs[] = colors[a_ind]
    # if a != 0
    #     marker_obs[] = markers[a_ind]
    # end
    marker_obs[] = markers[a_ind]
    rotation_obs[] = atan(dir[t,2], dir[t,1])
    text_obs[] = string(t)

    # F(p[t], vel[t], a, offsets, feat_ind)
    # for ind in feat_ind
    #     if !isnan(ind)
    #         lw_table[Int(ind),a_ind][] = 1.0
    #     end
    # end

    sleep(0.05)

    # Back to original
    # for ind in feat_ind
    #     if !isnan(ind)
    #         lw_table[Int(ind), a_ind][] = 0.05
    #     end
    # end
end

##
timeseq = 1:T
record(scene, "car_100_epi.gif", timeseq; framerate = 30) do t
    a = act[t]
    a_ind = a + 2
    pos_obs[] = Point2f(pos[t,1] - shiftx, pos[t,2])
    color_obs[] = colors[a_ind]
    # if a != 0
    #     marker_obs[] = markers[a_ind]
    # end
    marker_obs[] = markers[a_ind]
    rotation_obs[] = atan(dir[t,2], dir[t,1])
    text_obs[] = string(t)
end

##
