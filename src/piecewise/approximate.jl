"""
    piecewise/approximate.jl

Adaptive piecewise-polynomial approximation built on top of the fixed-degree
absolute/relative-error drivers.

This file is the include hub for the piecewise subsystem. The implementation
is split by responsibility so public API wrappers, result types, fitting,
bisection, and cost-minimizing search can evolve independently.
"""

include("types.jl")
include("config.jl")
include("fitting.jl")
include("bisect.jl")
include("min_cost.jl")
include("fixed.jl")
include("budget.jl")
include("public.jl")
