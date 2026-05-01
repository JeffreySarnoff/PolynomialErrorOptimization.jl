"""
    search.jl

Search-strategy types and maximisation routines used by `find_new_index`.
"""

# ---------------------------------------------------------------------------
# Search strategies for find_new_index
# ---------------------------------------------------------------------------

"""
    SearchStrategy

Abstract supertype for strategies that locate the maximum-violation point
`tstar ∈ I` inside `find_new_index`.
"""
abstract type SearchStrategy end

"""
    GridSearch(M::Int)

Coarse equispaced grid of `M` points on `I`. Picks the maximiser among the
grid points. Cheap, robust default.
"""
struct GridSearch <: SearchStrategy
    M::Int
    function GridSearch(M::Integer)
        @argcheck M ≥ 2 ArgumentError("M must be ≥ 2, got $M")
        return new(Int(M))
    end
end

"""
    GridThenLocal(M::Int; bracket = 3)

Coarse grid of size `M`, then a few iterations of golden-section search
inside the `±bracket` neighbouring grid cells around the coarse maximum.
Use this when high accuracy on the new index is desired.
"""
struct GridThenLocal <: SearchStrategy
    M::Int
    bracket::Int
    function GridThenLocal(M::Integer; bracket::Integer=3)
        @argcheck M ≥ 2 ArgumentError("M must be ≥ 2, got $M")
        @argcheck bracket ≥ 1 ArgumentError("bracket must be ≥ 1, got $bracket")
        return new(Int(M), Int(bracket))
    end
end

"""
    GridThenOptim(M::Int; bracket = 3)

Coarse grid of size `M`, then bounded local refinement with `Optim.Brent()`
on the `±bracket` neighbouring grid cells around the coarse maximum.
This often gives sharper `tstar`/`astar` than pure grid search at similar
sampling cost.
"""
struct GridThenOptim <: SearchStrategy
    M::Int
    bracket::Int
    function GridThenOptim(M::Integer; bracket::Integer=3)
        @argcheck M ≥ 2 ArgumentError("M must be ≥ 2, got $M")
        @argcheck bracket ≥ 1 ArgumentError("bracket must be ≥ 1, got $bracket")
        return new(Int(M), Int(bracket))
    end
end

"""
    locate_maximum(g, I, strategy::SearchStrategy) -> (tstar, value⋆)

Maximise `g : T → T` on the closed interval `I = (tl, tr)` using the given
strategy. Returns the maximiser and the maximum value.
"""
function locate_maximum(g, I::Tuple{T,T}, strategy::GridSearch) where T<:AbstractFloat
    best_t, best_v, _ = _coarse_grid_max(g, I, strategy.M)
    return best_t, best_v
end

function locate_maximum(g, I::Tuple{T,T}, strategy::GridThenLocal) where T<:AbstractFloat
    tl, tr = I
    M = strategy.M
    bracket = strategy.bracket

    best_t, best_v, best_k = _coarse_grid_max(g, I, M)

    # Local refinement: golden-section on the ±bracket window.
    klo = max(0, best_k - bracket)
    khi = min(M - 1, best_k + bracket)
    a = tl + (tr - tl) * (T(klo) / T(M - 1))
    b = tl + (tr - tl) * (T(khi) / T(M - 1))
    t_loc, v_loc = _golden_section_max(g, a, b; tol=T(1e-12), max_iter=200)
    if v_loc > best_v
        best_t = t_loc
        best_v = v_loc
    end
    return best_t, best_v
end

function locate_maximum(g, I::Tuple{T,T}, strategy::GridThenOptim) where T<:AbstractFloat
    tl, tr = I
    M = strategy.M
    bracket = strategy.bracket

    best_t, best_v, best_k = _coarse_grid_max(g, I, M)

    # Local refinement with bounded Brent on a bracket around the best cell.
    klo = max(0, best_k - bracket)
    khi = min(M - 1, best_k + bracket)
    a = tl + (tr - tl) * (T(klo) / T(M - 1))
    b = tl + (tr - tl) * (T(khi) / T(M - 1))

    if a < b
        h(t) = -g(T(t))
        result = Optim.optimize(h, a, b, Optim.Brent())
        t_loc = T(Optim.minimizer(result))
        v_loc = -T(Optim.minimum(result))
        if v_loc > best_v
            best_t = t_loc
            best_v = v_loc
        end
    end

    return best_t, best_v
end

function _coarse_grid_max(g, I::Tuple{T,T}, M::Int) where T<:AbstractFloat
    tl, tr = I
    best_k = 0
    best_t = tl
    best_v = g(tl)
    @inbounds for k in 1:(M-1)
        t = tl + (tr - tl) * (T(k) / T(M - 1))
        v = g(t)
        if v > best_v
            best_k = k
            best_t = t
            best_v = v
        end
    end
    return best_t, best_v, best_k
end

# Standard golden-section search for the maximum of a unimodal-on-bracket g.
function _golden_section_max(g, a::T, b::T; tol::Real=1e-12,
    max_iter::Int=200) where T<:AbstractFloat
    φ = (sqrt(T(5)) - one(T)) / T(2)      # ≈ 0.618
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fc = g(c)
    fd = g(d)
    iter = 0
    while abs(b - a) > tol * (one(T) + abs(a) + abs(b)) && iter < max_iter
        if fc > fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a)
            fc = g(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a)
            fd = g(d)
        end
        iter += 1
    end
    if fc > fd
        return c, fc
    else
        return d, fd
    end
end

"""
    default_strategy(scheme::EvalScheme) -> GridSearch

Default search strategy used by `find_new_index`. Currently a `GridSearch`
with `M = max(2048, 64*(n+2))`.
"""
default_strategy(scheme::EvalScheme) =
    GridSearch(max(2048, 64 * (scheme.n + 2)))
