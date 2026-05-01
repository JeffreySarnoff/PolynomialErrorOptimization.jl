# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

"""
    ApproxPiece

One subinterval `[a, b]` together with the optimal degree-`n` polynomial fit
on it and the verified error bound `total_error` (= `astar`).

# Fields
* `a::ComputeT`             — left endpoint.
* `b::ComputeT`             — right endpoint.
* `result::OptimResult{TargetT,ComputeT}` — full driver output for this piece.
"""
struct ApproxPiece{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    a::ComputeT
    b::ComputeT
    result::OptimResult{TargetT,ComputeT}
end

Base.show(io::IO, p::ApproxPiece) = print(io,
    "ApproxPiece([", p.a, ", ", p.b, "], err=", p.result.total_error, ")")

"""
    PiecewisePolyApprox

A piecewise polynomial approximation of a function on an interval.

# Fields
* `pieces::Vector{ApproxPiece}` — subintervals and their fits, in left-to-right order.
* `max_n::Int`                  — upper bound on the polynomial degree across
                                  pieces. For the fixed-degree drivers
                                  (`approximate_abs`, `approximate_rel`) this
                                  equals the common degree. For the
                                  coefficient-budget drivers it equals
                                  `max_coeffs - 1`.
* `mode::Symbol`                — `:abs` or `:rel`.
* `target::ComputeT`            — the requested error target.
* `worst_error::ComputeT`       — `maximum(p.result.total_error for p in pieces)`.

A `PiecewisePolyApprox` is callable:

    pa(t::Real)

evaluates the local polynomial at `t`. `t` must lie in the original interval
`[pieces[1].a, pieces[end].b]`; otherwise a `DomainError` is thrown.
"""
struct PiecewisePolyApprox{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    pieces::Vector{ApproxPiece{TargetT,ComputeT}}
    max_n::Int
    mode::Symbol
    target::ComputeT
    worst_error::ComputeT
end

function Base.show(io::IO, pa::PiecewisePolyApprox)
    a = pa.pieces[1].a
    b = pa.pieces[end].b
    print(io, "PiecewisePolyApprox(",
        "[", a, ", ", b, "]; ",
        "max_n=", pa.max_n, ", ",
        "mode=:", pa.mode, ", ",
        "pieces=", length(pa.pieces), ", ",
        "target=", pa.target, ", ",
        "worst_error=", pa.worst_error, ")")
end

"""
    _locate_piece(pa, t) -> Int

Binary-search the piece index containing `t`. Boundary points belong to the
left piece (the convention is irrelevant for continuous `f` since adjacent
fits agree to within the per-piece tolerance, but the rule must be fixed).
"""
function _locate_piece(pa::PiecewisePolyApprox{TargetT,ComputeT},
    t::ComputeT) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    pieces = pa.pieces
    a0 = pieces[1].a
    bN = pieces[end].b
    (t < a0 || t > bN) && throw(DomainError(t,
        "_locate_piece: t=$t outside approximation interval [$a0, $bN]"))
    lo, hi = 1, length(pieces)
    while lo < hi
        mid = (lo + hi) >>> 1
        if t <= pieces[mid].b
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo
end

(pa::PiecewisePolyApprox{TargetT,ComputeT})(t::Real) where {TargetT,ComputeT} =
    pa.pieces[_locate_piece(pa, ComputeT(t))].result.poly(ComputeT(t))
