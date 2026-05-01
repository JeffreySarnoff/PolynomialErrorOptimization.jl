"""
    _bisect(fitter, I, mode; max_n, target, τ, max_depth, min_width,
            total_coeffs, verbose) -> PiecewisePolyApprox

Generic adaptive-bisection loop. `fitter(a, b)` must return a 3-tuple
`(res, ok, why)` matching the contract of `_try_fit`:

* `res::OptimResult` (or `nothing` if the inner driver threw),
* `ok::Bool` — `true` iff this piece is accepted,
* `why::AbstractString` — short rejection reason for `verbose`/error messages.

The fitter is the only mode-specific part; everything else (budget checks,
left-to-right ordering, worst-error aggregation) lives here.

If `total_coeffs > 0` the loop tracks the cumulative coefficient count
across accepted pieces (using `length(Polynomials.coeffs(res.poly))` per
piece) and raises `ErrorException` if accepting the next piece would push
the cumulative total strictly above `total_coeffs`. A bisection cannot
reduce already-accepted cost, so the failure is reported eagerly rather
than reverted. Pass `total_coeffs = 0` (the default in callers) to disable
the cap.
"""
function _bisect(fitter,
    I::Tuple{T,T},
    cfg::BisectConfig{TargetT,T}) where {TargetT<:AbstractFloat,T<:AbstractFloat}
    @argcheck I[1] < I[2] ArgumentError("I = $I must satisfy I[1] < I[2]")

    mode = cfg.mode
    max_n = cfg.max_n
    target = cfg.target
    max_depth = cfg.max_depth
    min_width = cfg.min_width
    total_coeffs = cfg.total_coeffs
    verbose = cfg.verbose
    mode_label = _mode_label(mode)

    accepted = Vector{ApproxPiece{TargetT,T}}()
    stack = PendingInterval{T}[PendingInterval{T}(I[1], I[2], 0)]
    running = 0

    while !isempty(stack)
        cur = pop!(stack)
        a, b, depth = cur.a, cur.b, cur.depth
        width = b - a

        res, ok, why = fitter(a, b)

        if ok
            piece_cost = length(Polynomials.coeffs(res.poly))
            if total_coeffs > 0 && running + piece_cost > total_coeffs
                error("approximate ($mode_label): total coefficient budget " *
                      "exceeded — accepting piece [$a, $b] (cost $piece_cost) " *
                      "would push the running total from $running to " *
                      "$(running + piece_cost), above total_coeffs = " *
                      "$total_coeffs.")
            end
            running += piece_cost
            verbose && println("accept  [", a, ", ", b,
                "]  err = ", res.total_error,
                "  depth = ", depth,
                "  coeffs = ", piece_cost,
                "  running = ", running)
            push!(accepted, ApproxPiece{TargetT,T}(a, b, res))
            continue
        end

        # Need to bisect. Check budgets.
        if depth ≥ max_depth
            error("approximate ($mode_label): could not reach target = $target on " *
                  "subinterval [$a, $b] within max_depth = $max_depth " *
                  "(reason: $why).")
        end
        if width ≤ 2 * min_width
            error("approximate ($mode_label): could not reach target = $target on " *
                  "subinterval [$a, $b]; bisection would produce pieces " *
                  "narrower than min_width = $min_width (reason: $why).")
        end

        verbose && println("bisect  [", a, ", ", b,
            "]  reason = ", why,
            "  depth = ", depth)

        mid = (a + b) / 2
        if mid ≤ a || mid ≥ b
            error("approximate ($mode_label): could not reach target = $target on " *
                  "subinterval [$a, $b]; midpoint is not representable " *
                  "(reason: $why).")
        end

        push!(stack, PendingInterval{T}(mid, b, depth + 1))
        push!(stack, PendingInterval{T}(a, mid, depth + 1))
    end

    worst = maximum(p.result.total_error for p in accepted)
    return PiecewisePolyApprox{TargetT,T}(accepted, max_n, _mode_symbol(mode), target, worst)
end
