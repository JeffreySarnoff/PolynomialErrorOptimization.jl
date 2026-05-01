# Internal: cost-minimising recursive driver (degree_policy = :min_cost)
# ---------------------------------------------------------------------------

"""
    _bisect_min_cost(fit_at_degree, I, mode; max_n, ...)
        -> PiecewisePolyApprox

Recursive variant of `_bisect` that minimises the *total* number of
polynomial coefficients across the partition.

At each subinterval `[a, b]` it computes:

* `cost_accept` — the smallest degree `d ∈ 0:max_n` for which
  `fit_at_degree(a, b, d)` succeeds, plus 1; or `+∞` if none does.
* `cost_bisect` — the recursive total cost of the best partition of
  `[a, b]` into halves. `+∞` if the recursion is forced to give up at
  this depth.

It returns the cheaper alternative. α–β pruning is used: the second half
is only computed if the first half's cost leaves room to beat
`cost_accept`.

`fit_at_degree(a, b, d)` must return the same `(res, ok, why)` triple as
`_try_fit`. The function does not need to know about `target`/`τ`/
`driver_max_iter` — those are baked into the closure.
"""
function _bisect_min_cost(fit_at_degree,
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

    # `cap` is the upper bound (inclusive) on the total cost we'll accept
    # from the remaining subtree. We use `typemax(Int)` as the "no cap"
    # sentinel — actual costs are bounded by 2^max_depth · (max_n + 1)
    # which is far smaller than typemax(Int).
    INF = typemax(Int)
    global_cap = total_coeffs > 0 ? total_coeffs : INF

    # Recursive worker. Returns (pieces::Vector{ApproxPiece}, cost::Int).
    # `cost == INF` signals "no feasible partition of [a, b] under cap".
    # The caller treats that as an alternative being unavailable.
    # Throws (with a helpful message) only when the *whole* call has no
    # feasible solution at all (both accept and bisect fail and we're at
    # the budget floor).
    function recurse(a::T, b::T, depth::Int, cap::Int)
        width = b - a

        # ---- (1) Find the cheapest accepting degree on [a, b] ----
        accept_res::Union{OptimResult{TargetT,T},Nothing} = nothing
        accept_cost::Int = INF
        last_why::String = "no degree attempted"
        for d in 0:max_n
            res, ok, why = fit_at_degree(a, b, d)
            if ok
                accept_res = res
                accept_cost = d + 1
                break
            end
            last_why = why
        end

        # ---- (2) Decide whether we can bisect ----
        mid = 0.5 * (a + b)
        can_bisect = depth < max_depth && width > 2 * min_width &&
                     mid > a && mid < b

        # ---- (3) Leaf case: must accept or fail ----
        if !can_bisect
            if accept_res === nothing
                error("approximate ($mode_label, :min_cost): could not reach " *
                      "target = $target on subinterval [$a, $b] within " *
                      "max_depth = $max_depth / min_width = $min_width " *
                      "(reason: $last_why).")
            end
            if accept_cost > cap
                return (ApproxPiece{TargetT,T}[], INF)
            end
            verbose && println("accept  [", a, ", ", b,
                "]  err = ", accept_res.total_error,
                "  depth = ", depth,
                "  cost = ", accept_cost)
            return (ApproxPiece{TargetT,T}[ApproxPiece{TargetT,T}(a, b, accept_res)], accept_cost)
        end

        # ---- (4) Compute α–β bound for the bisection alternative ----
        #
        # We accept (rather than bisect) when:
        #     accept_cost ≤ cap       (feasible) AND
        #     accept_cost ≤ bisect_cost  (no worse than bisecting; ties
        #                                 favour accept → fewer pieces).
        # So bisect can only win if  bisect_cost < accept_cost  and
        # bisect_cost ≤ cap. Combine:  bisect_cost ≤ min(cap, accept_cost - 1).
        #
        # Each half costs ≥ 1, so left+right ≥ 2; we can skip bisection
        # entirely when the bound is < 2.
        bisect_bound = min(cap, accept_cost == INF ? INF : accept_cost - 1)
        try_bisect = bisect_bound ≥ 2

        bisect_pieces = ApproxPiece{TargetT,T}[]
        bisect_cost = INF

        if try_bisect
            # Left half cap: left ≥ 1 and right ≥ 1, so left ≤ bound - 1.
            left_cap = bisect_bound - 1
            local left_pieces = []
            local cost_L = INF
            try
                left_pieces, cost_L = recurse(a, mid, depth + 1, left_cap)
            catch e
                if e isa InterruptException || e isa OutOfMemoryError
                    rethrow()
                end
                cost_L = INF
                left_pieces = ApproxPiece{TargetT,T}[]
            end

            if cost_L < INF && cost_L ≤ left_cap
                right_cap = bisect_bound - cost_L
                local right_pieces = []
                local cost_R = INF
                try
                    right_pieces, cost_R = recurse(mid, b, depth + 1, right_cap)
                catch e
                    if e isa InterruptException || e isa OutOfMemoryError
                        rethrow()
                    end
                    cost_R = INF
                    right_pieces = ApproxPiece{TargetT,T}[]
                end
                if cost_R < INF && (cost_L + cost_R) ≤ bisect_bound
                    bisect_cost = cost_L + cost_R
                    bisect_pieces = vcat(left_pieces, right_pieces)
                end
            end
        end

        # ---- (5) Pick the cheaper alternative within `cap` ----
        accept_ok = accept_res !== nothing && accept_cost ≤ cap
        bisect_ok = bisect_cost < INF && bisect_cost ≤ cap

        if accept_ok && (!bisect_ok || accept_cost ≤ bisect_cost)
            verbose && println("accept  [", a, ", ", b,
                "]  err = ", accept_res.total_error,
                "  depth = ", depth,
                "  cost = ", accept_cost,
                bisect_ok ? "  (vs bisect $(bisect_cost))" : "")
            return (ApproxPiece{TargetT,T}[ApproxPiece{TargetT,T}(a, b, accept_res)], accept_cost)
        end

        if bisect_ok
            verbose && println("bisect  [", a, ", ", b,
                "]  depth = ", depth,
                "  cost = ", bisect_cost,
                accept_ok ? "  (vs accept $(accept_cost))" :
                "  (accept infeasible)")
            return (bisect_pieces, bisect_cost)
        end

        # ---- (6) Neither alternative fits within the cap ----
        # If we have no accept_res at all and bisection failed too, the
        # subproblem is genuinely infeasible — error out. Otherwise we
        # bubble up INF so the parent can try a different decomposition.
        if accept_res === nothing && bisect_cost == INF
            error("approximate ($mode_label, :min_cost): could not reach " *
                  "target = $target on subinterval [$a, $b] within " *
                  "max_depth = $max_depth (reason: $last_why).")
        end
        return (ApproxPiece{TargetT,T}[], INF)
    end

    pieces, cost = recurse(I[1], I[2], 0, global_cap)
    if cost == INF || isempty(pieces)
        error("approximate ($mode_label, :min_cost): could not satisfy total " *
              "coefficient cap total_coeffs = $total_coeffs on the entire " *
              "interval $I (best feasible cost exceeds cap).")
    end

    worst = maximum(p.result.total_error for p in pieces)
    return PiecewisePolyApprox{TargetT,T}(pieces, max_n, _mode_symbol(mode), target, worst)
end
