# ===========================================================================
# Coefficient-budget variants
# ===========================================================================
#
# `approximate_abs_budget` and `approximate_rel_budget` enforce *two*
# constraints simultaneously: the per-piece error target AND a hard upper
# bound `max_coeffs` on the number of polynomial coefficients per piece
# (i.e. degree ≤ `max_coeffs - 1`). Pieces that cannot meet the error
# target inside that degree budget are bisected.
#
# The caller supplies a `scheme_builder::Function` mapping a candidate
# degree `d` to an `EvalScheme` of degree `d`. This is necessary because
# evaluation schemes are degree-specific. A sensible default is provided:
# `default_scheme_builder` returns `horner_scheme(d; u = 2.0^-53)`.
#
# Two strategies are offered for choosing the per-piece degree:
#
#   degree_policy = :max
#       Always fit at degree `max_coeffs - 1`. Simplest; fewest pieces.
#
#   degree_policy = :min
#       Try degrees 0, 1, 2, …, `max_coeffs - 1` in order; accept the first
#       that meets the error target. Cheaper polynomials on easy pieces,
#       but does up to `max_coeffs` driver calls per accepted piece.

"""
    approximate_abs_budget(f, max_coeffs, I;
                           target,
                           scheme_builder = nothing,
                           compute_type   = Float64,
                           target_type    = compute_type,
                           degree_policy  = :max,
                           τ              = 1e-3,
                           max_depth      = 30,
                           min_width      = 0.0,
                           total_coeffs   = 0,
                           driver_max_iter = 100,
                           strategy       = nothing,
                           verbose        = false) -> PiecewisePolyApprox

Adaptively subdivide `I = (a, b)` until each piece admits a polynomial of
**at most `max_coeffs` coefficients** (degree `≤ max_coeffs - 1`) whose
verified absolute-error bound is `≤ target`.

# Arguments
- `f`            : the function to approximate. `f(t::T) -> Real`, where
                   `T` is `compute_type`.
- `max_coeffs`   : maximum number of coefficients per piece. Must be `≥ 1`,
                   so the per-piece degree is at most `max_coeffs - 1`.
- `I`            : `(tl, tr)` with `tl < tr`.

# Keyword arguments
- `target`         : absolute-error target (required, `> 0`).
- `scheme_builder` : `d -> EvalScheme` of degree `d`. Default: Horner at
                     `u = eps(target_type)/2` represented in `compute_type`.
                     For `degree_policy = :max` only the
                     single degree `max_coeffs - 1` is requested; for
                     `:min` and `:min_cost` any degree in
                     `0 : max_coeffs - 1` may be requested.
- `compute_type`    : floating-point type used for internal nodes, solves,
                     searches, and verified error bounds when using the
                     default scheme builder.
- `target_type`     : floating-point type for returned polynomial
                     coefficients and the default finite-precision error
                     model.
- `degree_policy`  : `:max`, `:min`, or `:min_cost`.
                     - `:max` (default) fits every piece at degree
                       `max_coeffs - 1`. Simplest; fewest driver calls.
                     - `:min` tries degrees `0, 1, …, max_coeffs - 1` in
                       order and accepts the first that meets the target.
                       Locally greedy: minimises per-piece cost given the
                       current partition.
                     - `:min_cost` is a globally-aware version: at each
                       piece it computes both `cost_accept` (smallest `d`
                       that fits, +1) and `cost_bisect` (recursive cost of
                       partitioning the piece). Picks whichever is
                       cheaper, with α–β-style pruning.
- `τ`              : per-piece convergence tolerance for the inner driver.
- `max_depth`      : maximum bisection depth.
- `min_width`      : minimum allowed piece width (`0.0` = unconstrained).
- `total_coeffs`   : optional cap on the **sum** of coefficient counts
                     across all accepted pieces. `0` (default) disables.
                     For `:max` this caps the number of pieces at
                     `floor(total_coeffs / max_coeffs)`. For `:min` and
                     `:min_cost` per-piece cost varies, so the cap acts
                     directly on the cumulative count. The check is eager:
                     accepting a piece whose addition would exceed the cap
                     raises `ErrorException`. With `:min_cost` the
                     cumulative cost is also used as the α–β bound, so the
                     cap can prune the search tree.
- `driver_max_iter`: `max_iter` forwarded to the inner driver.
- `strategy`       : `SearchStrategy` for `find_new_index`. If `nothing`
                     (the default), the inner driver's
                     `default_strategy(scheme)` is used per scheme.
- `verbose`        : per-piece logging.

# Returns
A `PiecewisePolyApprox`. The `max_n` field equals `max_coeffs - 1`.
Individual pieces may have a *lower* degree under `degree_policy = :min`
or `:min_cost`; their effective degree is
`length(Polynomials.coeffs(piece.result.poly)) - 1`.

# Errors
- `ArgumentError` for malformed inputs (including `max_coeffs < 1`,
  `degree_policy ∉ (:max, :min, :min_cost)`, or a `scheme_builder(d)`
  whose `scheme.n ≠ d`).
- `ErrorException` if a subinterval cannot be brought below `target`
  within both the depth/width and the coefficient-count budgets, or if the
  total coefficient cap is exceeded.
"""
function approximate_abs_budget(f, max_coeffs::Integer,
    I::Tuple{<:Real,<:Real};
    target::Real,
    scheme_builder=nothing,
    compute_type::Type{<:AbstractFloat}=Float64,
    target_type::Type{<:AbstractFloat}=compute_type,
    degree_policy::Symbol=:max,
    τ::Real=1e-3,
    max_depth::Integer=30,
    min_width::Real=0.0,
    total_coeffs::Integer=0,
    driver_max_iter::Integer=100,
    strategy=nothing,
    verbose::Bool=false)
    T = compute_type
    builder = scheme_builder === nothing ? d -> default_scheme_builder(d, T) : scheme_builder
    cfg = BudgetApproxConfig(AbsoluteMode(), T(target), builder,
        degree_policy, T(τ), Int(max_depth), T(min_width),
        Int(total_coeffs), Int(driver_max_iter), strategy, target_type,
        verbose)
    return _approximate_budget(f, Int(max_coeffs),
        (T(I[1]), T(I[2])), cfg)
end

"""
    approximate_rel_budget(f, max_coeffs, I; ...)
        -> PiecewisePolyApprox

Relative-error counterpart of `approximate_abs_budget`. Each piece's fit
satisfies, in the verified sense of Theorem 2,

    max_{t ∈ piece} (|f(t) − p(t)| + θ(a, t)) / |f(t)|  ≤  target,

and uses at most `max_coeffs` polynomial coefficients. `f` must not
vanish on `I`. As in `approximate_rel`, an interior zero of `f` is
handled by repeated bisection until the zero sits on a piece boundary
(or the budgets are exhausted).

All keyword arguments and error semantics are the same as
`approximate_abs_budget`.
"""
function approximate_rel_budget(f, max_coeffs::Integer,
    I::Tuple{<:Real,<:Real};
    target::Real,
    scheme_builder=nothing,
    compute_type::Type{<:AbstractFloat}=Float64,
    target_type::Type{<:AbstractFloat}=compute_type,
    degree_policy::Symbol=:max,
    τ::Real=1e-3,
    max_depth::Integer=30,
    min_width::Real=0.0,
    total_coeffs::Integer=0,
    driver_max_iter::Integer=100,
    strategy=nothing,
    verbose::Bool=false)
    T = compute_type
    builder = scheme_builder === nothing ? d -> default_scheme_builder(d, T) : scheme_builder
    cfg = BudgetApproxConfig(RelativeMode(), T(target), builder,
        degree_policy, T(τ), Int(max_depth), T(min_width),
        Int(total_coeffs), Int(driver_max_iter), strategy, target_type,
        verbose)
    return _approximate_budget(f, Int(max_coeffs),
        (T(I[1]), T(I[2])), cfg)
end

# ---------------------------------------------------------------------------
# Internal: shared budget-aware driver
# ---------------------------------------------------------------------------

function _approximate_budget(f, max_coeffs::Int,
    I::Tuple{T,T},
    cfg::BudgetApproxConfig{TargetT,T}) where {TargetT<:AbstractFloat,T<:AbstractFloat}
    mode = cfg.mode
    target = cfg.target
    scheme_builder = cfg.scheme_builder
    degree_policy = cfg.degree_policy
    τ = cfg.τ
    max_depth = cfg.max_depth
    min_width = cfg.min_width
    total_coeffs = cfg.total_coeffs
    driver_max_iter = cfg.driver_max_iter
    strategy = cfg.strategy
    target_type = cfg.target_type
    verbose = cfg.verbose

    # ---- input validation specific to budget mode ----
    @argcheck max_coeffs ≥ 1 ArgumentError(
        "max_coeffs must be ≥ 1, got $max_coeffs")

    max_n = max_coeffs - 1
    bisect_cfg = BisectConfig(mode, max_n, target, τ, max_depth,
        min_width, total_coeffs, target_type, verbose)

    # Cache schemes by degree to avoid rebuilding on every piece.
    scheme_cache = Dict{Int,EvalScheme{T}}()
    function _get_scheme(d::Int)
        get!(scheme_cache, d) do
            sc = scheme_builder(d)
            @argcheck sc isa EvalScheme ArgumentError(
                "scheme_builder($d) must return an EvalScheme, got $(typeof(sc))")
            @argcheck fptype(sc) === T ArgumentError(
                "scheme_builder($d) returned EvalScheme{$(fptype(sc))}; expected EvalScheme{$T}")
            @argcheck sc.n == d ArgumentError(
                "scheme_builder($d).n = $(sc.n) ≠ requested degree $d")
            sc
        end
    end

    # Per-degree fitter used by all three policies.
    fit_at_degree = function (a, b, d)
        sc = _get_scheme(d)
        strat = strategy === nothing ? default_strategy(sc) : strategy
        fit_cfg = FitConfig(target, τ, driver_max_iter, strat, target_type)
        return _try_fit(f, d, (a, b), sc, mode, fit_cfg)
    end

    if degree_policy === :max || degree_policy === :min
        # Per-piece fitter that honours `degree_policy`. The fitter tries
        # one degree (`:max`) or scans 0..max_n (`:min`); on success it
        # returns the OptimResult; on exhaustion it returns the *last*
        # attempt's result so the bisection logger has something
        # informative.
        fitter = function (a, b)
            last_res = nothing
            last_report = _fit_attempt_report((a, b), max_n, mode, target;
                accepted=false,
                kind=:no_degree_attempted,
                message="no degree attempted")
            degrees = degree_policy === :max ? (max_n:max_n) : (0:max_n)
            for d in degrees
                res, ok, report = fit_at_degree(a, b, d)
                if ok
                    verbose && degree_policy === :min && println(
                            "        accepted at degree d = ", d)
                    return (res, true, report)
                end
                last_res, last_report = res, report
            end
            return (last_res, false, last_report)
        end

        return _bisect(fitter, I, bisect_cfg)
    else
        # :min_cost — global cost minimisation via recursive search.
        return _bisect_min_cost(fit_at_degree, I, bisect_cfg)
    end
end
