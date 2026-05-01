# ---------------------------------------------------------------------------
# Public drivers
# ---------------------------------------------------------------------------

"""
    approximate_abs(f, n, I, scheme;
                    target,
                    τ            = 1e-3,
                    max_depth    = 30,
                    min_width    = 0.0,
                    total_coeffs = 0,
                    driver_max_iter = 100,
                    strategy     = default_strategy(scheme),
                    verbose      = false) -> PiecewisePolyApprox

Adaptively subdivide `I = (a, b)` until each piece admits a degree-`n`
polynomial fit whose verified absolute-error bound `total_error` (Theorem 2
of the paper) is `≤ target`.

The bound is on `|f(t) − p(t)| + θ(a, t)` — i.e. the same quantity
optimised by `eval_approx_optimize`. Since `total_error` is *guaranteed* to
be ≥ `ε⋆`, an accepted piece is accepted soundly: there is no chance of a
false pass.

# Arguments
- `f, n, I, scheme` : as in `eval_approx_optimize`.

# Keyword arguments
- `target`          : the absolute-error target. Required, must be > 0.
- `τ`               : per-piece convergence tolerance for the inner driver.
                      Looser `τ` means fewer driver iterations per piece but
                      may slightly over-estimate the per-piece error and
                      cause unnecessary subdivisions. Default `1e-3`.
- `max_depth`       : maximum bisection depth; `2^max_depth` is the cap on
                      the number of pieces. A piece that is rejected at
                      `max_depth` raises an error.
- `min_width`       : refuse to bisect a piece narrower than this. `0.0`
                      means rely on `max_depth` only.
- `total_coeffs`    : optional cap on the **sum** of coefficient counts
                      across all accepted pieces. `0` (the default)
                      disables the cap. With fixed degree `n` every piece
                      contributes `n+1` coefficients, so this is equivalent
                      to capping the number of pieces at
                      `floor(total_coeffs / (n+1))`.
- `driver_max_iter` : `max_iter` forwarded to the inner driver.
- `strategy`        : `SearchStrategy` forwarded to the inner driver.
- `target_type`     : floating-point type for returned polynomial
                      coefficients. Internal computation uses `fptype(scheme)`.
- `verbose`         : print one line per accepted/rejected piece.

# Errors
Throws `ArgumentError` for malformed inputs, and an `ErrorException` if a
subinterval cannot be brought below `target` within the depth/width budget,
or if the cumulative coefficient count would exceed `total_coeffs`.
"""
function approximate_abs(f, n::Integer,
    I::Tuple{<:Real,<:Real},
    scheme::EvalScheme{T};
    target::Real,
    τ::Real=1e-3,
    max_depth::Integer=30,
    min_width::Real=0.0,
    total_coeffs::Integer=0,
    driver_max_iter::Integer=100,
    strategy::SearchStrategy=default_strategy(scheme),
    target_type::Type{<:AbstractFloat}=T,
    verbose::Bool=false) where T<:AbstractFloat
    cfg = FixedApproxConfig(AbsoluteMode(), T(target), T(τ), Int(max_depth),
        T(min_width), Int(total_coeffs), Int(driver_max_iter),
        strategy, target_type, verbose)
    return _approximate(f, Int(n), (T(I[1]), T(I[2])), scheme, cfg)
end

"""
    approximate_rel(f, n, I, scheme;
                    target,
                    τ            = 1e-3,
                    max_depth    = 30,
                    min_width    = 0.0,
                    driver_max_iter = 100,
                    strategy     = default_strategy(scheme),
                    verbose      = false) -> PiecewisePolyApprox

Adaptively subdivide `I` so that each piece admits a degree-`n` polynomial
fit whose verified relative-error bound is `≤ target`. Internally calls
`eval_approx_optimize_relative` on each candidate piece.

`f` must not vanish on `I`. If `f` does vanish at an interior point, the
inner driver raises `DomainError`; this routine treats that as an
unacceptable piece and bisects it (eventually moving the zero to a
subinterval boundary, where the heuristic vanishing-check no longer
triggers). If a piece narrower than `min_width` (or beyond `max_depth`)
still contains a zero of `f`, an error is raised. Use
`eval_approx_optimize_relative_zero` directly if `f` has a known zero.

Other arguments behave as in `approximate_abs`.
"""
function approximate_rel(f, n::Integer,
    I::Tuple{<:Real,<:Real},
    scheme::EvalScheme{T};
    target::Real,
    τ::Real=1e-3,
    max_depth::Integer=30,
    min_width::Real=0.0,
    total_coeffs::Integer=0,
    driver_max_iter::Integer=100,
    strategy::SearchStrategy=default_strategy(scheme),
    target_type::Type{<:AbstractFloat}=T,
    verbose::Bool=false) where T<:AbstractFloat
    cfg = FixedApproxConfig(RelativeMode(), T(target), T(τ), Int(max_depth),
        T(min_width), Int(total_coeffs), Int(driver_max_iter),
        strategy, target_type, verbose)
    return _approximate(f, Int(n), (T(I[1]), T(I[2])), scheme, cfg)
end

function _approximate(f, n::Int,
    I::Tuple{T,T},
    scheme::EvalScheme{T},
    cfg::FixedApproxConfig{TargetT,T}) where {TargetT<:AbstractFloat,T<:AbstractFloat}
    # ---- input validation specific to fixed-degree drivers ----
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    @argcheck scheme.n == n ArgumentError(
        "scheme.n ($(scheme.n)) ≠ n ($n)")

    mode = cfg.mode
    fit_cfg = FitConfig(cfg.target, cfg.τ, cfg.driver_max_iter, cfg.strategy,
        cfg.target_type)
    bisect_cfg = BisectConfig(mode, n, cfg.target, cfg.τ, cfg.max_depth,
        cfg.min_width, cfg.total_coeffs, cfg.target_type, cfg.verbose)

    # Fitter: closes over scheme and forwards to _try_fit.
    fitter = (a, b) -> _try_fit(f, n, (a, b), scheme, mode, fit_cfg)

    return _bisect(fitter, I, bisect_cfg)
end

