
"""
    _try_fit(f, n, (a, b), scheme, mode; ...) -> (res, ok, why)

Attempt to fit a degree-`n` polynomial on `[a, b]` to within `target`.

Returns:
* `res` — the `OptimResult` (or a sentinel `nothing` if the inner driver
  threw and we could not produce one — `ok` is `false` in that case);
* `ok`  — `true` iff `res.total_error ≤ target`;
* `why` — a short string explaining a rejection (`"err > target"`, the
  exception type, …) — used only for `verbose` and error messages.

Inner driver exceptions (`ConvergenceFailure`, `ExchangeFailure`,
`DomainError` from the `RelativeMode` non-vanishing check, plus generic
numerical failures like `LinearAlgebra.SingularException`) are caught and
mapped to a rejection: bisection then takes care of it.
"""
function _try_fit(f, n::Int, ab::Tuple{T,T},
    scheme::EvalScheme{T}, mode::AbstractMode,
    cfg::FitConfig{TargetT,T}) where {TargetT<:AbstractFloat,T<:AbstractFloat}
    target = cfg.target
    τ = cfg.τ
    driver_max_iter = cfg.driver_max_iter
    strategy = cfg.strategy
    target_type = cfg.target_type

    res = try
        if mode isa AbsoluteMode
            eval_approx_optimize(f, n, ab, scheme;
                τ=τ, max_iter=driver_max_iter,
                strategy=strategy, verbose=false, target_type=target_type)
        else  # RelativeMode
            eval_approx_optimize_relative(f, n, ab, scheme;
                τ=τ, max_iter=driver_max_iter,
                strategy=strategy, verbose=false, target_type=target_type)
        end
    catch e
        if e isa InterruptException || e isa OutOfMemoryError
            rethrow()
        end
        return (nothing, false, string(typeof(e)))
    end

    if !isfinite(res.total_error)
        return (res, false, "non-finite total_error")
    end
    if res.total_error ≤ target
        return (res, true, "ok")
    end
    return (res, false, "err > target")
end


"""
    default_scheme_builder(d) -> EvalScheme

Default scheme builder used by the budget-aware drivers when the caller
does not supply one: `horner_scheme(d; u = 2.0^-53)` (Horner rule at
binary64 precision).
"""
default_scheme_builder(d::Integer, ::Type{T}=Float64) where T<:AbstractFloat =
    horner_scheme(Int(d), T; u=eps(T) / 2)
