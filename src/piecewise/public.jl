# ===========================================================================
# Unified entry point
# ===========================================================================

"""
    approximate(f, I; target, mode = :abs, ...) -> PiecewisePolyApprox

Single entry point to the four piecewise-approximation drivers
(`approximate_abs`, `approximate_rel`, `approximate_abs_budget`,
`approximate_rel_budget`).

The choice of driver is determined by which of `n` or `max_coeffs` is
specified (exactly one is required) and by `mode`:

| `mode`  | `n` given                  | `max_coeffs` given                   |
| :------ | :------------------------- | :----------------------------------- |
| `:abs`  | `approximate_abs`          | `approximate_abs_budget`             |
| `:rel`  | `approximate_rel`          | `approximate_rel_budget`             |

# Required keyword arguments
- `target`              : error target (`> 0`).
- exactly one of:
  - `n::Integer`        : fixed polynomial degree (uses the fixed-degree
                          driver). When `n` is given, `scheme::EvalScheme`
                          is also required (no degree-specific default).
  - `max_coeffs::Integer`: per-piece coefficient cap (uses the
                          budget-aware driver). `scheme_builder` is
                          optional and defaults to
                          `default_scheme_builder`.

# Common optional keyword arguments
- `mode::Symbol`        : `:abs` or `:rel`. Default `:abs`.
- `target_type`         : floating-point type for returned polynomial
                          coefficients. For budget mode, this also sets the
                          default finite-precision error model.
- `compute_type`        : floating-point type used internally by budget mode
                          when building default schemes. Fixed-degree mode
                          takes its computation type from `scheme`.
- `τ`, `max_depth`, `min_width`, `total_coeffs`, `driver_max_iter`,
  `verbose` — forwarded to the chosen driver.

# Optional keyword arguments specific to the budget drivers
- `scheme_builder`      : `d -> EvalScheme`. Default `default_scheme_builder`.
- `degree_policy`       : `:max`, `:min`, or `:min_cost`. Default `:max`.

# Optional keyword arguments specific to the fixed-degree drivers
- `scheme::EvalScheme`  : required when `n` is given. The driver checks
                          `scheme.n == n`.
- `strategy`            : forwarded as `strategy = ...`. For the
                          fixed-degree drivers `default_strategy(scheme)`
                          is used if omitted; for the budget drivers
                          `nothing` (per-degree default) is used.

# Examples

```julia
# Fixed degree, absolute error, no budget cap.
scheme = horner_scheme(4; u = 2.0^-53)
pa = approximate(sin, (-3.0, 3.0); target = 1e-8, n = 4, scheme = scheme)

# Per-piece coefficient budget, relative error, choose smallest fit.
pa = approximate(exp, (0.5, 2.5);
                 target = 1e-10, mode = :rel,
                 max_coeffs = 6, degree_policy = :min)

# Globally-cost-aware partition with both per-piece and total caps.
pa = approximate(t -> sin(3t), (-2.0, 2.0);
                 target = 1e-6,
                 max_coeffs = 8, degree_policy = :min_cost,
                 total_coeffs = 64)
```
"""
function approximate(f, I::Tuple{<:Real,<:Real};
    target::Real,
    mode::Symbol=:abs,
    n::Union{Integer,Nothing}=nothing,
    max_coeffs::Union{Integer,Nothing}=nothing,
    scheme::Union{EvalScheme,Nothing}=nothing,
    scheme_builder=default_scheme_builder,
    compute_type::Type{<:AbstractFloat}=Float64,
    target_type::Union{Type{<:AbstractFloat},Nothing}=nothing,
    degree_policy::Symbol=:max,
    τ::Real=1e-3,
    max_depth::Integer=30,
    min_width::Real=0.0,
    total_coeffs::Integer=0,
    driver_max_iter::Integer=100,
    strategy=nothing,
    verbose::Bool=false)

    # Exactly one of n / max_coeffs.
    nspec, ncs = n !== nothing, max_coeffs !== nothing
    if !(nspec ⊻ ncs)
        throw(ArgumentError(
            "approximate: pass exactly one of `n` or `max_coeffs`; " *
            "got n=$(n), max_coeffs=$(max_coeffs)."))
    end
    mode_tag = _mode_from_symbol(mode)

    if nspec
        # Fixed-degree branch.
        @argcheck scheme !== nothing ArgumentError(
            "approximate: when `n` is given, `scheme::EvalScheme` is required " *
            "(no degree-specific default exists).")
        tt = target_type === nothing ? fptype(scheme) : target_type
        # `strategy` for fixed-degree drivers defaults to
        # default_strategy(scheme) if the caller didn't override.
        strat = strategy === nothing ? default_strategy(scheme) : strategy
        if mode_tag isa AbsoluteMode
            return approximate_abs(f, n, I, scheme;
                target=target,
                τ=τ,
                max_depth=max_depth,
                min_width=min_width,
                total_coeffs=total_coeffs,
                driver_max_iter=driver_max_iter,
                strategy=strat,
                target_type=tt,
                verbose=verbose)
        else
            return approximate_rel(f, n, I, scheme;
                target=target,
                τ=τ,
                max_depth=max_depth,
                min_width=min_width,
                total_coeffs=total_coeffs,
                driver_max_iter=driver_max_iter,
                strategy=strat,
                target_type=tt,
                verbose=verbose)
        end
    else
        # Budget branch.
        @argcheck scheme === nothing ArgumentError(
            "approximate: pass `scheme_builder`, not `scheme`, when " *
            "`max_coeffs` is given (each degree needs its own scheme).")
        tt = target_type === nothing ? compute_type : target_type
        if mode_tag isa AbsoluteMode
            return approximate_abs_budget(f, max_coeffs, I;
                target=target,
                scheme_builder=scheme_builder,
                compute_type=compute_type,
                target_type=tt,
                degree_policy=degree_policy,
                τ=τ,
                max_depth=max_depth,
                min_width=min_width,
                total_coeffs=total_coeffs,
                driver_max_iter=driver_max_iter,
                strategy=strategy,
                verbose=verbose)
        else
            return approximate_rel_budget(f, max_coeffs, I;
                target=target,
                scheme_builder=scheme_builder,
                compute_type=compute_type,
                target_type=tt,
                degree_policy=degree_policy,
                τ=τ,
                max_depth=max_depth,
                min_width=min_width,
                total_coeffs=total_coeffs,
                driver_max_iter=driver_max_iter,
                strategy=strategy,
                verbose=verbose)
        end
    end
end
