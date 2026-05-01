"""
    interface.jl

High-level convenience interface built on top of the fixed-degree and
piecewise expert APIs.
"""

const _FIT_MODES = (:abs, :rel)
const _FIT_EFFORTS = (:fast, :balanced, :accurate)
const _FIT_SCHEMES = (:horner, :horner_fma, :estrin, :estrin_fma)
const _AUTO_STRATEGY = :auto

"""
    ObjectiveSpec(mode, target)

Typed objective description for a high-level fit plan.
"""
struct ObjectiveSpec{T<:AbstractFloat}
    mode::Symbol
    target::T
    function ObjectiveSpec(mode::Symbol, target::T) where {T<:AbstractFloat}
        @argcheck mode in _FIT_MODES ArgumentError(
            "mode must be :abs or :rel, got :$mode")
        @argcheck target > zero(T) ArgumentError(
            "target must be > 0, got $target")
        return new{T}(mode, target)
    end
end

ObjectiveSpec(mode::Symbol, target::Real) = ObjectiveSpec(mode, float(target))

"""
    ComplexitySpec(; degree=nothing, max_coeffs=nothing, total_coeffs=0,
                     piecewise=:auto, degree_policy=nothing)

Typed complexity and partitioning description for a high-level fit plan.
"""
struct ComplexitySpec
    degree::Union{Int,Nothing}
    max_coeffs::Union{Int,Nothing}
    total_coeffs::Int
    piecewise::Union{Bool,Symbol}
    degree_policy::Union{Symbol,Nothing}
end

function ComplexitySpec(; degree::Union{Integer,Nothing}=nothing,
    max_coeffs::Union{Integer,Nothing}=nothing,
    total_coeffs::Integer=0,
    piecewise::Union{Bool,Symbol}=:auto,
    degree_policy::Union{Symbol,Nothing}=nothing)
    _validate_complexity_spec_inputs(
        degree, max_coeffs, total_coeffs, piecewise, degree_policy)
    return ComplexitySpec(
        degree === nothing ? nothing : Int(degree),
        max_coeffs === nothing ? nothing : Int(max_coeffs),
        Int(total_coeffs),
        piecewise,
        degree_policy)
end

"""
    PrecisionSpec(; target_type=Float64, compute_type=target_type)

Typed precision description for a high-level fit plan.
"""
struct PrecisionSpec{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    target_type::Type{TargetT}
    compute_type::Type{ComputeT}
end

function PrecisionSpec(; target_type::Type{TargetT}=Float64,
    compute_type::Type{ComputeT}=target_type) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    return PrecisionSpec{TargetT,ComputeT}(target_type, compute_type)
end

"""
    SearchSpec([compute_type=Float64]; scheme=:horner, effort=:balanced,
               τ=nothing, max_depth=nothing, min_width=nothing,
               driver_max_iter=nothing, strategy=nothing)

Typed search and scheme description for a high-level fit plan.
"""
struct SearchSpec{T<:AbstractFloat}
    scheme::Symbol
    effort::Symbol
    τ::Union{T,Nothing}
    max_depth::Union{Int,Nothing}
    min_width::Union{T,Nothing}
    driver_max_iter::Union{Int,Nothing}
    strategy::Union{SearchStrategy,Nothing}
end

SearchSpec(; kwargs...) = SearchSpec(Float64; kwargs...)

function SearchSpec(::Type{T}; scheme::Symbol=:horner,
    effort::Symbol=:balanced,
    τ::Union{Real,Nothing}=nothing,
    max_depth::Union{Integer,Nothing}=nothing,
    min_width::Union{Real,Nothing}=nothing,
    driver_max_iter::Union{Integer,Nothing}=nothing,
    strategy::Union{SearchStrategy,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_search_spec_inputs(
        scheme, effort, τ, max_depth, min_width, driver_max_iter)
    return SearchSpec{T}(
        scheme,
        effort,
        τ === nothing ? nothing : T(τ),
        max_depth === nothing ? nothing : Int(max_depth),
        min_width === nothing ? nothing : T(min_width),
        driver_max_iter === nothing ? nothing : Int(driver_max_iter),
        strategy)
end

"""
    FitPlan

Structured plan produced by `plan_fit`. It captures the typed problem
specification, the executable `FitParameters`, and which choices were inferred
from defaults rather than explicitly requested.
"""
struct FitPlan{TargetT<:AbstractFloat,ComputeT<:AbstractFloat,InferredT<:NamedTuple}
    interval::Tuple{ComputeT,ComputeT}
    objective::ObjectiveSpec{ComputeT}
    complexity::ComplexitySpec
    precision::PrecisionSpec{TargetT,ComputeT}
    search::SearchSpec{ComputeT}
    parameters
    inferred::InferredT
end

function Base.show(io::IO, plan::FitPlan)
    print(io, "FitPlan(",
        "mode=:", plan.objective.mode,
        ", target=", plan.objective.target,
        ", interval=", plan.interval,
        ", scheme=:", plan.search.scheme,
        ", effort=:", plan.search.effort,
        ", piecewise=", plan.complexity.piecewise,
        ")")
end

"""
    FitParameters

Concrete parameter bundle returned by `recommend_parameters` and accepted by
`approxfit`/`fit`. It records the choices that map a high-level request onto
the expert APIs.
"""
struct FitParameters{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    target::ComputeT
    mode::Symbol
    degree::Union{Int,Nothing}
    max_coeffs::Union{Int,Nothing}
    total_coeffs::Int
    piecewise::Union{Bool,Symbol}
    scheme::Symbol
    target_type::Type{TargetT}
    compute_type::Type{ComputeT}
    eval_scheme::Symbol
    eval_op::Symbol
    effort::Symbol
    τ::ComputeT
    max_depth::Int
    min_width::ComputeT
    driver_max_iter::Int
    strategy::Union{SearchStrategy,Nothing}
    degree_policy::Symbol
end

function Base.show(io::IO, p::FitParameters)
    print(io, "FitParameters(",
        "target=", p.target,
        ", mode=:", p.mode,
        p.degree === nothing ? "" : ", degree=$(p.degree)",
        p.max_coeffs === nothing ? "" : ", max_coeffs=$(p.max_coeffs)",
        p.total_coeffs == 0 ? "" : ", total_coeffs=$(p.total_coeffs)",
        ", piecewise=", p.piecewise,
        ", scheme=:", p.scheme,
        ", target_type=", p.target_type,
        ", compute_type=", p.compute_type,
        ", eval_scheme=:", p.eval_scheme,
        ", eval_op=:", p.eval_op,
        ", effort=:", p.effort,
        ", τ=", p.τ,
        ", max_depth=", p.max_depth,
        ", driver_max_iter=", p.driver_max_iter,
        ", degree_policy=:", p.degree_policy,
        ")")
end

"""
    Approximation

Result wrapper returned by `approxfit`. The wrapped `model` is either an
`OptimResult` for one global polynomial or a `PiecewisePolyApprox` for a
piecewise approximation. `Approximation` is callable and forwards evaluation
to the wrapped model.
"""
struct Approximation{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    model
    parameters::FitParameters{TargetT,ComputeT}
    interval::Tuple{ComputeT,ComputeT}
end

(a::Approximation)(t::Real) =
    is_piecewise(a) ? a.model(t) : a.model.poly(t)

Base.show(io::IO, a::Approximation) = print(io,
    "Approximation(",
    is_piecewise(a) ? "piecewise" : "single",
    ", mode=:", a.parameters.mode,
    ", target=", a.parameters.target,
    ", error_bound=", error_bound(a),
    ", coeff_count=", coeff_count(a), ")")

"""
    recommend_parameters(f, I; target=nothing, abs_tol=nothing, rel_tol=nothing,
                         mode=:abs, effort=:balanced, degree=nothing,
                         max_coeffs=nothing, total_coeffs=0, piecewise=:auto,
                         scheme=:horner, target_type=Float64,
                         compute_type=target_type)

Choose a coherent parameter bundle for `approxfit`.

Minimal input is `f`, `I`, and one tolerance:

* `target`: objective-space target. For `mode=:abs`, this is an absolute
  error target; for `mode=:rel`, this is a relative target.
* `abs_tol`: alias for an absolute target.
* `rel_tol`: relative tolerance. With `mode=:rel`, it is used directly as
  `target`. With `mode=:abs`, it is converted to `target = rel_tol * scale`,
  where `scale` is sampled from `f` on `I`.

If neither `degree` nor `max_coeffs` is provided, the helper chooses a
piecewise coefficient budget from `effort`, with small adjustments based on
sampled variation of `f` on `I`.

`target_type` is the type of returned polynomial coefficients and the
floating-point format modeled by the default Horner/Estrin error scheme
(`u = eps(target_type)/2`). `compute_type` is the type used internally for
sampling, exchange nodes, dense solves, searches, and verified error bounds.
"""
function recommend_parameters(f, I::Tuple{<:Real,<:Real};
    target::Union{Real,Nothing}=nothing,
    abs_tol::Union{Real,Nothing}=nothing,
    rel_tol::Union{Real,Nothing}=nothing,
    mode::Symbol=:abs,
    effort::Symbol=:balanced,
    degree::Union{Integer,Nothing}=nothing,
    max_coeffs::Union{Integer,Nothing}=nothing,
    total_coeffs::Integer=0,
    piecewise::Union{Bool,Symbol}=:auto,
    scheme::Symbol=:horner,
    target_type::Type{<:AbstractFloat}=Float64,
    compute_type::Type{<:AbstractFloat}=target_type,
    τ::Union{Real,Nothing}=nothing,
    max_depth::Union{Integer,Nothing}=nothing,
    min_width::Union{Real,Nothing}=nothing,
    driver_max_iter::Union{Integer,Nothing}=nothing,
    strategy=nothing,
    degree_policy::Union{Symbol,Nothing}=nothing)

    _validate_recommendation_inputs(
        I, mode, effort, piecewise, scheme, degree, max_coeffs, total_coeffs)

    profile = _sample_profile(f, I, compute_type)
    if mode === :rel && profile.crosses_zero
        throw(DomainError(I,
            "relative mode requires f to stay away from zero on sampled points; " *
            "use mode=:abs or a relative-zero expert driver for known zeros"))
    end

    target_value = _target_from_tolerances(
        mode, target, abs_tol, rel_tol, profile)
    defaults = _effort_defaults(effort)

    chosen_degree, chosen_max_coeffs, chosen_piecewise =
        _choose_degree_budget(degree, max_coeffs, piecewise, defaults, profile)
    chosen_max_depth = _choose_max_depth(max_depth, defaults, effort, profile)

    return FitParameters(
        compute_type(target_value),
        mode,
        chosen_degree,
        chosen_max_coeffs,
        Int(total_coeffs),
        chosen_piecewise,
        scheme,
        target_type,
        compute_type,
        _eval_scheme_for_scheme(scheme),
        _eval_op_for_scheme(scheme),
        effort,
        compute_type(τ === nothing ? defaults.τ : τ),
        chosen_max_depth,
        compute_type(min_width === nothing ? defaults.min_width : min_width),
        Int(driver_max_iter === nothing ? defaults.driver_max_iter : driver_max_iter),
        strategy === _AUTO_STRATEGY ? nothing : strategy,
        degree_policy === nothing ? defaults.degree_policy : degree_policy)
end

"""
    plan_fit(f, I; kwargs...) -> FitPlan
    plan_fit(f, I, objective::ObjectiveSpec; complexity=ComplexitySpec(),
             precision=PrecisionSpec(), search=nothing) -> FitPlan

Build a structured high-level fit plan without executing the solver.
"""
function plan_fit(f, I::Tuple{<:Real,<:Real};
    target::Union{Real,Nothing}=nothing,
    abs_tol::Union{Real,Nothing}=nothing,
    rel_tol::Union{Real,Nothing}=nothing,
    mode::Symbol=:abs,
    effort::Symbol=:balanced,
    degree::Union{Integer,Nothing}=nothing,
    max_coeffs::Union{Integer,Nothing}=nothing,
    total_coeffs::Integer=0,
    piecewise::Union{Bool,Symbol}=:auto,
    scheme::Symbol=:horner,
    target_type::Type{<:AbstractFloat}=Float64,
    compute_type::Type{<:AbstractFloat}=target_type,
    τ::Union{Real,Nothing}=nothing,
    max_depth::Union{Integer,Nothing}=nothing,
    min_width::Union{Real,Nothing}=nothing,
    driver_max_iter::Union{Integer,Nothing}=nothing,
    strategy=nothing,
    degree_policy::Union{Symbol,Nothing}=nothing)

    params = recommend_parameters(f, I;
        target=target, abs_tol=abs_tol, rel_tol=rel_tol,
        mode=mode, effort=effort, degree=degree,
        max_coeffs=max_coeffs, total_coeffs=total_coeffs,
        piecewise=piecewise, scheme=scheme,
        target_type=target_type, compute_type=compute_type,
        τ=τ, max_depth=max_depth, min_width=min_width,
        driver_max_iter=driver_max_iter, strategy=strategy,
        degree_policy=degree_policy)

    inferred = (
        degree=degree === nothing,
        max_coeffs=max_coeffs === nothing,
        piecewise=piecewise === :auto,
        τ=τ === nothing,
        max_depth=max_depth === nothing,
        min_width=min_width === nothing,
        driver_max_iter=driver_max_iter === nothing,
        strategy=strategy === nothing || strategy === _AUTO_STRATEGY,
        degree_policy=degree_policy === nothing)
    return _fit_plan(I, params, inferred)
end

function plan_fit(f, I::Tuple{<:Real,<:Real}, objective::ObjectiveSpec;
    complexity::ComplexitySpec=ComplexitySpec(),
    precision::PrecisionSpec=PrecisionSpec(),
    search::Union{SearchSpec,Nothing}=nothing)
    search_spec = search === nothing ? SearchSpec(precision.compute_type) :
                  _convert_search_spec(search, precision.compute_type)
    params = _parameters_from_specs(
        f, I, objective, complexity, precision, search_spec)
    inferred = (
        degree=complexity.degree === nothing,
        max_coeffs=complexity.max_coeffs === nothing,
        piecewise=complexity.piecewise === :auto,
        τ=search_spec.τ === nothing,
        max_depth=search_spec.max_depth === nothing,
        min_width=search_spec.min_width === nothing,
        driver_max_iter=search_spec.driver_max_iter === nothing,
        strategy=search_spec.strategy === nothing,
        degree_policy=complexity.degree_policy === nothing)
    return _fit_plan(I, params, inferred)
end

"""
    fit(f, I; kwargs...) -> Approximation

High-level approximation interface. This unexported function calls
`recommend_parameters(f, I; kwargs...)`, then dispatches to the appropriate
expert API. Prefer exported `approxfit` in user code; call
`PolynomialErrorOptimization.fit` if you explicitly want this name.

Examples:

```julia
approxfit(sin, (-3.0, 3.0); target=1e-8)
approxfit(exp, (-1.0, 1.0); target=1e-10, degree=6, piecewise=false)
approxfit(exp, (0.5, 2.5); rel_tol=1e-9, mode=:rel, max_coeffs=7)
```
"""
function fit(f, I::Tuple{<:Real,<:Real}; kwargs...)
    params = recommend_parameters(f, I; kwargs...)
    return fit(f, I, params)
end

fit(f, plan::FitPlan) = fit(f, plan.interval, plan.parameters)

"""
    approxfit(f, I; kwargs...) -> Approximation
    approxfit(f, I, params::FitParameters) -> Approximation

Preferred high-level approximation entry point. This is an exported alias for
`fit` with a package-specific name to avoid collisions with other packages
that export `fit`.
"""
approxfit(f, I::Tuple{<:Real,<:Real}; kwargs...) =
    fit(f, I; kwargs...)

approxfit(f, I::Tuple{<:Real,<:Real}, params::FitParameters) =
    fit(f, I, params)

approxfit(f, plan::FitPlan) = fit(f, plan)

"""
    fit(f, I, params::FitParameters) -> Approximation

Run a high-level fit from an explicit parameter bundle.
"""
function fit(f, I::Tuple{<:Real,<:Real}, params::FitParameters)
    C = params.compute_type
    interval = (C(I[1]), C(I[2]))
    if params.max_coeffs !== nothing
        model = _fit_budget(f, interval, params)
    elseif params.piecewise === false
        model = _fit_single(f, interval, params)
    elseif params.piecewise === :auto
        model = _fit_auto_degree(f, interval, params)
    else
        model = _fit_piecewise_fixed(f, interval, params)
    end
    return Approximation(model, params, interval)
end

"""
    fit_abs(f, I; kwargs...) -> Approximation

Convenience wrapper for `approxfit(f, I; mode=:abs, kwargs...)`.
"""
fit_abs(f, I::Tuple{<:Real,<:Real}; kwargs...) =
    approxfit(f, I; mode=:abs, kwargs...)

"""
    fit_rel(f, I; kwargs...) -> Approximation

Convenience wrapper for `approxfit(f, I; mode=:rel, kwargs...)`.
"""
fit_rel(f, I::Tuple{<:Real,<:Real}; kwargs...) =
    approxfit(f, I; mode=:rel, kwargs...)

"""
    error_bound(x)

Return the verified error bound from an `Approximation`, `OptimResult`, or
`PiecewisePolyApprox`.
"""
error_bound(a::Approximation) = error_bound(a.model)
error_bound(r::OptimResult) = r.total_error
error_bound(pa::PiecewisePolyApprox) = pa.worst_error

"""
    coeff_count(x)

Return the number of polynomial coefficients stored in an approximation.
For piecewise approximations this is the sum across all pieces.
"""
coeff_count(a::Approximation) = coeff_count(a.model)
coeff_count(r::OptimResult) = length(Polynomials.coeffs(r.poly))
coeff_count(pa::PiecewisePolyApprox) =
    sum(length(Polynomials.coeffs(p.result.poly)) for p in pa.pieces)

"""
    is_piecewise(x) -> Bool

Return whether an approximation object is piecewise.
"""
is_piecewise(a::Approximation) = is_piecewise(a.model)
is_piecewise(::OptimResult) = false
is_piecewise(::PiecewisePolyApprox) = true

"""
    pieces(x)

Return the vector of `ApproxPiece` objects for a piecewise approximation.
Returns `nothing` for a single-polynomial `OptimResult`.
"""
pieces(a::Approximation) = pieces(a.model)
pieces(pa::PiecewisePolyApprox) = pa.pieces
pieces(::OptimResult) = nothing

function provide_source(a::Approximation; kwargs...)
    eval_kwargs = haskey(kwargs, :eval_op) ? kwargs : (eval_op=a.parameters.eval_op, kwargs...)
    eval_kwargs = haskey(eval_kwargs, :eval_scheme) ? eval_kwargs :
                  (eval_scheme=a.parameters.eval_scheme, eval_kwargs...)
    if is_piecewise(a) || haskey(kwargs, :interval)
        return provide_source(a.model; eval_kwargs...)
    end
    return provide_source(a.model; interval=a.interval, eval_kwargs...)
end

function provide(a::Approximation; kwargs...)
    eval_kwargs = haskey(kwargs, :eval_op) ? kwargs : (eval_op=a.parameters.eval_op, kwargs...)
    eval_kwargs = haskey(eval_kwargs, :eval_scheme) ? eval_kwargs :
                  (eval_scheme=a.parameters.eval_scheme, eval_kwargs...)
    return provide(a.model; eval_kwargs...)
end

function _fit_single(f, I::Tuple{C,C}, params::FitParameters{TargetT,C}) where {TargetT,C}
    @argcheck params.degree !== nothing ArgumentError(
        "single-polynomial fit requires degree or max_coeffs")
    n = params.degree
    scheme = _scheme_for(params.scheme, n, params.target_type, params.compute_type)
    strategy = _strategy_for(params.effort, scheme, params.strategy)
    if params.mode === :abs
        return eval_approx_optimize(f, n, I, scheme;
            τ=params.τ, max_iter=params.driver_max_iter,
            strategy=strategy, target_type=params.target_type)
    else
        return eval_approx_optimize_relative(f, n, I, scheme;
            τ=params.τ, max_iter=params.driver_max_iter,
            strategy=strategy, target_type=params.target_type)
    end
end

function _fit_piecewise_fixed(f, I::Tuple{C,C}, params::FitParameters{TargetT,C}) where {TargetT,C}
    @argcheck params.degree !== nothing ArgumentError(
        "fixed-degree piecewise fit requires degree")
    n = params.degree
    scheme = _scheme_for(params.scheme, n, params.target_type, params.compute_type)
    strategy = _strategy_for(params.effort, scheme, params.strategy)
    if params.mode === :abs
        return approximate_abs(f, n, I, scheme;
            target=params.target, τ=params.τ, max_depth=params.max_depth,
            min_width=params.min_width, total_coeffs=params.total_coeffs,
            driver_max_iter=params.driver_max_iter, strategy=strategy,
            target_type=params.target_type)
    else
        return approximate_rel(f, n, I, scheme;
            target=params.target, τ=params.τ, max_depth=params.max_depth,
            min_width=params.min_width, total_coeffs=params.total_coeffs,
            driver_max_iter=params.driver_max_iter, strategy=strategy,
            target_type=params.target_type)
    end
end

function _fit_auto_degree(f, I::Tuple{C,C}, params::FitParameters{TargetT,C}) where {TargetT,C}
    try
        single_fit = _fit_single(f, I, params)
        error_bound(single_fit) ≤ params.target && return single_fit
    catch e
        if e isa InterruptException || e isa OutOfMemoryError
            rethrow()
        end
    end
    return _fit_piecewise_fixed(f, I, params)
end

function _fit_budget(f, I::Tuple{C,C}, params::FitParameters{TargetT,C}) where {TargetT,C}
    builder = d -> _scheme_for(params.scheme, d, params.target_type, params.compute_type)
    strategy = params.strategy
    if params.mode === :abs
        return approximate_abs_budget(f, params.max_coeffs, I;
            target=params.target, scheme_builder=builder,
            degree_policy=params.degree_policy, τ=params.τ,
            max_depth=params.max_depth, min_width=params.min_width,
            total_coeffs=params.total_coeffs,
            driver_max_iter=params.driver_max_iter, strategy=strategy,
            compute_type=params.compute_type, target_type=params.target_type)
    else
        return approximate_rel_budget(f, params.max_coeffs, I;
            target=params.target, scheme_builder=builder,
            degree_policy=params.degree_policy, τ=params.τ,
            max_depth=params.max_depth, min_width=params.min_width,
            total_coeffs=params.total_coeffs,
            driver_max_iter=params.driver_max_iter, strategy=strategy,
            compute_type=params.compute_type, target_type=params.target_type)
    end
end

function _validate_recommendation_inputs(
    I, mode, effort, piecewise, scheme, degree, max_coeffs, total_coeffs)
    @argcheck mode in _FIT_MODES ArgumentError(
        "mode must be :abs or :rel, got :$mode")
    @argcheck effort in _FIT_EFFORTS ArgumentError(
        "effort must be :fast, :balanced, or :accurate, got :$effort")
    @argcheck piecewise === :auto || piecewise isa Bool ArgumentError(
        "piecewise must be true, false, or :auto, got $piecewise")
    @argcheck scheme in _FIT_SCHEMES ArgumentError(
        "scheme must be :horner, :horner_fma, :estrin, or :estrin_fma, got :$scheme")
    @argcheck I[1] < I[2] ArgumentError("I must satisfy I[1] < I[2], got $I")
    @argcheck total_coeffs ≥ 0 ArgumentError(
        "total_coeffs must be ≥ 0, got $total_coeffs")
    @argcheck !(degree !== nothing && max_coeffs !== nothing) ArgumentError(
        "pass at most one of degree or max_coeffs")
    degree !== nothing && @argcheck degree ≥ 0 ArgumentError(
        "degree must be ≥ 0, got $degree")
    max_coeffs !== nothing && @argcheck max_coeffs ≥ 1 ArgumentError(
        "max_coeffs must be ≥ 1, got $max_coeffs")
end

function _validate_complexity_spec_inputs(
    degree, max_coeffs, total_coeffs, piecewise, degree_policy)
    @argcheck piecewise === :auto || piecewise isa Bool ArgumentError(
        "piecewise must be true, false, or :auto, got $piecewise")
    @argcheck total_coeffs ≥ 0 ArgumentError(
        "total_coeffs must be ≥ 0, got $total_coeffs")
    @argcheck !(degree !== nothing && max_coeffs !== nothing) ArgumentError(
        "pass at most one of degree or max_coeffs")
    degree !== nothing && @argcheck degree ≥ 0 ArgumentError(
        "degree must be ≥ 0, got $degree")
    max_coeffs !== nothing && @argcheck max_coeffs ≥ 1 ArgumentError(
        "max_coeffs must be ≥ 1, got $max_coeffs")
    degree_policy !== nothing && @argcheck degree_policy in (:max, :min, :min_cost) ArgumentError(
        "degree_policy must be :max, :min, or :min_cost; got :$degree_policy")
end

function _validate_search_spec_inputs(
    scheme, effort, τ, max_depth, min_width, driver_max_iter)
    @argcheck scheme in _FIT_SCHEMES ArgumentError(
        "scheme must be :horner, :horner_fma, :estrin, or :estrin_fma, got :$scheme")
    @argcheck effort in _FIT_EFFORTS ArgumentError(
        "effort must be :fast, :balanced, or :accurate, got :$effort")
    τ !== nothing && @argcheck τ > 0 ArgumentError(
        "τ must be > 0, got $τ")
    max_depth !== nothing && @argcheck max_depth ≥ 0 ArgumentError(
        "max_depth must be ≥ 0, got $max_depth")
    min_width !== nothing && @argcheck min_width ≥ 0 ArgumentError(
        "min_width must be ≥ 0, got $min_width")
    driver_max_iter !== nothing && @argcheck driver_max_iter ≥ 0 ArgumentError(
        "driver_max_iter must be ≥ 0, got $driver_max_iter")
end

function _choose_degree_budget(degree, max_coeffs, piecewise, defaults, profile)
    chosen_degree = degree === nothing ? nothing : Int(degree)
    chosen_max_coeffs = max_coeffs === nothing ? nothing : Int(max_coeffs)
    chosen_piecewise = piecewise

    if chosen_degree === nothing && chosen_max_coeffs === nothing
        chosen_max_coeffs = defaults.max_coeffs + _profile_coeff_bonus(profile)
        chosen_piecewise = piecewise === :auto ? true : piecewise
    end

    if chosen_piecewise === false && chosen_degree === nothing
        chosen_degree = chosen_max_coeffs - 1
        chosen_max_coeffs = nothing
    end

    return chosen_degree, chosen_max_coeffs, chosen_piecewise
end

function _parameters_from_specs(f, I::Tuple{<:Real,<:Real},
    objective::ObjectiveSpec,
    complexity::ComplexitySpec,
    precision::PrecisionSpec{TargetT,ComputeT},
    search::SearchSpec{ComputeT}) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    _validate_recommendation_inputs(
        I, objective.mode, search.effort, complexity.piecewise,
        search.scheme, complexity.degree, complexity.max_coeffs,
        complexity.total_coeffs)

    profile = _sample_profile(f, I, ComputeT)
    if objective.mode === :rel && profile.crosses_zero
        throw(DomainError(I,
            "relative mode requires f to stay away from zero on sampled points; " *
            "use mode=:abs or a relative-zero expert driver for known zeros"))
    end

    defaults = _effort_defaults(search.effort)
    chosen_degree, chosen_max_coeffs, chosen_piecewise =
        _choose_degree_budget(
            complexity.degree, complexity.max_coeffs,
            complexity.piecewise, defaults, profile)
    chosen_max_depth = _choose_max_depth(
        search.max_depth, defaults, search.effort, profile)

    return FitParameters(
        ComputeT(objective.target),
        objective.mode,
        chosen_degree,
        chosen_max_coeffs,
        complexity.total_coeffs,
        chosen_piecewise,
        search.scheme,
        precision.target_type,
        precision.compute_type,
        _eval_scheme_for_scheme(search.scheme),
        _eval_op_for_scheme(search.scheme),
        search.effort,
        ComputeT(search.τ === nothing ? defaults.τ : search.τ),
        chosen_max_depth,
        ComputeT(search.min_width === nothing ? defaults.min_width : search.min_width),
        Int(search.driver_max_iter === nothing ? defaults.driver_max_iter : search.driver_max_iter),
        search.strategy,
        complexity.degree_policy === nothing ? defaults.degree_policy : complexity.degree_policy)
end

function _convert_search_spec(search::SearchSpec, ::Type{T}) where {T<:AbstractFloat}
    return SearchSpec(T;
        scheme=search.scheme,
        effort=search.effort,
        τ=search.τ,
        max_depth=search.max_depth,
        min_width=search.min_width,
        driver_max_iter=search.driver_max_iter,
        strategy=search.strategy)
end

function _fit_plan(I, params::FitParameters{TargetT,ComputeT}, inferred) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    interval = (ComputeT(I[1]), ComputeT(I[2]))
    objective = ObjectiveSpec(params.mode, params.target)
    complexity = ComplexitySpec(
        degree=params.degree,
        max_coeffs=params.max_coeffs,
        total_coeffs=params.total_coeffs,
        piecewise=params.piecewise,
        degree_policy=params.degree_policy)
    precision = PrecisionSpec(
        target_type=params.target_type,
        compute_type=params.compute_type)
    search = SearchSpec(ComputeT;
        scheme=params.scheme,
        effort=params.effort,
        τ=params.τ,
        max_depth=params.max_depth,
        min_width=params.min_width,
        driver_max_iter=params.driver_max_iter,
        strategy=params.strategy)
    return FitPlan(interval, objective, complexity, precision, search, params, inferred)
end

function _profile_coeff_bonus(profile)
    return (profile.variation > 8 ? 1 : 0) +
           (profile.variation > 20 ? 1 : 0)
end

function _choose_max_depth(max_depth, defaults, effort, profile)
    chosen = Int(max_depth === nothing ? defaults.max_depth : max_depth)
    if max_depth === nothing &&
       (profile.variation > 20 || profile.endpoint_jump > 1.5)
        chosen += effort === :fast ? 2 : 4
    end
    return chosen
end

function _target_from_tolerances(mode, target, abs_tol, rel_tol, profile)
    supplied = count(x -> x !== nothing, (target, abs_tol, rel_tol))
    @argcheck supplied == 1 ArgumentError(
        "provide exactly one of target, abs_tol, or rel_tol")
    if target !== nothing
        @argcheck target > 0 ArgumentError("target must be > 0, got $target")
        return target
    elseif abs_tol !== nothing
        @argcheck abs_tol > 0 ArgumentError("abs_tol must be > 0, got $abs_tol")
        return abs_tol
    else
        @argcheck rel_tol > 0 ArgumentError("rel_tol must be > 0, got $rel_tol")
        if mode === :rel
            return rel_tol
        end
        return rel_tol * profile.scale
    end
end

function _sample_profile(f, I, ::Type{T}; samples::Int=257) where T<:AbstractFloat
    tl, tr = T(I[1]), T(I[2])
    @argcheck samples ≥ 2 ArgumentError("samples must be ≥ 2, got $samples")

    values = Vector{T}(undef, samples)
    scale = zero(T)
    for k in 0:(samples-1)
        t = tl + (tr - tl) * T(k) / T(samples - 1)
        y = T(f(t))
        @argcheck isfinite(y) DomainError(t,
            "f(t) must be finite on sampled points; got f($t) = $y")
        values[k+1] = y
        scale = max(scale, abs(y))
    end
    scale = max(scale, eps(T))
    minabs = minimum(abs, values)
    variation = sum(abs(values[i+1] - values[i]) for i in 1:(samples-1)) / scale
    endpoint_jump = abs(values[end] - values[1]) / scale
    zero_floor = sqrt(eps(T)) * scale
    crosses_zero = minabs ≤ zero_floor ||
                   any(values[i] * values[i+1] < 0 for i in 1:(samples-1))
    return (scale=scale, minabs=minabs, variation=variation,
        endpoint_jump=endpoint_jump, crosses_zero=crosses_zero)
end

function _effort_defaults(effort::Symbol)
    if effort === :fast
        return (τ=1e-3, max_depth=22, min_width=0.0,
            driver_max_iter=80, max_coeffs=5, degree_policy=:max)
    elseif effort === :balanced
        return (τ=1e-3, max_depth=28, min_width=0.0,
            driver_max_iter=120, max_coeffs=6, degree_policy=:min)
    else
        return (τ=5e-4, max_depth=32, min_width=0.0,
            driver_max_iter=200, max_coeffs=8, degree_policy=:min_cost)
    end
end

function _scheme_for(kind::Symbol, n::Int,
    ::Type{TargetT}, ::Type{ComputeT}) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    @argcheck n ≥ 0 ArgumentError("degree must be ≥ 0, got $n")
    u = ComputeT(eps(TargetT) / 2)
    if kind === :horner
        return horner_scheme(n, ComputeT; u=u)
    elseif kind === :horner_fma
        return fma_horner_scheme(n, ComputeT; u=u)
    elseif kind === :estrin
        return estrin_scheme(n, ComputeT; u=u)
    elseif kind === :estrin_fma
        return fma_estrin_scheme(n, ComputeT; u=u)
    end
    throw(ArgumentError("unknown scheme :$kind"))
end

_eval_scheme_for_scheme(scheme::Symbol) =
    (scheme === :estrin || scheme === :estrin_fma) ? :estrin : :horner

_eval_op_for_scheme(scheme::Symbol) =
    (scheme === :horner_fma || scheme === :estrin_fma) ? :fma : :muladd

function _strategy_for(effort::Symbol, scheme::EvalScheme, strategy)
    strategy !== nothing && return strategy
    if effort === :fast
        return GridSearch(max(2048, 48 * (scheme.n + 2)))
    elseif effort === :balanced
        return default_strategy(scheme)
    else
        return GridThenOptim(max(6001, 128 * (scheme.n + 2)); bracket=4)
    end
end
