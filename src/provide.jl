"""
    provide_source(pa::PiecewisePolyApprox; name = :approx_eval,
                   check_domain = true, eval_type = <coefficient type>,
                   eval_scheme = :horner, eval_op = :muladd)

Generate standalone Julia source code that evaluates a piecewise approximation
without depending on `PolynomialErrorOptimization` or `Polynomials`.

The emitted function:

- finds the correct subinterval via binary search over piece right endpoints,
- evaluates the corresponding polynomial with the requested evaluation shape,
- optionally performs domain checking.

Coefficients are emitted in the approximation's target coefficient type.
`eval_type` controls the arithmetic type used by the generated function.
By default, `eval_type` is the same as the target coefficient type.
`eval_scheme` controls the polynomial evaluation shape and may be `:horner`
(default) or `:estrin`. `eval_op` controls affine combines and may be
`:muladd` (default) or `:fma`.

The returned string is suitable for copy/paste into another program.
"""
function provide_source(pa::PiecewisePolyApprox{TargetT,ComputeT};
    name::Symbol=:approx_eval,
    check_domain::Bool=true,
    eval_type::Type{EvalT}=TargetT,
    eval_scheme::Symbol=:horner,
    eval_op::Symbol=:muladd) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat,EvalT<:AbstractFloat}
    _validate_generated_name(name)
    scheme = _validate_eval_scheme(eval_scheme)
    op = _validate_eval_op(eval_op)

    pieces = pa.pieces
    @argcheck !isempty(pieces) ArgumentError("provide_source: piecewise approximation has no pieces")

    coeff_type_code = _type_code(TargetT)
    eval_type_code = _type_code(EvalT)
    a0 = pieces[1].a
    bN = pieces[end].b

    right_bounds = Tuple(p.b for p in pieces)
    coeff_table = Tuple(_coeff_tuple(p.result.poly, TargetT) for p in pieces)

    bounds_code = _tuple_float_code(Tuple(EvalT(x) for x in right_bounds), eval_type_code)
    coeffs_code = _tuple_tuple_float_code(coeff_table, coeff_type_code)
    lo_name = String(name)
    eval_code = _eval_source_code(scheme, op, lo_name; piecewise=true)

    domain_guard = check_domain ? """
    if t < $(_float_code(EvalT(a0), eval_type_code)) || t > $(_float_code(EvalT(bN), eval_type_code))
        throw(DomainError(t, \"$lo_name: t=\$t outside approximation interval [$(_float_code(EvalT(a0), eval_type_code)), $(_float_code(EvalT(bN), eval_type_code))]\"))
    end
""" : ""

    return """
begin
const $(lo_name)_T = $eval_type_code
const $(lo_name)_coeff_T = $coeff_type_code
const $(lo_name)_bounds = $bounds_code
const $(lo_name)_coeffs = $coeffs_code

function $lo_name(t_in)
    t = $(lo_name)_T(t_in)
$domain_guard
    lo = 1
    hi = length($(lo_name)_bounds)
    while lo < hi
        mid = (lo + hi) >>> 1
        if t <= $(lo_name)_bounds[mid]
            hi = mid
        else
            lo = mid + 1
        end
    end

$eval_code
end
end
"""
end

"""
    provide_source(res::OptimResult; name = :approx_eval,
                   interval = nothing, check_domain = interval !== nothing,
                   eval_type = <coefficient type>,
                   eval_scheme = :horner, eval_op = :muladd)

Generate standalone Julia source code that evaluates a single-polynomial
approximation (`OptimResult.poly`) with the requested evaluation shape.

If `interval` is provided, optional domain checking can be emitted.
Coefficients are emitted in the result's target coefficient type. `eval_type`
controls the arithmetic type used by the generated function and defaults to
the target coefficient type. `eval_scheme` may be `:horner` (default) or
`:estrin`. `eval_op` controls affine combines and may be `:muladd` (default)
or `:fma`.
"""
function provide_source(res::OptimResult{TargetT,ComputeT};
    name::Symbol=:approx_eval,
    interval::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    check_domain::Bool=(interval !== nothing),
    eval_type::Type{EvalT}=TargetT,
    eval_scheme::Symbol=:horner,
    eval_op::Symbol=:muladd) where {TargetT<:AbstractFloat,ComputeT<:AbstractFloat,EvalT<:AbstractFloat}
    _validate_generated_name(name)
    scheme = _validate_eval_scheme(eval_scheme)
    op = _validate_eval_op(eval_op)

    if check_domain
        @argcheck interval !== nothing ArgumentError(
            "provide_source: interval must be provided when check_domain=true")
    end

    coeff_type_code = _type_code(TargetT)
    eval_type_code = _type_code(EvalT)
    coeffs = _coeff_tuple(res.poly, TargetT)
    coeffs_code = _tuple_float_code(coeffs, coeff_type_code)
    lo_name = String(name)
    eval_code = _eval_source_code(scheme, op, lo_name; piecewise=false)

    domain_guard = ""
    if check_domain && interval !== nothing
        tl = EvalT(interval[1])
        tr = EvalT(interval[2])
        @argcheck tl < tr ArgumentError("provide_source: interval must satisfy left < right")
        domain_guard = """
    if t < $(_float_code(tl, eval_type_code)) || t > $(_float_code(tr, eval_type_code))
        throw(DomainError(t, \"$lo_name: t=\$t outside approximation interval [$(_float_code(tl, eval_type_code)), $(_float_code(tr, eval_type_code))]\"))
    end
"""
    end

    return """
begin
const $(lo_name)_T = $eval_type_code
const $(lo_name)_coeff_T = $coeff_type_code
const $(lo_name)_coeffs = $coeffs_code

function $lo_name(t_in)
    t = $(lo_name)_T(t_in)
$domain_guard
$eval_code
end
end
"""
end

"""
    provide(model; name = :approx_eval, module_ = @__MODULE__, force = false, kwargs...)

Generate standalone source with `provide_source`, install it in `module_`, and
return `(fn = <generated function>, source = <source string>)`.

Supported `model` values:

- `PiecewisePolyApprox`
- `OptimResult`
"""
function provide(model;
    name::Symbol=:approx_eval,
    module_::Module=@__MODULE__,
    force::Bool=false,
    kwargs...)
    _validate_generated_name(name)
    if isdefined(module_, name) && !force
        throw(ArgumentError(
            "provide: symbol :$name already exists in target module; pass force=true to overwrite"))
    end

    source = provide_source(model; name=name, kwargs...)
    ex = Meta.parse(source)
    Core.eval(module_, ex)
    fn = Base.invokelatest(getfield, module_, name)
    return (fn=fn, source=source)
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _validate_generated_name(name::Symbol)
    s = String(name)
    @argcheck Base.isidentifier(s) ArgumentError(
        "generated function name must be a valid identifier, got :$name")
    return nothing
end

function _validate_eval_op(eval_op::Symbol)
    (eval_op === :muladd || eval_op === :fma) && return eval_op
    throw(ArgumentError("eval_op must be :muladd or :fma, got :$eval_op"))
end

function _validate_eval_scheme(eval_scheme::Symbol)
    (eval_scheme === :horner || eval_scheme === :estrin) && return eval_scheme
    throw(ArgumentError("eval_scheme must be :horner or :estrin, got :$eval_scheme"))
end

function _eval_source_code(eval_scheme::Symbol, eval_op::Symbol, lo_name::String; piecewise::Bool)
    coeffs_line = piecewise ? "    coeffs = $(lo_name)_coeffs[lo]" :
                  "    coeffs = $(lo_name)_coeffs"
    eval_scheme === :horner && return _horner_eval_source_code(eval_op, lo_name, coeffs_line)
    eval_scheme === :estrin && return _estrin_eval_source_code(eval_op, lo_name, coeffs_line)
    throw(ArgumentError("eval_scheme must be :horner or :estrin, got :$eval_scheme"))
end

function _horner_eval_source_code(eval_op::Symbol, lo_name::String, coeffs_line::String)
    step_code = _horner_step_code(eval_op, lo_name)
    return """
$coeffs_line
    y = $(lo_name)_T(coeffs[end])
    @inbounds for j in (length(coeffs)-1):-1:1
        y = $step_code
    end
    return y"""
end

function _horner_step_code(eval_op::Symbol, lo_name::String)
    coeff = "$(lo_name)_T(coeffs[j])"
    eval_op === :muladd && return "muladd(y, t, $coeff)"
    eval_op === :fma && return "fma(y, t, $coeff)"
    throw(ArgumentError("eval_op must be :muladd or :fma, got :$eval_op"))
end

function _estrin_eval_source_code(eval_op::Symbol, lo_name::String, coeffs_line::String)
    combine = eval_op === :muladd ? "muladd" :
              eval_op === :fma ? "fma" :
              throw(ArgumentError("eval_op must be :muladd or :fma, got :$eval_op"))
    return """
$coeffs_line
    len = length(coeffs)
    len == 0 && return zero($(lo_name)_T)
    len == 1 && return $(lo_name)_T(coeffs[1])

    vals = Vector{$(lo_name)_T}(undef, (len + 1) >>> 1)
    out = 1
    i = 1
    @inbounds while i <= len
        if i == len
            vals[out] = $(lo_name)_T(coeffs[i])
        else
            vals[out] = $combine($(lo_name)_T(coeffs[i + 1]), t, $(lo_name)_T(coeffs[i]))
        end
        out += 1
        i += 2
    end

    len = length(vals)
    power = t * t
    tmp = similar(vals)
    @inbounds while len > 1
        out = 1
        i = 1
        while i <= len
            if i == len
                tmp[out] = vals[i]
            else
                tmp[out] = $combine(vals[i + 1], power, vals[i])
            end
            out += 1
            i += 2
        end
        len = (len + 1) >>> 1
        vals, tmp = tmp, vals
        len > 1 && (power *= power)
    end
    return vals[1]"""
end

_type_code(::Type{T}) where T<:AbstractFloat = string(T)

_float_code(x::T, type_code::String) where T<:AbstractFloat =
    string(type_code, "(", repr(x), ")")

_tuple_float_code(xs::Tuple, type_code::String) =
    "(" * join((_float_code(x, type_code) for x in xs), ", ") *
    (length(xs) == 1 ? ",)" : ")")

_tuple_tuple_float_code(xss::Tuple, type_code::String) =
    "(" * join((_tuple_float_code(xs, type_code) for xs in xss), ", ") *
    (length(xss) == 1 ? ",)" : ")")


_coeff_tuple(poly, ::Type{T}) where T<:AbstractFloat = Tuple(T(c) for c in Polynomials.coeffs(poly))

# ---------------------------------------------------------------------------
# File output helper
# ---------------------------------------------------------------------------

"""
    provide_file(model, path; name = :approx_eval, kwargs...)

Generate standalone Julia source code for the given approximation model and write it directly to the file at `path`.
This is a convenience wrapper around `provide_source` for one-step export into another project.

Supported `model` values:
- `PiecewisePolyApprox`
- `OptimResult`

Additional keyword arguments are passed to `provide_source`.
Returns the file path as a string.
"""
function provide_file(model, path; name::Symbol=:approx_eval, kwargs...)
    source = provide_source(model; name=name, kwargs...)
    open(path, "w") do io
        write(io, source)
    end
    return path
end
