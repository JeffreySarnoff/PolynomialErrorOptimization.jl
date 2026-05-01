"""
    eval_error.jl

Symbolic evaluation-error analysis for arbitrary polynomial evaluation
schemes, implementing Algorithm 2 (`LINEVALERROR`) of the paper.

The user-facing entry point is `lin_eval_error(e)` which returns a
`Dict{Int, ErrExpr}` mapping each rounding-error variable id to its
symbolic coefficient.  `build_eval_scheme` converts this into the runtime
`EvalScheme` consumed by the rest of the package.

For most users the higher-level shortcuts such as `horner_scheme(n)`,
`fma_horner_scheme(n)`, `estrin_scheme(n)`, and `fma_estrin_scheme(n)` are
sufficient; the AST is here for users who need custom mixed-precision schemes.
"""

# ---------------------------------------------------------------------------
# Expression AST
# ---------------------------------------------------------------------------

"""
    ErrExpr

Abstract supertype of nodes in the symbolic expression tree used by
`lin_eval_error`. The concrete types are:

* `VarT`              — the input variable `t`.
* `VarA(i)`           — the polynomial coefficient `aᵢ`,  `0 ≤ i ≤ n`.
* `Const(v)`          — a numeric literal.
* `Neg(e)`            — `-e`.
* `Add(l, r)`         — `l + r`.
* `Mul(l, r)`         — `l * r`.
* `FMA(x, y, z)`      — exact fused multiply-add expression `x*y + z`
                        before any surrounding `Round`.
* `Round(e, u, id)`   — `RN(e, u)` where `u` is the rounding bound and `id`
                        is a globally-unique tag identifying this rounding op.
                        Two `Round` nodes sharing the same `id` are treated as
                        the *same* rounding operation (paper Section 2: the
                        same subexpression `RN(e′, u)` shares one `ε` variable).
"""
abstract type ErrExpr end

struct VarT <: ErrExpr end
struct VarA <: ErrExpr
    i::Int
end
struct Const <: ErrExpr
    v::Float64
end
struct Neg <: ErrExpr
    e::ErrExpr
end
struct Add <: ErrExpr
    l::ErrExpr
    r::ErrExpr
end
struct Mul <: ErrExpr
    l::ErrExpr
    r::ErrExpr
end
struct FMA <: ErrExpr
    x::ErrExpr
    y::ErrExpr
    z::ErrExpr
end
struct Round <: ErrExpr
    e::ErrExpr
    u::Float64
    id::Int
end

# Convenience constructors / pretty-printing
Base.show(io::IO, ::VarT) = print(io, "t")
Base.show(io::IO, x::VarA) = print(io, "a", x.i)
Base.show(io::IO, x::Const) = print(io, x.v)
Base.show(io::IO, x::Neg) = (print(io, "-("); show(io, x.e); print(io, ")"))
Base.show(io::IO, x::Add) =
    (print(io, "("); show(io, x.l); print(io, " + "); show(io, x.r); print(io, ")"))
Base.show(io::IO, x::Mul) =
    (print(io, "("); show(io, x.l); print(io, " * "); show(io, x.r); print(io, ")"))
Base.show(io::IO, x::FMA) =
    (print(io, "fma("); show(io, x.x); print(io, ", "); show(io, x.y);
     print(io, ", "); show(io, x.z); print(io, ")"))
Base.show(io::IO, x::Round) =
    (print(io, "RN["); show(io, x.e); print(io, ", id=", x.id, ", u=", x.u, "]"))

# ---------------------------------------------------------------------------
# Algebraic simplification used while building coefficient expressions
# ---------------------------------------------------------------------------

is_zero_const(::ErrExpr) = false
is_zero_const(x::Const) = iszero(x.v)
is_one_const(::ErrExpr) = false
is_one_const(x::Const) = x.v == 1.0

simplify_neg(e::Const) = Const(-e.v)
simplify_neg(e::Neg) = e.e
simplify_neg(e::ErrExpr) = Neg(e)

function simplify_add(l::ErrExpr, r::ErrExpr)
    is_zero_const(l) && return r
    is_zero_const(r) && return l
    if l isa Const && r isa Const
        return Const(l.v + r.v)
    end
    return Add(l, r)
end

function simplify_mul(l::ErrExpr, r::ErrExpr)
    is_zero_const(l) && return Const(0.0)
    is_zero_const(r) && return Const(0.0)
    is_one_const(l) && return r
    is_one_const(r) && return l
    if l isa Const && r isa Const
        return Const(l.v * r.v)
    end
    return Mul(l, r)
end

# ---------------------------------------------------------------------------
# Algorithm 2 — LINEVALERROR
# ---------------------------------------------------------------------------

"""
    lin_eval_error(e::ErrExpr) -> Dict{Int, ErrExpr}

Algorithm 2 of the paper. Given an arithmetic expression `e` that contains
rounding nodes `Round(e′, u, id)`, return a dictionary mapping each rounding
id to the symbolic coefficient `θ_lin,i` such that

    θ_lin(a, t)  =  Σᵢ θ_lin,i(a, t) · εᵢ

with `|εᵢ| ≤ uᵢ`.
"""
function lin_eval_error(e::ErrExpr)
    out = Dict{Int,ErrExpr}()
    _lin_eval!(out, e)
    return out
end

# Helper that adds a contribution to the (id => coeff) accumulator.
function _add_contribution!(d::Dict{Int,ErrExpr}, id::Int, coeff::ErrExpr)
    is_zero_const(coeff) && return d
    if haskey(d, id)
        d[id] = simplify_add(d[id], coeff)
    else
        d[id] = coeff
    end
    return d
end

function _lin_eval!(out::Dict{Int,ErrExpr}, e::ErrExpr)
    if e isa VarT || e isa VarA || e isa Const
        # No rounding inside; θ_lin contribution is empty
        return out

    elseif e isa Round
        # Recurse into subexpression first
        inner = lin_eval_error(e.e)
        for (id, coeff) in inner
            _add_contribution!(out, id, coeff)
        end
        # Add this rounding's own contribution: e.e * ε^{u}_{e.id}
        _add_contribution!(out, e.id, e.e)
        return out

    elseif e isa Neg
        inner = lin_eval_error(e.e)
        for (id, coeff) in inner
            _add_contribution!(out, id, simplify_neg(coeff))
        end
        return out

    elseif e isa Add
        left = lin_eval_error(e.l)
        right = lin_eval_error(e.r)
        for (id, coeff) in left
            _add_contribution!(out, id, coeff)
        end
        for (id, coeff) in right
            _add_contribution!(out, id, coeff)
        end
        return out

    elseif e isa Mul
        # ∂/∂ε for product rule:  g · θ_l  +  f · θ_r
        left = lin_eval_error(e.l)
        right = lin_eval_error(e.r)
        for (id, coeff) in left
            _add_contribution!(out, id, simplify_mul(e.r, coeff))
        end
        for (id, coeff) in right
            _add_contribution!(out, id, simplify_mul(e.l, coeff))
        end
        return out
    elseif e isa FMA
        # ∂/∂ε for exact x*y+z: y · θ_x + x · θ_y + θ_z.
        xerr = lin_eval_error(e.x)
        yerr = lin_eval_error(e.y)
        zerr = lin_eval_error(e.z)
        for (id, coeff) in xerr
            _add_contribution!(out, id, simplify_mul(e.y, coeff))
        end
        for (id, coeff) in yerr
            _add_contribution!(out, id, simplify_mul(e.x, coeff))
        end
        for (id, coeff) in zerr
            _add_contribution!(out, id, coeff)
        end
        return out
    end
    error("lin_eval_error: unhandled expression type $(typeof(e))")
end

# ---------------------------------------------------------------------------
# Compilation: from ErrExpr to a runtime function of t
# ---------------------------------------------------------------------------

"""
    extract_aj_coeff(e::ErrExpr, j::Int) -> ErrExpr

Return the coefficient of `aⱼ` in `e`, assuming `e` is linear in the
coefficients `a₀, …`.  Used to project the symbolic per-id error onto the
monomial basis to build `πᵢ(t)`.

Implementation: substitute aⱼ = 1, all other aₖ = 0, simplify.  The result
depends only on `t` (as needed) plus `Const`s.
"""
function extract_aj_coeff(e::ErrExpr, j::Int)::ErrExpr
    if e isa VarT
        return e
    elseif e isa VarA
        return e.i == j ? Const(1.0) : Const(0.0)
    elseif e isa Const
        return e
    elseif e isa Neg
        return simplify_neg(extract_aj_coeff(e.e, j))
    elseif e isa Add
        return simplify_add(extract_aj_coeff(e.l, j), extract_aj_coeff(e.r, j))
    elseif e isa Mul
        # Linear-in-a assumption: at most one factor mentions a's.
        # We just apply the (correct) chain rule:  ∂/∂aⱼ (lr) = (∂l/∂aⱼ)·r + l·(∂r/∂aⱼ).
        # Then evaluate the OTHER side at a=0.  But simpler: use the assumption
        # that l*r is linear in aⱼ (so at most one of l, r depends on aⱼ).
        l_dep = depends_on_aj(e.l, j)
        r_dep = depends_on_aj(e.r, j)
        if !l_dep && !r_dep
            return Const(0.0)
        elseif l_dep && !r_dep
            return simplify_mul(extract_aj_coeff(e.l, j), strip_a(e.r))
        elseif !l_dep && r_dep
            return simplify_mul(strip_a(e.l), extract_aj_coeff(e.r, j))
        else
            # Both sides of the Mul depend on `a`, so the expression is not
            # linear in `a`.  This violates the assumption documented after
            # eq. (8) of the paper ("the functions θ_lin,i are linear with
            # respect to the coefficients a") and indicates either a bad
            # evaluation scheme or a bug in `lin_eval_error`.
            throw(ArgumentError(
                "extract_aj_coeff: expression is non-linear in a (both " *
                "factors of a Mul depend on a). The evaluation scheme " *
                "must be linear in the polynomial coefficients."))
        end
    elseif e isa FMA
        return extract_aj_coeff(simplify_add(simplify_mul(e.x, e.y), e.z), j)
    elseif e isa Round
        # In our use-case, Round nodes only appear inside lin_eval_error's
        # output coefficients indirectly (they were stripped). If we hit one
        # here we just descend.
        return extract_aj_coeff(e.e, j)
    end
    error("extract_aj_coeff: unhandled expression type $(typeof(e))")
end

depends_on_aj(::VarT, ::Int) = false
depends_on_aj(x::VarA, j::Int) = x.i == j
depends_on_aj(::Const, ::Int) = false
depends_on_aj(x::Neg, j::Int) = depends_on_aj(x.e, j)
depends_on_aj(x::Add, j::Int) =
    depends_on_aj(x.l, j) || depends_on_aj(x.r, j)
depends_on_aj(x::Mul, j::Int) =
    depends_on_aj(x.l, j) || depends_on_aj(x.r, j)
depends_on_aj(x::FMA, j::Int) =
    depends_on_aj(x.x, j) || depends_on_aj(x.y, j) || depends_on_aj(x.z, j)
depends_on_aj(x::Round, j::Int) = depends_on_aj(x.e, j)

# Substitute a₀ = a₁ = … = 0  (used to evaluate the "other" factor in Mul above).
strip_a(::VarT) = VarT()
strip_a(::VarA) = Const(0.0)
strip_a(x::Const) = x
strip_a(x::Neg) = simplify_neg(strip_a(x.e))
strip_a(x::Add) = simplify_add(strip_a(x.l), strip_a(x.r))
strip_a(x::Mul) = simplify_mul(strip_a(x.l), strip_a(x.r))
strip_a(x::FMA) = simplify_add(simplify_mul(strip_a(x.x), strip_a(x.y)), strip_a(x.z))
strip_a(x::Round) = strip_a(x.e)

"""
    compile_t(e::ErrExpr, T::Type{<:AbstractFloat} = Float64) -> Function

Compile `e` (which must depend only on `t`, not on any `aⱼ`) into a
`T → T` closure.  Used to turn the per-`(id, j)` symbolic
coefficient into the entry of `πᵢ(t)`.
"""
function compile_t(e::ErrExpr, ::Type{T}=Float64) where T<:AbstractFloat
    if e isa VarT
        return (t::T) -> t
    elseif e isa VarA
        error("compile_t: residual VarA($(e.i)); expression must depend only on t.")
    elseif e isa Const
        v = T(e.v)
        return (t::T) -> v
    elseif e isa Neg
        f = compile_t(e.e, T)
        return (t::T) -> -f(t)
    elseif e isa Add
        f = compile_t(e.l, T)
        g = compile_t(e.r, T)
        return (t::T) -> f(t) + g(t)
    elseif e isa Mul
        f = compile_t(e.l, T)
        g = compile_t(e.r, T)
        return (t::T) -> f(t) * g(t)
    elseif e isa FMA
        f = compile_t(e.x, T)
        g = compile_t(e.y, T)
        h = compile_t(e.z, T)
        return (t::T) -> fma(f(t), g(t), h(t))
    elseif e isa Round
        return compile_t(e.e, T)
    end
    error("compile_t: unhandled expression type $(typeof(e))")
end

# ---------------------------------------------------------------------------
# Building an EvalScheme from the symbolic output
# ---------------------------------------------------------------------------

"""
    collect_rounding_us(e::ErrExpr) -> Dict{Int, Float64}

Walk `e` and record the `u` bound associated with each rounding `id`.
Inconsistent `u`s for the same `id` raise an error.
"""
function collect_rounding_us(e::ErrExpr)
    d = Dict{Int,Float64}()
    _collect_us!(d, e)
    return d
end

function _collect_us!(d::Dict{Int,Float64}, e::ErrExpr)
    if e isa VarT || e isa VarA || e isa Const
        return
    elseif e isa Round
        if haskey(d, e.id)
            @argcheck d[e.id] == e.u ArgumentError(
                "Inconsistent u for rounding id $(e.id): $(d[e.id]) vs $(e.u).")
        else
            d[e.id] = e.u
        end
        _collect_us!(d, e.e)
    elseif e isa Neg
        _collect_us!(d, e.e)
    elseif e isa Add
        _collect_us!(d, e.l)
        _collect_us!(d, e.r)
    elseif e isa Mul
        _collect_us!(d, e.l)
        _collect_us!(d, e.r)
    elseif e isa FMA
        _collect_us!(d, e.x)
        _collect_us!(d, e.y)
        _collect_us!(d, e.z)
    end
end

"""
    build_eval_scheme(θ, ids_to_u, n, label, T = Float64) -> EvalScheme{T}

Construct an `EvalScheme{T}` from the symbolic linearised-error dictionary `θ`
(returned by `lin_eval_error`) and the rounding-id → u-bound map
`ids_to_u` (returned by `collect_rounding_us`).

For each `(id, coeff_expr)` pair this produces a function `πᵢ : T →
Vector{T}` of length `n+1`, with the factor `uᵢ` already folded in:

    πᵢ(t)[j+1] = uᵢ · (∂ coeff_expr / ∂ aⱼ)(t)
"""
function build_eval_scheme(θ::Dict{Int,ErrExpr},
    ids_to_u::Dict{Int,Float64},
    n::Int, label::AbstractString,
    ::Type{T}=Float64) where T<:AbstractFloat
    ids = sort!(collect(keys(θ)))
    π = Vector{Function}(undef, length(ids))
    for (i, id) in pairs(ids)
        u_raw = get(ids_to_u, id) do
            throw(KeyError("build_eval_scheme: ids_to_u missing entry for id=$id"))
        end
        coeff_expr = θ[id]
        # Compile the n+1 functions  ∂coeff/∂aⱼ : T → T,  j = 0..n
        funs = Function[compile_t(extract_aj_coeff(coeff_expr, j), T) for j in 0:n]
        π[i] = let funs = funs, u = T(u_raw), n = n
            (t::T) -> begin
                v = Vector{T}(undef, n + 1)
                @inbounds for j in 1:(n+1)
                    v[j] = u * funs[j](t)
                end
                v
            end
        end
    end
    return EvalScheme{T}(n, length(ids), π, String(label))
end
