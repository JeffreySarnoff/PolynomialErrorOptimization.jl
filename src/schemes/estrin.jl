# ---------------------------------------------------------------------------
# Estrin schemes and evaluators
# ---------------------------------------------------------------------------

"""
    estrin_eval(p, t) -> typeof(float(t))

Reference Estrin evaluator using `muladd` for affine combines. `p` is in
natural coefficient order (`p[1] = a₀`, `p[end] = aₙ`). The computation is
carried out in `T = typeof(float(t))`.
"""
function estrin_eval(p::AbstractVector{<:Real}, t::Real)
    return _estrin_eval(p, t, Val(:muladd))
end

"""
    fma_estrin_eval(p, t) -> typeof(float(t))

Reference Estrin evaluator using explicit fused multiply-add for affine
combines. Powers `t², t⁴, ...` are formed by rounded multiplication in the
working type.
"""
function fma_estrin_eval(p::AbstractVector{<:Real}, t::Real)
    return _estrin_eval(p, t, Val(:fma))
end

function _estrin_eval(p::AbstractVector{<:Real}, t::Real, op)
    T = typeof(float(t))
    len = length(p)
    len == 0 && return zero(T)
    len == 1 && return T(p[1])

    x = T(t)
    vals = Vector{T}(undef, (len + 1) >>> 1)
    out = 1
    i = 1
    @inbounds while i <= len
        if i == len
            vals[out] = T(p[i])
        else
            vals[out] = _affine_combine(T(p[i + 1]), x, T(p[i]), op)
        end
        out += 1
        i += 2
    end

    len = length(vals)
    power = x * x
    tmp = similar(vals)
    @inbounds while len > 1
        out = 1
        i = 1
        while i <= len
            if i == len
                tmp[out] = vals[i]
            else
                tmp[out] = _affine_combine(vals[i + 1], power, vals[i], op)
            end
            out += 1
            i += 2
        end
        len = (len + 1) >>> 1
        vals, tmp = tmp, vals
        len > 1 && (power *= power)
    end
    return vals[1]
end

_affine_combine(x, y, z, ::Val{:muladd}) = muladd(x, y, z)
_affine_combine(x, y, z, ::Val{:fma}) = fma(x, y, z)

"""
    estrin_scheme(n, T::Type{<:AbstractFloat} = Float64; u::Real = eps(T)/2) -> EvalScheme{T}

Build an `EvalScheme{T}` for the Estrin parallel evaluation scheme.

Estrin pairs adjacent coefficients into "leaves" of the form `aᵢ + a_{i+1}·t`,
then combines them using powers `t², t⁴, …` arranged as a binary tree. This
reduces critical-path length but increases operation count vs. Horner.

Pass `T = Float32`, `T = BigFloat`, etc. to obtain a scheme for other
floating-point precisions.
"""
function estrin_scheme(n::Integer, ::Type{T}=Float64; u::Real=eps(T) / 2) where T<:AbstractFloat
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    n = Int(n)
    u = Float64(u)  # symbolic tree stores u as Float64; converted in build_eval_scheme
    e, ids_to_u = estrin_expr(n; u=u)
    θ = lin_eval_error(e)
    sch = build_eval_scheme(θ, ids_to_u, n, "estrin(T=$T, u=$(T(u)))", T)
    return sch
end

"""
    fma_estrin_scheme(n, T::Type{<:AbstractFloat} = Float64; u::Real = eps(T)/2) -> EvalScheme{T}

Build an `EvalScheme{T}` for Estrin evaluation where each affine combine is a
single rounded fused multiply-add. Power construction still uses rounded
multiplication. This is an opt-in FMA model; `estrin_scheme` remains unchanged.
"""
function fma_estrin_scheme(n::Integer, ::Type{T}=Float64; u::Real=eps(T) / 2) where T<:AbstractFloat
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    n = Int(n)
    u = Float64(u)
    e, ids_to_u = fma_estrin_expr(n; u=u)
    θ = lin_eval_error(e)
    return build_eval_scheme(θ, ids_to_u, n, "fma-estrin(T=$T, u=$(T(u)))", T)
end

"""
    estrin_expr(n; u = 2.0^-53) -> (ErrExpr, Dict{Int,Float64})

Symbolic Round-tree for Estrin's scheme; companion to `horner_expr`.
"""
function estrin_expr(n::Integer; u::Real=2.0^-53)
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    n = Int(n)
    u = Float64(u)
    ids_to_u = Dict{Int,Float64}()
    next_id = Ref(0)
    fresh!() = (next_id[] += 1; next_id[])
    function rnd(e::ErrExpr)
        id = fresh!()
        ids_to_u[id] = u
        return Round(e, u, id)
    end

    # Leaves: for each i in 0..n in steps of 2, build aᵢ + a_{i+1}·t (or just aᵢ).
    function leaf(i::Int)
        if i + 1 > n
            return VarA(i)
        end
        return rnd(simplify_add(VarA(i), rnd(simplify_mul(VarA(i + 1), VarT()))))
    end

    # Pre-compute powers of t at successive levels:  T₀ = t², T₁ = t⁴, …
    function tpow(level::Int)
        # Build t^(2^level) via repeated squaring with rounding at each step.
        tp = VarT()
        for _ in 1:level
            tp = rnd(simplify_mul(tp, tp))
        end
        return tp
    end

    # leaves (length ⌈(n+1)/2⌉)
    leaves = ErrExpr[leaf(i) for i in 0:2:n]

    # Build the Estrin tree level by level.
    level = 1
    while length(leaves) > 1
        T = tpow(level)
        new_leaves = ErrExpr[]
        i = 1
        while i ≤ length(leaves)
            if i + 1 ≤ length(leaves)
                # leaves[i] + leaves[i+1] · T
                push!(new_leaves,
                    rnd(simplify_add(leaves[i],
                        rnd(simplify_mul(leaves[i+1], T)))))
            else
                push!(new_leaves, leaves[i])
            end
            i += 2
        end
        leaves = new_leaves
        level += 1
    end

    return leaves[1], ids_to_u
end

"""
    fma_estrin_expr(n; u = 2.0^-53) -> (ErrExpr, Dict{Int,Float64})

Symbolic Round-tree for Estrin evaluation where every affine combine is one
rounded fused multiply-add. Powers `t², t⁴, ...` are still rounded
multiplications.
"""
function fma_estrin_expr(n::Integer; u::Real=2.0^-53)
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    n = Int(n)
    u = Float64(u)
    ids_to_u = Dict{Int,Float64}()
    next_id = Ref(0)
    fresh!() = (next_id[] += 1; next_id[])
    function rnd(e::ErrExpr)
        id = fresh!()
        ids_to_u[id] = u
        return Round(e, u, id)
    end

    function leaf(i::Int)
        if i + 1 > n
            return VarA(i)
        end
        return rnd(FMA(VarA(i + 1), VarT(), VarA(i)))
    end

    function tpow(level::Int)
        tp = VarT()
        for _ in 1:level
            tp = rnd(simplify_mul(tp, tp))
        end
        return tp
    end

    leaves = ErrExpr[leaf(i) for i in 0:2:n]
    level = 1
    while length(leaves) > 1
        T = tpow(level)
        new_leaves = ErrExpr[]
        i = 1
        while i ≤ length(leaves)
            if i + 1 ≤ length(leaves)
                push!(new_leaves, rnd(FMA(leaves[i + 1], T, leaves[i])))
            else
                push!(new_leaves, leaves[i])
            end
            i += 2
        end
        leaves = new_leaves
        level += 1
    end

    return leaves[1], ids_to_u
end
