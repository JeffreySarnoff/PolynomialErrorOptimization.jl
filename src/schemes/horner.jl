"""
    horner.jl

Reference Horner-rule evaluation (paper Algorithm 1) and Horner-specific
`EvalScheme` constructors:

* `horner_eval(p, t)`        — Horner evaluation in IEEE binary64 (for testing).
* `fma_horner_eval(p, t)`    — Horner evaluation with explicit `fma`.
* `horner_scheme(n; u)`      — closed-form `EvalScheme` matching paper eq. (7).
* `fma_horner_scheme(n; u)`  — closed-form `EvalScheme` for FMA Horner.
* `horner_expr(n; u)`        — symbolic `Round`-tree for the Horner scheme; can
                               be passed through `lin_eval_error`/`build_eval_scheme`
                               to verify `horner_scheme`.
* `fma_horner_expr(n; u)`    — symbolic tree for FMA Horner.
"""

# ---------------------------------------------------------------------------
# Plain Horner evaluation in Float64 (Algorithm 1 of the paper)
# ---------------------------------------------------------------------------

"""
    horner_eval(p, t) -> typeof(float(t))

Reference implementation of paper Algorithm 1: `p` is a vector of coefficients
in natural order (`p[1] = a₀`, `p[end] = aₙ`).  The computation is carried out
in the floating-point type `T = typeof(float(t))`.
"""
function horner_eval(p::AbstractVector{<:Real}, t::Real)
    T = typeof(float(t))
    isempty(p) && return zero(T)
    @inbounds r = T(p[end])
    @inbounds for k in (length(p)-1):-1:1
        r = muladd(r, T(t), T(p[k]))
    end
    return r
end

"""
    fma_horner_eval(p, t) -> typeof(float(t))

Reference Horner evaluator using an explicit fused multiply-add at every
Horner step. `p` is in natural coefficient order (`p[1] = a₀`,
`p[end] = aₙ`). The computation is carried out in
`T = typeof(float(t))`.
"""
function fma_horner_eval(p::AbstractVector{<:Real}, t::Real)
    T = typeof(float(t))
    isempty(p) && return zero(T)
    @inbounds r = T(p[end])
    @inbounds for k in (length(p)-1):-1:1
        r = fma(r, T(t), T(p[k]))
    end
    return r
end

# ---------------------------------------------------------------------------
# Closed-form Horner EvalScheme (paper Example 1, eq. (7))
# ---------------------------------------------------------------------------

"""
    horner_scheme(n; u = 2.0^-53) -> EvalScheme

Linearised-evaluation-error model for the standard Horner scheme of paper
Algorithm 1, when all operations use a single FP precision with rounding
unit `u`.  Reproduces paper Example 1 / eq. (7):

    π₁(t)ᵀ            = (u, ut, …, ut^{n-1}, ut^n)
    π_{i+1}(t)ᵀ       = (0, …, 0, 2ut^i, 2ut^{i+1}, …, 2ut^n)   for 1 ≤ i ≤ n-1
    π_{n+1}(t)ᵀ       = (0, …, 0, ut^n)

Note that this matches the bound (7) of the paper:

    |θ_lin^{Horner}|  ≤  2u · Σⱼ'' |Σᵢ aᵢ tⁱ|

where the double primes mean the first and last terms are halved.
"""
function horner_scheme(n::Integer, ::Type{T}=Float64; u::Real=eps(T) / 2) where T<:AbstractFloat
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    n = Int(n)
    u = T(u)

    # Special case n = 0: no arithmetic operations, hence no rounding error.
    if n == 0
        π = Vector{Function}(undef, 1)
        π[1] = let n = n
            (t::T) -> zeros(T, n + 1)
        end
        return EvalScheme{T}(0, 1, π, "horner(T=$T, u=$u)[degree-0]")
    end

    # k = n + 1 distinct rows.
    π = Vector{Function}(undef, n + 1)

    # Row 1: π₁(t)[j+1] = u · t^j   for j = 0..n
    π[1] = let u = u, n = n
        (t::T) -> begin
            v = Vector{T}(undef, n + 1)
            tj = one(T)
            @inbounds for j in 0:n
                v[j+1] = u * tj
                tj *= t
            end
            v
        end
    end

    # Rows 2..n: π_{i+1}(t)[j+1] = 2u · t^j for j ≥ i, else 0   for i = 1..n-1
    for i in 1:(n-1)
        π[i+1] = let u = u, n = n, i = i
            (t::T) -> begin
                v = zeros(T, n + 1)
                tj = one(T)
                @inbounds for _ in 1:i
                    tj *= t
                end
                @inbounds for j in i:n
                    v[j+1] = 2 * u * tj
                    if j < n
                        tj *= t
                    end
                end
                v
            end
        end
    end

    # Row n+1: π_{n+1}(t)[j+1] = u · t^n  for j = n, else 0
    π[n+1] = let u = u, n = n
        (t::T) -> begin
            v = zeros(T, n + 1)
            @inbounds v[n+1] = u * t^n
            v
        end
    end

    return EvalScheme{T}(n, n + 1, π, "horner(T=$T, u=$u)")
end

"""
    fma_horner_scheme(n, T = Float64; u = eps(T)/2) -> EvalScheme{T}

Linearised-evaluation-error model for Horner evaluation where each step is a
single rounded fused multiply-add:

    r = RN(r * t + a[k])

This is an opt-in FMA model. It does not replace `horner_scheme`, which models
separate rounded multiplication and addition.
"""
function fma_horner_scheme(n::Integer, ::Type{T}=Float64; u::Real=eps(T) / 2) where T<:AbstractFloat
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    n = Int(n)
    u = T(u)

    if n == 0
        π = Vector{Function}(undef, 1)
        π[1] = let n = n
            (t::T) -> zeros(T, n + 1)
        end
        return EvalScheme{T}(0, 1, π, "fma-horner(T=$T, u=$u)[degree-0]")
    end

    π = Vector{Function}(undef, n)
    for k in 0:(n - 1)
        π[k + 1] = let u = u, n = n, k = k
            (t::T) -> begin
                v = zeros(T, n + 1)
                tk = one(T)
                @inbounds for _ in 1:k
                    tk *= t
                end
                @inbounds for j in k:n
                    v[j + 1] = u * tk
                    j < n && (tk *= t)
                end
                v
            end
        end
    end

    return EvalScheme{T}(n, n, π, "fma-horner(T=$T, u=$u)")
end

# ---------------------------------------------------------------------------
# Symbolic Horner expression tree (for testing against horner_scheme)
# ---------------------------------------------------------------------------

"""
    horner_expr(n; u = 2.0^-53) -> (ErrExpr, Dict{Int,Float64})

Build the symbolic expression tree corresponding to paper Algorithm 1 with
all operations rounded at unit `u`. Returns `(e, ids_to_u)` so it can be
passed straight through `lin_eval_error` / `build_eval_scheme`.

Useful as a sanity check: the EvalScheme returned by `build_eval_scheme(...)`
on this output should be functionally equivalent to `horner_scheme(n; u)`.
"""
function horner_expr(n::Integer; u::Real=2.0^-53)
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

    # r_n = a_n  (no rounding; aₙ already representable)
    r = VarA(n)
    # for k = n-1 downto 0:
    #     r = RN( RN(r * t, u) + a_k, u )
    for k in (n-1):-1:0
        r = rnd(simplify_add(rnd(simplify_mul(r, VarT())), VarA(k)))
    end
    return r, ids_to_u
end

"""
    fma_horner_expr(n; u = 2.0^-53) -> (ErrExpr, Dict{Int,Float64})

Build the symbolic expression tree for Horner evaluation where every Horner
step is one rounded fused multiply-add.
"""
function fma_horner_expr(n::Integer; u::Real=2.0^-53)
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

    r = VarA(n)
    for k in (n - 1):-1:0
        r = rnd(FMA(r, VarT(), VarA(k)))
    end
    return r, ids_to_u
end
