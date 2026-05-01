"""
    types.jl

Core data types for `PolynomialErrorOptimization`.

This file defines:

* `Signature`        — signs σ ∈ {-1, 0, +1}^{k+1}
* `Index`            — ω = (t, σ) ∈ Ω = I × 𝔖
* `EvalScheme`       — bundle of `π_i(t)` functions giving the linearised
                       evaluation-error bound  θ(a, t) = Σᵢ |πᵢ(t)ᵀ a|
* `AbstractMode`,
  `AbsoluteMode`,
  `RelativeMode`,
  `RelativeZeroMode` — formulation tags for absolute / (P^rel) / (P^rel2)
"""

# ---------------------------------------------------------------------------
# Signature and Index
# ---------------------------------------------------------------------------

"""
    Signature

A length-`k+1` vector of signs (`-1`, `0`, or `+1`).

The first entry σ₀ multiplies the function term `f(t)`; entries σ₁…σ_k
multiply the error rows π₁(t)…π_k(t) coming from the evaluation-error model.
"""
const Signature = Vector{Int8}

"""
    Index{T<:AbstractFloat}(t::T, σ::Signature)

An element ω = (t, σ) ∈ Ω = I × 𝔖.  `t` is a real point in the input interval
and `σ` is the signature of length `k+1` where `k` is the number of distinct
rounding-error variables in the evaluation scheme.  The type parameter `T`
is the floating-point type used for the point `t` (e.g. `Float32`, `Float64`,
`BigFloat`).

`Index` values are constructed by `init_points`, `find_new_index`, and
`exchange`; users normally never build them by hand.
"""
struct Index{T<:AbstractFloat}
    t::T
    σ::Signature
end

function Base.show(io::IO, ω::Index)
    print(io, "Index(t=", ω.t, ", σ=", Int.(ω.σ), ")")
end

Base.:(==)(a::Index, b::Index) = a.t == b.t && a.σ == b.σ

# ---------------------------------------------------------------------------
# EvalScheme
# ---------------------------------------------------------------------------

"""
    EvalScheme{T<:AbstractFloat}(n, k, π, label)

A description of a polynomial evaluation scheme for floating-point type `T`.

# Fields
* `n::Int`               — polynomial degree (so `n+1` coefficients).
* `k::Int`               — number of distinct rounding-error variables.
* `π::Vector{Function}`  — length `k`. Each `πᵢ : T → Vector{T}` of
                           length `n+1`. The factor `uᵢ` is **folded into**
                           `πᵢ`, so the linearised evaluation error satisfies

                              |θ_lin(a, t)| ≤ Σᵢ |πᵢ(t)ᵀ a|  =:  θ(a, t)
                           (paper eq. (8)).
* `label::String`        — informational, e.g. `"horner-binary64"`.

Construct with `horner_scheme(n, T)`, `estrin_scheme(n, T)`, or by calling
`build_eval_scheme` on the result of `lin_eval_error`.
"""
struct EvalScheme{T<:AbstractFloat}
    n::Int
    k::Int
    π::Vector{Function}
    label::String

    function EvalScheme{T}(n::Integer, k::Integer, π::AbstractVector,
        label::AbstractString) where T<:AbstractFloat
        @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
        @argcheck k ≥ 0 ArgumentError("k must be ≥ 0, got $k")
        @argcheck length(π) == k ArgumentError(
            "length(π) = $(length(π)) must equal k = $k")
        return new{T}(Int(n), Int(k), Vector{Function}(π), String(label))
    end
    # Backward-compatible unparameterised constructor defaults to Float64.
    EvalScheme(n::Integer, k::Integer, π::AbstractVector,
        label::AbstractString) = EvalScheme{Float64}(n, k, π, label)
end

"""
    fptype(scheme::EvalScheme{T}) -> Type{T}

Return the floating-point element type `T` of `scheme`.
"""
fptype(::EvalScheme{T}) where T = T

function Base.show(io::IO, s::EvalScheme)
    print(io, "EvalScheme(", s.label, "; n=", s.n, ", k=", s.k, ")")
end

# ---------------------------------------------------------------------------
# Modes (absolute / relative / relative-with-zero)
# ---------------------------------------------------------------------------

"""
    AbstractMode

Tag type selecting the formulation of (P_general).
"""
abstract type AbstractMode end

"""
    AbsoluteMode()

Minimise   max_{t ∈ I}  ( |f(t) − p(t)|  +  θ(a, t) )    (paper (P_general)).
"""
struct AbsoluteMode <: AbstractMode end

"""
    RelativeMode()

Minimise   max_{t ∈ I}  ( |f(t) − p(t)|  +  θ(a, t) ) / |f(t)|

valid when `f` does not vanish on `I`.  Implemented as paper eq. (15) by
replacing `α(ω)` with `α_r(ω) = (f(t), σ₀π₀(t)ᵀ + Σᵢ σᵢ πᵢ(t)ᵀ)ᵀ`.
"""
struct RelativeMode <: AbstractMode end

"""
    RelativeZeroMode{T<:AbstractFloat}(t_z, s)

Relative-error mode when `f` has a zero of finite order `s` at `t_z ∈ I`
(paper eq. (16)).  The first `s` polynomial coefficients are forced to zero
by reformulating in the basis `((t-t_z)ˢ/f(t), …, (t-t_z)ⁿ/f(t))`.
The type parameter `T` matches the floating-point type of the computation.
"""
struct RelativeZeroMode{T<:AbstractFloat} <: AbstractMode
    t_z::T
    s::Int
    function RelativeZeroMode(t_z::Real, s::Integer)
        @argcheck s ≥ 1 ArgumentError("zero order s must be ≥ 1, got $s")
        ftz = float(t_z)
        return new{typeof(ftz)}(ftz, Int(s))
    end
    function RelativeZeroMode{T}(t_z::Real, s::Integer) where T<:AbstractFloat
        @argcheck s ≥ 1 ArgumentError("zero order s must be ≥ 1, got $s")
        return new{T}(T(t_z), Int(s))
    end
end

_mode_symbol(::AbsoluteMode) = :abs
_mode_symbol(::RelativeMode) = :rel

function _mode_from_symbol(mode::Symbol)
    mode === :abs && return AbsoluteMode()
    mode === :rel && return RelativeMode()
    throw(ArgumentError("mode must be :abs or :rel, got :$mode"))
end

_mode_label(mode::AbstractMode) = ":" * String(_mode_symbol(mode))
