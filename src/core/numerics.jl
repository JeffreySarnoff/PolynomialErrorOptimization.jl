"""
    core/numerics.jl

Shared helpers used throughout `PolynomialErrorOptimization`.

Provides:

* `monomial_dot(a, t)`                   — Horner evaluation of `a₀ + a₁t + … + aₙtⁿ`.
* `dot_view(a, π)`                       — bounds-checked dot product.
* `signum_int8(x)`                       — `±1` / `0` as `Int8`, used to build signatures.
"""

# ---------------------------------------------------------------------------
# Sign helper
# ---------------------------------------------------------------------------

"""
    signum_int8(x) -> Int8

Sign of `x` returned as `Int8`. Returns `Int8(0)` when `x` is exactly zero.
"""
@inline function signum_int8(x::Real)
    return x > 0 ? Int8(1) : (x < 0 ? Int8(-1) : Int8(0))
end

# ---------------------------------------------------------------------------
# Numerical kernels
# ---------------------------------------------------------------------------

"""
    monomial_dot(a, t) -> T

Evaluate `Σⱼ aⱼ tʲ` using Horner's rule. `a` is in natural order
(`a[1] = a₀`, `a[end] = aₙ`).
"""
@inline function monomial_dot(a::AbstractVector{T}, t::T) where T<:AbstractFloat
    n_plus_1 = length(a)
    n_plus_1 == 0 && return zero(T)
    @inbounds s = a[n_plus_1]
    @inbounds for i in (n_plus_1-1):-1:1
        s = muladd(s, t, a[i])
    end
    return s
end

"""
    dot_view(a, π) -> T

Compute `aᵀ π`. Both vectors must have equal length; uses `muladd` to enable
FMA on supported CPUs.
"""
@inline function dot_view(a::AbstractVector{T}, π::AbstractVector{T}) where T<:AbstractFloat
    @argcheck length(a) == length(π) DimensionMismatch(
        "dot_view: lengths $(length(a)) ≠ $(length(π))")
    s = zero(T)
    @inbounds for i in eachindex(a)
        s = muladd(a[i], π[i], s)
    end
    return s
end
