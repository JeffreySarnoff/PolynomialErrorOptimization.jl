"""
    constraints.jl

Constraint-row and right-hand-side builders for the absolute and relative
optimization formulations.

These helpers encode the paper's `α(ω)` / `c(ω)` objects and their
relative-error variants. They are shared by initialisation, primal solves,
and exchange steps.
"""

# ---------------------------------------------------------------------------
# Constraint row α(ω)
# ---------------------------------------------------------------------------

"""
    α(ω::Index{T}, scheme::EvalScheme{T}) -> Vector{T}

Constraint row for the absolute-error formulation (paper Section 3.1).
Returns `(1, σ₀ π₀(t)ᵀ + Σᵢ σᵢ πᵢ(t)ᵀ)ᵀ ∈ ℝ^{n+2}` where `π₀(t) = (1, t, …, tⁿ)`.
"""
function α(ω::Index{T}, scheme::EvalScheme{T}) where T<:AbstractFloat
    out = Vector{T}(undef, scheme.n + 2)
    return α!(out, ω, scheme)
end

"""
    α!(out, ω, scheme) -> out

In-place variant of `α`. `out` must have length `scheme.n + 2`.
"""
function α!(out::AbstractVector{T}, ω::Index{T}, scheme::EvalScheme{T}) where T<:AbstractFloat
    n = scheme.n
    @argcheck length(out) == n + 2 DimensionMismatch(
        "α!: out length $(length(out)) ≠ n+2 = $(n+2)")
    @argcheck length(ω.σ) == scheme.k + 1 DimensionMismatch(
        "α!: σ length $(length(ω.σ)) ≠ k+1 = $(scheme.k + 1)")

    @inbounds out[1] = one(T)
    σ0 = T(ω.σ[1])
    @inbounds begin
        out[2] = σ0                         # σ₀ · t⁰
        tj = one(T)
        for j in 1:n
            tj *= ω.t
            out[j+2] = σ0 * tj            # σ₀ · tʲ
        end
        # Σᵢ σᵢ · πᵢ(t),  i = 1..k
        for i in 1:scheme.k
            σi = T(ω.σ[i+1])
            iszero(σi) && continue
            πi = scheme.π[i](ω.t)::AbstractVector{T}
            @argcheck length(πi) == n + 1 DimensionMismatch(
                "scheme.π[$i] returned length $(length(πi)), expected $(n+1)")
            for j in 0:n
                out[j+2] += σi * πi[j+1]
            end
        end
    end
    return out
end

"""
    c(ω::Index{T}, f) -> T

Right-hand side `c(ω) = σ₀ · f(t)` for the absolute-error formulation.
"""
@inline function c(ω::Index{T}, f) where T<:AbstractFloat
    return T(ω.σ[1]) * T(f(ω.t))
end

# ---------------------------------------------------------------------------
# Relative-error formulation (P^rel), paper eq. (15)
# ---------------------------------------------------------------------------

"""
    α_rel(ω::Index{T}, scheme::EvalScheme{T}, f) -> Vector{T}

Constraint row for the (P^rel) formulation: identical to `α` except the first
component is `f(t)` instead of `1.0`.
"""
function α_rel(ω::Index{T}, scheme::EvalScheme{T}, f) where T<:AbstractFloat
    out = α(ω, scheme)
    out[1] = T(f(ω.t))
    return out
end

function α_rel!(out::AbstractVector{T}, ω::Index{T}, scheme::EvalScheme{T}, f) where T<:AbstractFloat
    α!(out, ω, scheme)
    out[1] = T(f(ω.t))
    return out
end

"""
    c_rel(ω::Index{T}, f) -> T

For (P^rel) the right-hand side is the same as the absolute-error case.
"""
@inline c_rel(ω::Index{T}, f) where T<:AbstractFloat = c(ω, f)

# ---------------------------------------------------------------------------
# Relative-error-with-zero formulation (P^rel2), paper eq. (16)
# ---------------------------------------------------------------------------

"""
    α_relzero(ω::Index{T}, scheme::EvalScheme{T}, f, t_z, s) -> Vector{T}

Constraint row for (P^rel2):

    α̃_r(ω, t)  = (1, σ₀ π̃₀(t)ᵀ/f(t) + Σᵢ σᵢ πᵢ(t)ᵀ/f(t))ᵀ ∈ ℝ^{n+2-s}

where `π̃₀(t) = ((t-t_z)ˢ, …, (t-t_z)ⁿ)`. The polynomial recovered from the
solution then has its first `s` coefficients identically zero.
"""
function α_relzero(ω::Index{T}, scheme::EvalScheme{T}, f, t_z::Real, s::Int) where T<:AbstractFloat
    n = scheme.n
    @argcheck s ≥ 1 && s ≤ n ArgumentError(
        "zero order s must satisfy 1 ≤ s ≤ n=$n, got s=$s")

    dim = n + 2 - s
    out = Vector{T}(undef, dim)
    fval = T(f(ω.t))
    iszero(fval) && throw(DomainError(ω.t,
        "α_relzero: f(t) = 0 at t = $(ω.t); cannot divide."))
    invf = one(T) / fval

    @inbounds out[1] = one(T)
    σ0 = T(ω.σ[1])

    # σ₀ · (t - t_z)^j / f(t)  for j = s..n.
    Δ = ω.t - T(t_z)
    base = Δ^s
    @inbounds out[2] = σ0 * base * invf
    val = base
    @inbounds for j in (s+1):n
        val *= Δ
        out[j-s+2] = σ0 * val * invf
    end

    # Σᵢ σᵢ · πᵢ(t)[j+1] / f(t)  for j = s..n.
    @inbounds for i in 1:scheme.k
        σi = T(ω.σ[i+1])
        iszero(σi) && continue
        πi = scheme.π[i](ω.t)::AbstractVector{T}
        @argcheck length(πi) == n + 1 DimensionMismatch(
            "scheme.π[$i] returned length $(length(πi)), expected $(n+1)")
        for j in s:n
            out[j-s+2] += σi * πi[j+1] * invf
        end
    end
    return out
end

"""
    c_relzero(ω::Index{T}, f, t_z, s) -> T

Right-hand side for (P^rel2); since we are approximating the constant `1`,
this is just `σ₀`.
"""
@inline c_relzero(ω::Index{T}, f, t_z::Real, s::Int) where T<:AbstractFloat = T(ω.σ[1])

# ---------------------------------------------------------------------------
# Mode-dispatched constraint helpers
# ---------------------------------------------------------------------------

@inline constraint_dim(scheme::EvalScheme, ::AbsoluteMode) = scheme.n + 2
@inline constraint_dim(scheme::EvalScheme, ::RelativeMode) = scheme.n + 2
@inline constraint_dim(scheme::EvalScheme, mode::RelativeZeroMode) =
    scheme.n + 2 - mode.s

constraint_row(ω::Index{T}, scheme::EvalScheme{T},
    ::AbsoluteMode, f) where T<:AbstractFloat = α(ω, scheme)

constraint_row(ω::Index{T}, scheme::EvalScheme{T},
    ::RelativeMode, f) where T<:AbstractFloat = α_rel(ω, scheme, f)

constraint_row(ω::Index{T}, scheme::EvalScheme{T},
    mode::RelativeZeroMode, f) where T<:AbstractFloat =
    α_relzero(ω, scheme, f, mode.t_z, mode.s)

function constraint_row!(out::AbstractVector{T}, ω::Index{T},
    scheme::EvalScheme{T}, ::AbsoluteMode, f) where T<:AbstractFloat
    return α!(out, ω, scheme)
end

function constraint_row!(out::AbstractVector{T}, ω::Index{T},
    scheme::EvalScheme{T}, ::RelativeMode, f) where T<:AbstractFloat
    @argcheck f !== nothing ArgumentError("RelativeMode requires f !== nothing")
    return α_rel!(out, ω, scheme, f)
end

function constraint_row!(out::AbstractVector{T}, ω::Index{T},
    scheme::EvalScheme{T}, mode::RelativeZeroMode, f) where T<:AbstractFloat
    @argcheck f !== nothing ArgumentError("RelativeZeroMode requires f !== nothing")
    row = α_relzero(ω, scheme, f, mode.t_z, mode.s)
    @argcheck length(out) == length(row) DimensionMismatch(
        "constraint_row!: out length $(length(out)) ≠ row length $(length(row))")
    copyto!(out, row)
    return out
end

rhs_value(ω::Index{T}, f, ::AbsoluteMode) where T<:AbstractFloat = c(ω, f)
rhs_value(ω::Index{T}, f, ::RelativeMode) where T<:AbstractFloat = c_rel(ω, f)
rhs_value(ω::Index{T}, f, mode::RelativeZeroMode) where T<:AbstractFloat =
    c_relzero(ω, f, mode.t_z, mode.s)
