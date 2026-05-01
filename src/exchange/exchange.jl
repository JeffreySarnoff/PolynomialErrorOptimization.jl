"""
    exchange.jl

Algorithm 7 of the paper: one-element exchange step.

Given a basis `ω = (ω_1, …, ω_{n+2})` with optimal dual `y > 0` and the new
candidate index `ωstar` from `FINDNEWINDEX`, perform:

1.  Solve  Σⱼ γⱼ α(ωⱼ) = α(ωstar)  for γ ∈ ℝ^{n+2}.
2.  j₀ = argmin_{j : γⱼ > 0}  yⱼ / γⱼ;   λ = y_{j₀} / γ_{j₀}.
3.  Update:  y′_⋆ = λ;  y′_j = yⱼ − λ γⱼ  for j ≠ j₀;  y′_{j₀} = 0 (dropped).
4.  Return  ω′ = (ω \\ {ω_{j₀}}) ∪ {ωstar}  and  y′  (length n+2).

By Lemma 4 the new dual `y′` is optimal for `(D′_{n+2})`, the dual objective
is non-decreasing (Lemma 5), and the basis property is preserved.
"""

"""
    ExchangeFailure(msg)

Thrown by `exchange` when no `γⱼ > 0` exists, indicating either Assumption 1
of the paper is violated or numerical breakdown of the basis.
"""
struct ExchangeFailure <: Exception
    msg::String
end
Base.showerror(io::IO, e::ExchangeFailure) = print(io, "ExchangeFailure: ", e.msg)

"""
    exchange(scheme, ω, y, ωstar;
             eps_γ = 1e-12) -> (ω′, y′)

Algorithm 7 of the paper.

Throws `ExchangeFailure` when no `γⱼ > 0` (Assumption 1 violated or
numerical breakdown).  `eps_γ` guards against spurious-positive `γⱼ` due to
round-off:  a candidate is considered positive iff
`γⱼ > eps_γ * maximum(abs, γ)`.
"""
function exchange(scheme::EvalScheme{T},
    ω::Vector{Index{T}},
    y::Vector{T},
    ωstar::Index{T};
    eps_γ::Real=1e-12) where T<:AbstractFloat
    m = scheme.n + 2
    @argcheck length(ω) == m DimensionMismatch(
        "exchange: length(ω) = $(length(ω)) ≠ n+2 = $m")
    @argcheck length(y) == m DimensionMismatch(
        "exchange: length(y) = $(length(y)) ≠ n+2 = $m")

    # 1. Solve A γ = α(ωstar),  A[:, j] = α(ωⱼ).
    A = Matrix{T}(undef, m, m)
    @inbounds for j in 1:m
        col = view(A, :, j)
        α!(col, ω[j], scheme)
    end
    rhs = α(ωstar, scheme)
    γ = solve_dense_system(A, rhs)

    # 2. j₀ = argmin yⱼ/γⱼ over γⱼ > eps_γ * max|γ|.
    γ_scale = maximum(abs, γ)
    threshold = T(eps_γ) * (γ_scale > zero(T) ? γ_scale : one(T))

    j0 = 0
    best_λ = T(Inf)
    @inbounds for j in 1:m
        if γ[j] > threshold
            λcand = y[j] / γ[j]
            if λcand < best_λ
                best_λ = λcand
                j0 = j
            end
        end
    end
    j0 == 0 && throw(ExchangeFailure(
        "exchange: no γⱼ > 0 found (max γⱼ = $(maximum(γ))). " *
        "Assumption 1 likely violated or scheme degenerate."))

    λ = best_λ

    # 3. Build new dual y′
    y_new = Vector{T}(undef, m)
    @inbounds for j in 1:m
        if j == j0
            y_new[j] = λ
        else
            y_new[j] = y[j] - λ * γ[j]
            # Floor tiny negatives to zero (round-off):
            if y_new[j] < zero(T) && y_new[j] > -T(eps_γ) * (abs(y[j]) + abs(λ * γ[j]) + one(T))
                y_new[j] = zero(T)
            end
        end
    end

    # 4. Build new discretisation ω′
    ω_new = copy(ω)
    ω_new[j0] = ωstar
    return ω_new, y_new
end

"""
    exchange(scheme, ω, y, ωstar, mode::AbstractMode; eps_γ = 1e-12) -> (ω′, y′)

Mode-aware variant: builds the constraint rows using `α_rel` / `α_relzero`
when the mode requires it.
"""
function exchange(scheme::EvalScheme{T},
    ω::Vector{Index{T}}, y::Vector{T},
    ωstar::Index{T}, mode::AbstractMode;
    eps_γ::Real=1e-12,
    f=nothing) where T<:AbstractFloat
    if mode isa AbsoluteMode
        return exchange(scheme, ω, y, ωstar; eps_γ=eps_γ)
    end

    @argcheck f !== nothing ArgumentError("$(typeof(mode)) requires f !== nothing")
    m = constraint_dim(scheme, mode)
    @argcheck length(ω) == m DimensionMismatch(
        "exchange (mode): length(ω) = $(length(ω)) ≠ $m")
    @argcheck length(y) == m DimensionMismatch(
        "exchange (mode): length(y) = $(length(y)) ≠ $m")

    A = Matrix{T}(undef, m, m)
    @inbounds for j in 1:m
        constraint_row!(view(A, :, j), ω[j], scheme, mode, f)
    end

    rhs = constraint_row(ωstar, scheme, mode, f)
    γ = solve_dense_system(A, rhs)

    γ_scale = maximum(abs, γ)
    threshold = T(eps_γ) * (γ_scale > zero(T) ? γ_scale : one(T))
    j0 = 0
    best_λ = T(Inf)
    @inbounds for j in 1:m
        if γ[j] > threshold
            λcand = y[j] / γ[j]
            if λcand < best_λ
                best_λ = λcand
                j0 = j
            end
        end
    end
    j0 == 0 && throw(ExchangeFailure(
        "exchange (mode): no γⱼ > 0 found in mode $(mode)."))

    λ = best_λ
    y_new = Vector{T}(undef, m)
    @inbounds for j in 1:m
        if j == j0
            y_new[j] = λ
        else
            y_new[j] = y[j] - λ * γ[j]
            if y_new[j] < zero(T) && y_new[j] > -T(eps_γ) * (abs(y[j]) + abs(λ * γ[j]) + one(T))
                y_new[j] = zero(T)
            end
        end
    end
    ω_new = copy(ω)
    ω_new[j0] = ωstar
    return ω_new, y_new
end
