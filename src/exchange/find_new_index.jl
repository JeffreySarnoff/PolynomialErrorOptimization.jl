"""
    find_new_index.jl

Algorithm 6 of the paper: given the current primal candidate `(ā, a)`,
locate

    ωstar = argmax_{ω ∈ Ω}  ( c(ω) − α(ω)ᵀ x )

and the corresponding signature.  In the absolute-error case this reduces
to

    tstar = argmax_{t ∈ I}  |aᵀπ₀(t) − f(t)| + Σᵢ |aᵀπᵢ(t)|
    σ⋆₀ = − sign(aᵀπ₀(tstar) − f(tstar))
    σ⋆ᵢ = − sign(aᵀπᵢ(tstar)),   i = 1, …, k.

The maximum value `astar` is a verified upper bound on the true total error
of the current polynomial; together with `ā` from `solve_primal` it yields
the enclosure  `ā ≤ ε⋆ ≤ astar`  of paper Theorem 2.

For relative modes (paper Section 5) the per-`t` objective is divided by
`|f(t)|`, since (P^rel) is (P_general) with the error variable scaled by
`|f|`.
"""

# ---------------------------------------------------------------------------
# Absolute-error case
# ---------------------------------------------------------------------------

"""
    find_new_index(f, scheme, I, a;
                   strategy = default_strategy(scheme)) -> (ωstar, astar)

Algorithm 6 (absolute-error mode).
"""
function find_new_index(f, scheme::EvalScheme{T},
    I::Tuple{T,T},
    a::AbstractVector{T};
    strategy::SearchStrategy=default_strategy(scheme)) where T<:AbstractFloat
    @argcheck length(a) == scheme.n + 1 DimensionMismatch(
        "find_new_index: length(a) = $(length(a)) ≠ n+1 = $(scheme.n + 1)")

    objective = let f = f, scheme = scheme, a = a
        function (t::T)
            v0 = monomial_dot(a, t) - T(f(t))
            s = abs(v0)
            @inbounds for i in 1:scheme.k
                πi = scheme.π[i](t)::AbstractVector{T}
                s += abs(dot_view(a, πi))
            end
            return s
        end
    end

    tstar, astar = locate_maximum(objective, I, strategy)
    σ = _build_signature(f, scheme, a, tstar)
    return Index(tstar, σ), astar
end

# ---------------------------------------------------------------------------
# Mode-aware variant
# ---------------------------------------------------------------------------

"""
    find_new_index(f, scheme, I, a, mode::AbstractMode;
                   strategy = default_strategy(scheme)) -> (ωstar, astar)

Mode-aware variant.  For relative modes the objective is divided by `|f(t)|`
so the returned `astar` is the relative total error.
"""
function find_new_index(f, scheme::EvalScheme{T},
    I::Tuple{T,T},
    a::AbstractVector{T},
    mode::AbstractMode;
    strategy::SearchStrategy=default_strategy(scheme)) where T<:AbstractFloat

    @argcheck length(a) == scheme.n + 1 DimensionMismatch(
        "find_new_index: length(a) = $(length(a)) ≠ n+1 = $(scheme.n + 1)")

    if mode isa AbsoluteMode
        return find_new_index(f, scheme, I, a; strategy=strategy)
    end

    objective = let f = f, scheme = scheme, a = a
        function (t::T)
            ft = T(f(t))
            v0 = monomial_dot(a, t) - ft
            s = abs(v0)
            @inbounds for i in 1:scheme.k
                πi = scheme.π[i](t)::AbstractVector{T}
                s += abs(dot_view(a, πi))
            end
            absft = abs(ft)
            if absft == zero(T)
                return T(Inf)
            end
            return s / absft
        end
    end

    tstar, astar = locate_maximum(objective, I, strategy)
    σ = _build_signature(f, scheme, a, tstar)
    return Index(tstar, σ), astar
end

# ---------------------------------------------------------------------------
# Signature reconstruction (paper Algorithm 6 lines 3–4)
# ---------------------------------------------------------------------------

@inline function _build_signature(f, scheme::EvalScheme{T},
    a::AbstractVector{T}, tstar::T) where T<:AbstractFloat
    σ = Vector{Int8}(undef, scheme.k + 1)
    v0 = monomial_dot(a, tstar) - T(f(tstar))
    σ[1] = -signum_int8(v0)
    @inbounds for i in 1:scheme.k
        πi = scheme.π[i](tstar)::AbstractVector{T}
        σ[i+1] = -signum_int8(dot_view(a, πi))
    end
    return σ
end
