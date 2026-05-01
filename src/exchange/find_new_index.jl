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
    elseif mode isa RelativeZeroMode{T}
        return _find_new_index_relzero(f, scheme, I, a, mode; strategy=strategy)
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

function _relzero_step_inside_interval(t::T, tl::T, tr::T, toward_right::Bool) where T<:AbstractFloat
    if toward_right
        moved = nextfloat(t)
        return moved <= tr ? moved : prevfloat(t)
    end
    moved = prevfloat(t)
    return moved >= tl ? moved : nextfloat(t)
end

function _regularize_relzero_sample(t::T, I::Tuple{T,T}, f,
    mode::RelativeZeroMode{T}) where T<:AbstractFloat
    tl, tr = I
    candidate = t
    toward_right = candidate <= mode.t_z
    for _ in 1:16
        value = T(f(candidate))
        if isfinite(value) && !iszero(value)
            return candidate
        end
        moved = _relzero_step_inside_interval(candidate, tl, tr, toward_right)
        moved == candidate && break
        candidate = moved
        toward_right = candidate <= mode.t_z
    end
    return candidate
end

function _shifted_to_monomial_search(a::AbstractVector{T}, shift::T) where T<:AbstractFloat
    n = length(a) - 1
    mono = zeros(T, n + 1)
    @inbounds for j in 0:n
        aj = a[j+1]
        if !iszero(aj)
            for k in 0:j
                mono[k+1] += aj * T(binomial(j, k)) * ((-shift)^(j - k))
            end
        end
    end
    return mono
end

_relzero_search_coefficients(a::AbstractVector{T}, mode::RelativeZeroMode{T}) where T<:AbstractFloat =
    iszero(mode.t_z) ? collect(a) : _shifted_to_monomial_search(a, mode.t_z)

function _find_new_index_relzero(f, scheme::EvalScheme{T},
    I::Tuple{T,T},
    a::AbstractVector{T},
    mode::RelativeZeroMode{T};
    strategy::SearchStrategy=default_strategy(scheme)) where T<:AbstractFloat

    coeffs = _relzero_search_coefficients(a, mode)
    objective = let f = f, scheme = scheme, coeffs = coeffs, I = I, mode = mode
        function (t::T)
            ts = _regularize_relzero_sample(t, I, f, mode)
            ft = T(f(ts))
            absft = abs(ft)
            absft == zero(T) && return T(Inf)

            v0 = monomial_dot(coeffs, ts) - ft
            s = abs(v0)
            @inbounds for i in 1:scheme.k
                πi = scheme.π[i](ts)::AbstractVector{T}
                s += abs(dot_view(coeffs, πi))
            end
            return s / absft
        end
    end

    tstar_raw, astar = locate_maximum(objective, I, strategy)
    tstar = _regularize_relzero_sample(tstar_raw, I, f, mode)
    σ = _build_signature(f, scheme, coeffs, tstar, mode)
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

@inline _relative_signature_scale(ft::T) where T<:AbstractFloat = signum_int8(ft)

function _build_signature(f, scheme::EvalScheme{T},
    a::AbstractVector{T}, tstar::T,
    ::RelativeMode) where T<:AbstractFloat
    return _build_signature(f, scheme, a, tstar)
end

function _build_signature(f, scheme::EvalScheme{T},
    a::AbstractVector{T}, tstar::T,
    ::RelativeZeroMode{T}) where T<:AbstractFloat
    ft = T(f(tstar))
    @argcheck !iszero(ft) DomainError(tstar,
        "find_new_index: RelativeZeroMode signature requires f(tstar) != 0")
    scale = _relative_signature_scale(ft)

    σ = Vector{Int8}(undef, scheme.k + 1)
    v0 = monomial_dot(a, tstar) - ft
    σ[1] = -signum_int8(v0) * scale
    @inbounds for i in 1:scheme.k
        πi = scheme.π[i](tstar)::AbstractVector{T}
        σ[i+1] = -signum_int8(dot_view(a, πi)) * scale
    end
    return σ
end
