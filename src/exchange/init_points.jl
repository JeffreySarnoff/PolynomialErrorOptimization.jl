"""
    init_points.jl

Algorithm 4 of the paper: initialise the discretisation `ω⁽⁰⁾ ∈ Ω^{n+2}`
with Chebyshev nodes and Remez-style alternating signatures, and compute
the corresponding optimal dual `y⁽⁰⁾`.

By Lemma 1 of the paper, this initialisation:

* makes `{α(ωⱼ)}_{j=1}^{n+2}` a basis of `ℝ^{n+2}`, and
* yields a unique optimal dual `y⁽⁰⁾` with `yⱼ⁽⁰⁾ > 0` for all `j`.
"""

function _step_inside_interval(t::T, tl::T, tr::T, toward_right::Bool) where T<:AbstractFloat
    if toward_right
        moved = nextfloat(t)
        return moved <= tr ? moved : prevfloat(t)
    end
    moved = prevfloat(t)
    return moved >= tl ? moved : nextfloat(t)
end

function _avoid_relzero_singularity(t::T, tl::T, tr::T, f,
    mode::RelativeZeroMode{T}) where T<:AbstractFloat
    candidate = t
    toward_right = candidate <= mode.t_z
    for _ in 1:16
        value = T(f(candidate))
        if isfinite(value) && !iszero(value)
            return candidate
        end
        moved = _step_inside_interval(candidate, tl, tr, toward_right)
        moved == candidate && break
        candidate = moved
        toward_right = candidate <= mode.t_z
    end
    throw(DomainError(t,
        "init_points: could not place a RelativeZeroMode node away from a zero/singularity near t=$t"))
end

"""
    init_points(scheme, I) -> (ω, y)

Algorithm 4 of the paper. `ω` is a length-`n+2` `Vector{Index}` of Chebyshev
nodes paired with Remez-style alternating signatures `((-1)ʲ, 0, …, 0)`.
`y` is the optimal dual on that discretisation, obtained by solving

    Σⱼ yⱼ α(ωⱼ) = e₁    (paper eq. above (D), with z = e₁)

# Arguments
- `scheme::EvalScheme`
- `I::Tuple{T,T}` — `(tl, tr)` with `tl < tr`, where
  `T = fptype(scheme)`.

# Returns
`(ω::Vector{Index{T}}, y::Vector{T})`.
"""
function init_points(scheme::EvalScheme{T}, I::Tuple{T,T}) where T<:AbstractFloat
    tl, tr = I
    @argcheck tl < tr ArgumentError(
        "init_points: I = ($tl, $tr) must satisfy tl < tr")
    n = scheme.n
    m = n + 2

    ω = Vector{Index{T}}(undef, m)
    half_sum = (tl + tr) / 2
    half_diff = (tr - tl) / 2
    σlen = scheme.k + 1

    @inbounds for j in 1:m
        tj = half_sum + cospi(T(j - 1) / T(n + 1)) * half_diff
        σ = zeros(Int8, σlen)
        σ[1] = isodd(j) ? Int8(1) : Int8(-1)
        ω[j] = Index{T}(tj, σ)
    end

    # Build the (n+2)×(n+2) system  A · y = e₁   where  A[:,j] = α(ωⱼ).
    A = Matrix{T}(undef, m, m)
    @inbounds for j in 1:m
        col = view(A, :, j)
        α!(col, ω[j], scheme)
    end
    z = zeros(T, m)
    z[1] = one(T)

    y = solve_dense_system(A, z)
    return ω, y
end

"""
    init_points(scheme, I, mode_aux::AbstractMode; f = nothing, t_z = NaN, s = 0)
        -> (ω, y)

Mode-aware initialisation. Builds the appropriate constraint rows `α_rel` /
`α_relzero` instead of `α` when `mode_aux` is `RelativeMode` /
`RelativeZeroMode`.  Used by the relative-error drivers; not normally called
directly by users.
"""
function init_points(scheme::EvalScheme{T}, I::Tuple{T,T},
    mode::AbstractMode; f=nothing,
    t_z::Real=zero(T), s::Int=0) where T<:AbstractFloat
    if mode isa AbsoluteMode
        return init_points(scheme, I)
    end

    tl, tr = I
    @argcheck tl < tr ArgumentError(
        "init_points: I = ($tl, $tr) must satisfy tl < tr")

    @argcheck f !== nothing ArgumentError("$(typeof(mode)) requires f !== nothing")
    m = constraint_dim(scheme, mode)

    ω = Vector{Index{T}}(undef, m)
    half_sum = (tl + tr) / 2
    half_diff = (tr - tl) / 2
    σlen = scheme.k + 1

    @inbounds for j in 1:m
        tj = half_sum + cospi(T(j - 1) / T(max(m - 1, 1))) * half_diff
        if mode isa RelativeZeroMode
            tj = _avoid_relzero_singularity(tj, tl, tr, f, mode)
        end
        σ = zeros(Int8, σlen)
        σ[1] = isodd(j) ? Int8(1) : Int8(-1)
        ω[j] = Index{T}(tj, σ)
    end

    A = Matrix{T}(undef, m, m)
    @inbounds for j in 1:m
        constraint_row!(view(A, :, j), ω[j], scheme, mode, f)
    end
    z = zeros(T, m)
    z[1] = one(T)
    y = solve_dense_system(A, z)
    return ω, y
end
