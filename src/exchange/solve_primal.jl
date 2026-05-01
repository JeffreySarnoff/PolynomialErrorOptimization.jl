"""
    solve_primal.jl

Algorithm 5 of the paper: given the current discretisation `ω`, solve

    α(ωⱼ)ᵀ x = c(ωⱼ),   j = 1, …, n+2

for `x = (ā, a) ∈ ℝ^{n+2}`.  By Lemma 2 this is the optimal primal solution
of the discretised LP `(P_{n+2})`.

The system is dense and small (size `n+2`), so the solver backend is routed
through `solve_dense_system` (LinearSolve with LU by default).
"""

"""
    solve_primal(f, scheme, ω) -> (ā, a)

Algorithm 5 of the paper.  Returns the discrete optimum `ā` and the
polynomial coefficient vector `a` of length `n+1` (natural order:
`a[1] = a₀`, `a[end] = aₙ`).
"""
function solve_primal(f, scheme::EvalScheme{T}, ω::Vector{Index{T}}) where T<:AbstractFloat
    m = scheme.n + 2
    @argcheck length(ω) == m DimensionMismatch(
        "solve_primal: length(ω) = $(length(ω)) ≠ n+2 = $m")

    A = Matrix{T}(undef, m, m)
    rhs = Vector{T}(undef, m)
    @inbounds for j in 1:m
        row = view(A, j, :)
        α!(row, ω[j], scheme)               # row j is α(ωⱼ)ᵀ
        rhs[j] = c(ω[j], f)
    end
    x = solve_dense_system(A, rhs)
    return x[1], x[2:end]
end

# ---------------------------------------------------------------------------
# Mode-aware version
# ---------------------------------------------------------------------------

"""
    solve_primal(f, scheme, ω, mode::AbstractMode) -> (ā, a)

Mode-aware variant. For `RelativeMode` and `RelativeZeroMode` the row `α(ω)`
and right-hand side `c(ω)` are replaced by their mode-specific counterparts.

For `RelativeZeroMode`, the returned `a` is padded with `s` leading zeros so
the polynomial is still degree-`n` in standard form.
"""
function solve_primal(f, scheme::EvalScheme{T}, ω::Vector{Index{T}},
    mode::AbstractMode) where T<:AbstractFloat
    if mode isa AbsoluteMode
        return solve_primal(f, scheme, ω)
    elseif mode isa RelativeZeroMode
        x = _solve_mode_system(f, scheme, ω, mode)
        ā = x[1]

        n = scheme.n
        b = zeros(T, n + 1)
        @inbounds for j in mode.s:n
            b[j+1] = x[j-mode.s+2]
        end
        return ā, b
    elseif mode isa RelativeMode
        x = _solve_mode_system(f, scheme, ω, mode)
        return x[1], x[2:end]
    end
    error("solve_primal: unhandled mode $(mode)")
end

function _solve_mode_system(f, scheme::EvalScheme{T}, ω::Vector{Index{T}},
    mode::AbstractMode) where T<:AbstractFloat
    m = constraint_dim(scheme, mode)
    @argcheck length(ω) == m DimensionMismatch(
        "solve_primal: length(ω) = $(length(ω)) ≠ expected size $m")

    A = Matrix{T}(undef, m, m)
    rhs = Vector{T}(undef, m)
    @inbounds for j in 1:m
        constraint_row!(view(A, j, :), ω[j], scheme, mode, f)
        rhs[j] = rhs_value(ω[j], f, mode)
    end
    return solve_dense_system(A, rhs)
end
