"""
    driver.jl

Top-level Algorithm 3 of the paper (EVAL_APPROX_OPTIMIZE) and the relative-
error variants of Section 5.

The public entry points are:

* `eval_approx_optimize`              — absolute-error mode (P_general).
* `eval_approx_optimize_relative`     — (P^rel), `f` does not vanish on `I`.
* `eval_approx_optimize_relative_zero` — (P^rel2), `f` has a finite-order zero.
"""

# ---------------------------------------------------------------------------
# Driver result and configuration
# ---------------------------------------------------------------------------

"""
    ResultBasis{T<:AbstractFloat}

Explicit basis metadata for an `OptimResult`.

# Fields
* `coefficient_basis::Symbol` — basis used by `result.poly`.
* `solution_basis::Symbol`    — basis used by `solution_coefficients(result)`.
* `shift::T`                  — origin shift for a shifted basis.
* `zero_order::Int`           — known zero order used by `RelativeZeroMode`.
"""
struct ResultBasis{T<:AbstractFloat}
    coefficient_basis::Symbol
    solution_basis::Symbol
    shift::T
    zero_order::Int
end

"""
    OptimResult{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}

Return value of `eval_approx_optimize` (and the relative variants).

# Fields
* `poly::Polynomial{TargetT,:t}`  — the optimal degree-n polynomial converted
                                    to the requested target coefficient type.
* `total_error::ComputeT`         — `astar`, a verified upper bound on the true
                                    total error (approximation + evaluation).
* `discrete_error::ComputeT`      — `ā`, the discrete optimum at termination.
* `iterations::Int`               — number of EXCHANGE iterations performed.
* `discretization::Vector{Index{ComputeT}}` — final ω.
* `dual::Vector{ComputeT}`        — final y (length `n+2`, all ≥ 0).
* `converged::Bool`               — `true` iff the τ-tolerance was met.
* `basis::ResultBasis{ComputeT}`  — explicit basis metadata for the stored
                                    polynomial and the optimisation solve.
* `solution_coeffs`               — coefficient vector in `basis.solution_basis`
                                    (natural order, converted to `TargetT`).

The contract of paper Theorem 2 is

    discrete_error  ≤  ε⋆  ≤  total_error  ≤  (1 + τ) · discrete_error.
"""
struct OptimResult{TargetT<:AbstractFloat,ComputeT<:AbstractFloat}
    poly                          # Polynomial{TargetT,:t} — kept untyped here
    total_error::ComputeT
    discrete_error::ComputeT
    iterations::Int
    discretization::Vector{Index{ComputeT}}
    dual::Vector{ComputeT}
    converged::Bool
    basis::ResultBasis{ComputeT}
    solution_coeffs::Vector{TargetT}
end

function Base.show(io::IO, r::OptimResult)
    print(io, "OptimResult(",
        "total_error=", r.total_error,
        ", iter=", r.iterations,
        r.basis.solution_basis === r.basis.coefficient_basis ? "" :
        ", basis=:" * String(r.basis.solution_basis) * "->:" *
        String(r.basis.coefficient_basis),
        ", converged=", r.converged, ")")
end

"""
    basis_info(result::OptimResult) -> ResultBasis

Return the basis metadata for an optimization result.
"""
basis_info(r::OptimResult) = r.basis

"""
    solution_coefficients(result::OptimResult)

Return the coefficient vector in the basis recorded by `basis_info(result)`.
For most modes this matches `Polynomials.coeffs(result.poly)`. For
`RelativeZeroMode` with a nonzero shift it preserves the shifted-basis
coefficients used by the optimisation solve.
"""
solution_coefficients(r::OptimResult) = copy(r.solution_coeffs)

function _result_basis(mode::AbstractMode, ::Type{T}) where {T<:AbstractFloat}
    return ResultBasis{T}(:monomial, :monomial, zero(T), 0)
end

function _result_basis(mode::RelativeZeroMode{T}, ::Type{T}) where {T<:AbstractFloat}
    solution_basis = iszero(mode.t_z) ? :monomial : :shifted
    return ResultBasis{T}(:monomial, solution_basis, mode.t_z, mode.s)
end

function _shifted_to_monomial(a::AbstractVector{T}, shift::T) where {T<:AbstractFloat}
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

function _result_polynomial(a::AbstractVector{T}, mode::AbstractMode, ::Type{TargetT}) where {T<:AbstractFloat,TargetT<:AbstractFloat}
    return Polynomials.Polynomial(TargetT.(a), :t)
end

function _result_polynomial(a::AbstractVector{T}, mode::RelativeZeroMode{T}, ::Type{TargetT}) where {T<:AbstractFloat,TargetT<:AbstractFloat}
    coeffs = iszero(mode.t_z) ? a : _shifted_to_monomial(a, mode.t_z)
    return Polynomials.Polynomial(TargetT.(coeffs), :t)
end

"""
    ConvergenceFailure{T<:AbstractFloat}(msg, ω, y, ā, astar)

Thrown by `eval_approx_optimize` when `max_iter` is exceeded without reaching
the requested tolerance. Carries the partial state for inspection.
"""
struct ConvergenceFailure{T<:AbstractFloat} <: Exception
    msg::String
    ω::Vector{Index{T}}
    y::Vector{T}
    ā::T
    astar::T
end
function Base.showerror(io::IO, e::ConvergenceFailure)
    print(io, "ConvergenceFailure: ", e.msg,
        "  (ā=", e.ā, ", astar=", e.astar, ")")
end

struct DriveConfig{T<:AbstractFloat}
    τ::T
    max_iter::Int
    strategy::SearchStrategy
    verbose::Bool
    function DriveConfig(τ::T, max_iter::Integer,
        strategy::SearchStrategy, verbose::Bool) where T<:AbstractFloat
        @argcheck τ > zero(T) ArgumentError("τ must be > 0, got $τ")
        @argcheck max_iter ≥ 0 ArgumentError("max_iter must be ≥ 0, got $max_iter")
        return new{T}(τ, Int(max_iter), strategy, verbose)
    end
end

"""
    eval_approx_optimize(f, n, I, scheme;
                         τ = 1e-3,
                         max_iter = 100,
                         strategy = default_strategy(scheme),
                         verbose = false,
                         target_type = fptype(scheme)) -> OptimResult

Algorithm 3 of the paper.  Compute a degree-`n` polynomial that minimises

    max_{t ∈ I}  ( |f(t) − p(t)|  +  θ(a, t) )

where `θ` is the linearised evaluation-error bound carried by `scheme`.

The returned polynomial satisfies the Theorem 2 enclosure:

    ε⋆  ≤  result.total_error  ≤  (1 + τ) · ε⋆

where `ε⋆` is the optimum of `(P_general)`.

# Arguments
- `f`        : `Function`. `f(t::T) -> Real`, where `T = fptype(scheme)`.
               Treated as a black box.
- `n::Int`   : non-negative degree.
- `I`        : `(tl, tr)` with `tl < tr`.
- `scheme`   : `EvalScheme` describing the evaluation scheme.

# Keyword arguments
- `τ`        : convergence tolerance (paper notation), `τ > 0`.
- `max_iter` : safety cap; throws `ConvergenceFailure` if exceeded.
- `strategy` : `SearchStrategy` for `find_new_index`.
- `verbose`  : log per-iteration `ā`, `astar`, `ratio` to `stdout`.
- `target_type` : floating-point type for returned polynomial coefficients.
                  Internal computation uses `fptype(scheme)`.

# Returns
`OptimResult` (see `?OptimResult`).
"""
function eval_approx_optimize(f, n::Integer,
    I::Tuple{<:Real,<:Real},
    scheme::EvalScheme{T};
    τ::Real=1e-3,
    max_iter::Integer=100,
    strategy::SearchStrategy=default_strategy(scheme),
    verbose::Bool=false,
    target_type::Type{<:AbstractFloat}=T) where T<:AbstractFloat
    cfg = DriveConfig(T(τ), Int(max_iter), strategy, verbose)
    return _drive(f, Int(n), (T(I[1]), T(I[2])), scheme,
        AbsoluteMode(), cfg, target_type)
end

"""
    eval_approx_optimize_relative(f, n, I, scheme;
                                  τ = 1e-3, max_iter = 100,
                                  strategy = default_strategy(scheme),
                                  verbose = false,
                                  target_type = fptype(scheme)) -> OptimResult

Relative-error formulation `(P^rel)` of Section 5; valid when `f` does not
vanish anywhere in `I`.  The total-error bound is interpreted relatively:

    max_{t ∈ I}  ( |f(t) − p(t)|  +  θ(a, t) ) / |f(t)|.
"""
function eval_approx_optimize_relative(f, n::Integer,
    I::Tuple{<:Real,<:Real},
    scheme::EvalScheme{T};
    τ::Real=1e-3,
    max_iter::Integer=100,
    strategy::SearchStrategy=default_strategy(scheme),
    verbose::Bool=false,
    target_type::Type{<:AbstractFloat}=T) where T<:AbstractFloat
    tl, tr = T(I[1]), T(I[2])
    _check_nonvanishing(f, tl, tr, T)
    cfg = DriveConfig(T(τ), Int(max_iter), strategy, verbose)
    return _drive(f, Int(n), (tl, tr), scheme,
        RelativeMode(), cfg, target_type)
end

"""
    eval_approx_optimize_relative_zero(f, n, I, scheme;
                                       t_z, s = 1,
                                       τ = 1e-3, max_iter = 100,
                                       strategy = default_strategy(scheme),
                                       verbose = false,
                                       target_type = fptype(scheme)) -> OptimResult

Relative-error formulation `(P^rel2)` of Section 5; valid when `f` has a
zero of finite order `s` at `t_z ∈ I`. The polynomial returned has its first
`s` coefficients identically zero, so `p(t)/(t-t_z)^s` is regular at `t_z`.
"""
function eval_approx_optimize_relative_zero(f, n::Integer,
    I::Tuple{<:Real,<:Real},
    scheme::EvalScheme{T};
    t_z::Real,
    s::Integer=1,
    τ::Real=1e-3,
    max_iter::Integer=100,
    strategy::SearchStrategy=default_strategy(scheme),
    verbose::Bool=false,
    target_type::Type{<:AbstractFloat}=T) where T<:AbstractFloat
    n = Int(n)
    s = Int(s)
    @argcheck s ≥ 1 && s ≤ n ArgumentError(
        "RelativeZeroMode requires 1 ≤ s ≤ n; got s=$s, n=$n")
    tl, tr = T(I[1]), T(I[2])
    _check_relzero_structure(f, tl, tr, T(t_z), s, T)
    cfg = DriveConfig(T(τ), Int(max_iter), strategy, verbose)
    return _drive(f, n, (tl, tr), scheme,
        RelativeZeroMode{T}(t_z, s), cfg, target_type)
end

# ---------------------------------------------------------------------------
# Internal core driver — shared between absolute and relative modes
# ---------------------------------------------------------------------------

function _drive(f, n::Int, I::Tuple{T,T},
    scheme::EvalScheme{T}, mode::AbstractMode,
    cfg::DriveConfig{T},
    ::Type{TargetT}=T) where {T<:AbstractFloat,TargetT<:AbstractFloat}
    @argcheck n ≥ 0 ArgumentError("n must be ≥ 0, got $n")
    @argcheck I[1] < I[2] ArgumentError("I = $I must satisfy I[1] < I[2]")
    @argcheck scheme.n == n ArgumentError(
        "scheme.n ($(scheme.n)) ≠ n ($n)")

    τ = cfg.τ
    max_iter = cfg.max_iter
    strategy = cfg.strategy
    verbose = cfg.verbose

    # ---------- Initialisation (paper lines 1–3) ----------
    ω, y = init_points(scheme, I, mode; f=f)
    ā, a = solve_primal(f, scheme, ω, mode)
    ωstar, astar = find_new_index(f, scheme, I, a, mode; strategy=strategy)

    if verbose
        Printf.@printf("iter %3d  ā = %.6e  astar = %.6e  ratio = %.6f\n",
            0, ā, astar, astar / max(abs(ā), eps(T)))
    end

    is_converged(ā, astar) = astar ≤ (1 + τ) * max(ā, eps(T))

    # ---------- Main loop (paper lines 5–10) ----------
    iter = 0
    converged = is_converged(ā, astar)
    while !converged
        if iter ≥ max_iter
            throw(ConvergenceFailure(
                "_drive: did not reach tolerance τ=$τ in $max_iter iterations.",
                ω, y, ā, astar))
        end
        iter += 1

        ω, y = exchange(scheme, ω, y, ωstar, mode; f=f)
        ā, a = solve_primal(f, scheme, ω, mode)
        ωstar, astar = find_new_index(f, scheme, I, a, mode; strategy=strategy)

        if verbose
            Printf.@printf("iter %3d  ā = %.6e  astar = %.6e  ratio = %.6f\n",
                iter, ā, astar, astar / max(abs(ā), eps(T)))
        end

        converged = is_converged(ā, astar)
    end

    basis = _result_basis(mode, T)
    poly = _result_polynomial(a, mode, TargetT)
    return OptimResult{TargetT,T}(
        poly,
        astar,
        ā,
        iter,
        ω,
        y,
        true,
        basis,
        TargetT.(a))
end

# ---------------------------------------------------------------------------
# Pre-flight check for RelativeMode
# ---------------------------------------------------------------------------

function _check_nonvanishing(f, tl::T, tr::T, ::Type{T};
    samples::Int=64) where T<:AbstractFloat
    v = abs(T(f(tl)))
    iszero(v) && throw(DomainError(tl,
        "_check_nonvanishing: f(tl) = 0; consider eval_approx_optimize_relative_zero."))
    minabs = v
    @inbounds for k in 1:samples
        t = tl + (tr - tl) * T(k) / T(samples)
        fv = abs(T(f(t)))
        iszero(fv) && throw(DomainError(t,
            "_check_nonvanishing: f vanishes inside I (at t=$t); " *
            "use eval_approx_optimize_relative_zero."))
        minabs = min(minabs, fv)
    end
    return minabs
end

function _relzero_probe_delta(tl::T, tr::T, t_z::T) where T<:AbstractFloat
    scale = max(one(T), abs(tl), abs(tr), abs(t_z), abs(tr - tl))
    return sqrt(eps(T)) * scale
end

function _quotient_sample_points(tl::T, tr::T, t_z::T, samples::Int) where T<:AbstractFloat
    points = T[]
    push!(points, tl)
    if tl < t_z < tr
        δ = _relzero_probe_delta(tl, tr, t_z)
        left = max(tl, t_z - δ)
        right = min(tr, t_z + δ)
        left > tl && push!(points, left)
        right < tr && push!(points, right)
    end
    @inbounds for k in 1:(samples-1)
        t = tl + (tr - tl) * T(k) / T(samples)
        t == t_z && continue
        push!(points, t)
    end
    push!(points, tr)
    return unique(points)
end

function _check_relzero_structure(f, tl::T, tr::T, t_z::T, s::Int, ::Type{T};
    samples::Int=64) where T<:AbstractFloat
    @argcheck tl ≤ t_z ≤ tr ArgumentError(
        "RelativeZeroMode requires t_z=$t_z to lie inside I=($tl, $tr)")

    seen_sign = Int8(0)
    minabs = T(Inf)
    for t in _quotient_sample_points(tl, tr, t_z, samples)
        Δ = t - t_z
        iszero(Δ) && continue
        q = T(f(t)) / (Δ^s)
        isfinite(q) || throw(DomainError(t,
            "RelativeZeroMode quotient f(t)/(t-t_z)^s is not finite at t=$t"))
        iszero(q) && throw(DomainError(t,
            "RelativeZeroMode quotient vanishes at sampled point t=$t; check t_z and s or use absolute mode"))

        qsign = signum_int8(q)
        if seen_sign == Int8(0)
            seen_sign = qsign
        elseif qsign != seen_sign
            throw(DomainError(t,
                "RelativeZeroMode quotient changes sign on sampled points; check t_z and s or use absolute mode"))
        end
        minabs = min(minabs, abs(q))
    end
    return minabs
end
