# Using LinearSolve.jl in PolynomialErrorOptimization

## Implementation status

LinearSolve.jl is now integrated in this repository.

Implemented pieces:

- `LinearSolve` added to `Project.toml` dependencies.
- Central helper added in `src/core/linear_solve_backend.jl`:
    `solve_dense_system(A, b; alg=LinearSolve.LUFactorization())`.
- `src/exchange/solve_primal.jl` now uses `solve_dense_system(...)`.
- `src/exchange/exchange.jl` now uses `solve_dense_system(...)`.
- `src/exchange/init_points.jl` now uses `solve_dense_system(...)`.

## Why consider LinearSolve.jl here

This project solved dense linear systems directly with Base Julia syntax.
That was already idiomatic Julia for small dense systems. LinearSolve.jl was
added to centralize solver policy and improve maintainability.

LinearSolve.jl is useful here because it gives:

1. A single, explicit solver interface across dense/sparse/direct/iterative methods.
2. Easy experimentation with solver algorithms without rewriting model code.
3. Better control over tolerances, diagnostics, and numerical behavior.
4. Optional future performance wins when matrix structure changes.

In short: Base was a strong baseline; LinearSolve now gives flexibility and control from a single backend entry point.

## Where it is used in this codebase

Current call sites:

1. src/exchange/solve_primal.jl
- absolute mode and relative modes use `solve_dense_system(A, rhs)`.

2. src/exchange/exchange.jl
- absolute and mode-aware variants use `solve_dense_system(A, rhs)`.

3. src/exchange/init_points.jl
- absolute and mode-aware initialization solve `A*y = z` via
    `solve_dense_system(A, z)`.

No algorithmic behavior changes were required; this is a backend routing change.

## Integration pattern used

The repository uses one helper to centralize solver setup while keeping
algorithm code in place.

~~~julia
using LinearSolve

function solve_dense_system(A::AbstractMatrix{T}, b::AbstractVector{T};
        alg=LinearSolve.LUFactorization()) where {T<:AbstractFloat}
        prob = LinearSolve.LinearProblem(A, b)
        sol = LinearSolve.solve(prob, alg)
    return sol.u
end
~~~

Then replace local solves:

~~~julia
# before
x = A \ rhs

# after
x = solve_dense_system(A, rhs)
~~~

This keeps solver call sites readable and gives one stable place to adjust
solver policy.

## Good default algorithms for this project

For current matrix sizes and dense matrices:

1. LUFactorization()
- Closest semantic match to A \ rhs for generic dense systems.
- Best default starting point.

2. QRFactorization()
- More robust if you hit near-rank-deficient systems.
- Usually a bit more expensive than LU.

3. SVDFactorization()
- Most robust for ill-conditioned cases.
- Most expensive; useful for diagnostics/fallback paths.

## Practical advantages in this project

### 1) Solver choice becomes a parameter

You can evolve toward a keyword such as `solver_alg` in top-level drivers
without rewriting internals.

### 2) Better failure handling

LinearSolve returns a richer solution object, so you can inspect status and condition-related warnings more directly.

### 3) Cleaner experimentation

For research work, it is valuable to quickly compare LU vs QR vs SVD in difficult intervals or relative modes.

### 4) Easier future transition to sparse/iterative methods

If you later change basis construction and matrix structure, LinearSolve reduces migration effort.

## Suggested phased adoption

Phase 1 is complete in this repository.

Next sensible phases:

1. Phase 2 (robustness option)
- add optional fallback policy (for example LU then QR),
- keep LU default for backward-compatible behavior.

2. Phase 3 (research mode)
- expose solver algorithm selection from public driver APIs for benchmarking.

## Caveats

1. For very small dense systems, Base A \ rhs is already excellent.
- Do not expect immediate speedups from LinearSolve alone.

2. The main payoff is maintainability and solver-control, not guaranteed runtime improvements.

3. Keep solver setup centralized.
- Avoid sprinkling solver-specific details across multiple source files.

## Minimal code sketch for this repository

~~~julia
# src/core/linear_solve_backend.jl
using LinearSolve

function solve_dense_system(A::AbstractMatrix{T}, b::AbstractVector{T};
    alg = LinearSolve.LUFactorization()) where {T<:AbstractFloat}
    prob = LinearSolve.LinearProblem(A, b)
    sol = LinearSolve.solve(prob, alg)
    return sol.u
end
~~~

Then at call sites (currently in `src/exchange/solve_primal.jl`,
`src/exchange/exchange.jl`, and `src/exchange/init_points.jl`):

~~~julia
x = solve_dense_system(A, rhs)
~~~

## Bottom line

LinearSolve.jl is a good fit if your goal is solver flexibility, controlled numerical behavior, and research-friendly experimentation, while preserving the current algorithm structure. For pure simplicity and current small dense problems, Base Julia solves are already a strong baseline.
