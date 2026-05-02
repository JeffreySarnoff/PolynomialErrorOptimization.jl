# Technical Guide

This guide is for contributors and advanced users who need implementation-level detail.

For contributor workflow and the staged redesign plan, see
[Contributor Guide](contributor-guide.md).

## 1. Architecture overview

The package is organized around a small set of core abstractions:

- **`EvalScheme`**: the evaluation-error model for a specific polynomial
  evaluation scheme (Horner, Estrin, FMA variants).  Encapsulates degree `n`,
  error row count `k`, and functions `pi_i(t)` that produce the constraint row
  vectors fed to the linear solver in each exchange iteration.  Changing the
  scheme changes what the optimizer certifies: errors in Horner evaluation
  differ from errors in Estrin evaluation at the same floating-point precision.

- **exchange-loop state**: the active working set of `n + 2` discretization
  points (called "indices"), together with their alternation signatures (sign
  pattern used by the dual equioscillation certificate), plus the current
  primal coefficient vector and the dual equioscillation level.  The exchange
  algorithm maintains and updates this state on every iteration: one old index
  leaves the set and the new worst-case point `tstar` enters.

- **mode tags** (`AbsoluteMode`, `RelativeMode`, `RelativeZeroMode`): type
  tags that dispatch constraint-row construction and worst-case search to the
  appropriate error formulation.
  - `AbsoluteMode`: minimizes `max |f(t) - p(t)|` directly.
  - `RelativeMode`: minimizes `max |f(t) - p(t)| / |f(t)|`; requires `f`
    strictly non-vanishing on the interval.
  - `RelativeZeroMode`: minimizes `max |f(t) - p(t)| / |t - t_z|^s`; handles
    functions with a known finite-order zero, shifting the initial points away
    from `t_z` and skipping it during worst-case search.

- **strategy layer**: the `SearchStrategy` subtypes (`GridSearch`,
  `GridThenLocal`, `GridThenOptim`) dispatched by `find_new_index` to locate
  the worst-case point `tstar` in each iteration.  This is the only component
  of the algorithm that varies between strategies; primal solve, exchange
  pivot, and convergence check are all strategy-independent.

High-level flow:

1. **Build or select an `EvalScheme`**: choose the evaluation model (Horner,
   Estrin, FMA) and supply the unit roundoff `u` for the target platform.
   The scheme fixes the error-model matrix that the linear solver uses.
2. **Initialize discretization** (`init_points`): seed the active working set
   with `n + 2` Chebyshev-distributed points on the interval.  For
   `RelativeZeroMode` the nodes are shifted to avoid landing on the known zero
   `t_z`.
3. **Solve primal linear system** (`solve_primal`): given the current working
   set, compute the polynomial coefficients and the equioscillation level that
   the dual certificate requires.  This is a dense `(n+2) x (n+2)` system
   solved via `LUFactorization`.
4. **Locate worst violation** (`find_new_index`): evaluate the error objective
   at many candidate points to find `tstar` — the point where the error most
   exceeds the equioscillation level.  Delegates to the chosen `SearchStrategy`
   for efficient localization.
5. **Exchange one index** (`exchange`): remove the least-critical point from
   the active set, insert `tstar`, and update alternation signatures.  If the
   exchange step cannot maintain the alternation structure an `ExchangeFailure`
   is raised.
6. **Iterate in the driver**: repeat steps 3–5 until the relative improvement
   in the maximum error falls below `τ` (convergence declared) or `max_iter`
   is exhausted (a `ConvergenceFailure` is raised).

## 2. Source map

- `src/core/types.jl`: foundational domain types (`Index`, `Signature`, `EvalScheme`, mode tags).
- `src/core/numerics.jl`: low-level numeric helpers (`monomial_dot`, `dot_view`, sign handling).
- `src/core/constraints.jl`: absolute and relative constraint rows/RHS builders (`α`, `c`, and mode dispatch helpers).
- `src/core/linear_solve_backend.jl`: centralized dense solver backend.
- `src/schemes/eval_error.jl`: symbolic error-expression machinery.
- `src/schemes/horner.jl`: Horner evaluator, closed-form scheme, and symbolic Horner tree.
- `src/schemes/estrin.jl`: Estrin scheme and symbolic Estrin tree.
- `src/exchange/search.jl`: search strategy types and max-location implementations.
- `src/exchange/init_points.jl`: initialization algorithm.
- `src/exchange/solve_primal.jl`: primal solve in each mode.
- `src/exchange/find_new_index.jl`: worst-case point search and signatures.
- `src/exchange/exchange.jl`: exchange pivot logic, update step, and `ExchangeFailure`.
- `src/exchange/driver.jl`: top-level fixed-degree optimization APIs,
  `OptimResult`, basis metadata helpers, and `ConvergenceFailure`.
- `src/piecewise/`: adaptive piecewise and budget-aware drivers split by type/config/fitting/bisection policy.
- `src/provide.jl`: standalone evaluator source generation for exporting approximations.
- `src/interface.jl`: high-level `approxfit`/recommendation wrapper over the expert APIs.

## 3. Core contracts and invariants

### Eval scheme invariants

- `scheme.n >= 0`
- `scheme.k >= 0`
- `length(scheme.pi) == scheme.k`
- each `pi_i(t)` returns a vector of length `n + 1`

### Driver invariants

- fixed-degree APIs require `scheme.n == n`
- fixed-degree internal arithmetic uses `fptype(scheme)`
- returned polynomial coefficients use the requested `target_type`
- `OptimResult.poly` is always monomial-basis even when `RelativeZeroMode`
  solves in a shifted basis
- interval must satisfy `I[1] < I[2]`
- `τ > 0`
- iteration bounds and depth/budget parameters are nonnegative

### Piecewise invariants

- pieces remain left-to-right and non-overlapping
- accepted piece satisfies per-piece target by verified bound
- optional `total_coeffs` cap is enforced eagerly

## 4. Internal configuration structs

Recent refactors consolidated repeated high-arity argument bundles into explicit config types.
The public piecewise API still accepts `mode = :abs`/`:rel`; internally the
piecewise configs use `AbsoluteMode()` and `RelativeMode()` so mode handling
matches the fixed-degree driver layer.

### Fixed-degree path

- `DriveConfig` (`src/exchange/driver.jl`): loop tolerance/iteration/search/logging controls.
- `FixedApproxConfig` (`src/piecewise/config.jl`): fixed-degree piecewise controls.
- `FitConfig` and `BisectConfig` (`src/piecewise/config.jl`): shared internals for per-piece fitting and bisection.

### Budget path

- `BudgetApproxConfig` (`src/piecewise/config.jl`): degree policy, builder, and shared budget controls.

These structs make call chains easier to audit and reduce signature drift during refactors.

## 5. Numerical backends

### Linear system backend

All dense linear solves are routed through `solve_dense_system` in `src/core/linear_solve_backend.jl`, currently using LinearSolve defaults (`LUFactorization()` by default).

Why this matters:

- one place to tune direct-solver policy,
- easier fallback strategies (for example LU -> QR),
- avoids backend details in algorithm files.

### Maximum-search backend

`find_new_index` delegates localization through strategy dispatch:

- `GridSearch`
- `GridThenLocal`
- `GridThenOptim` (uses `Optim.Brent()` bounded refinement)

This keeps global algorithm logic independent from local search details.

## 6. Error handling policy

The code distinguishes argument/precondition failures from algorithmic failures.

- `@argcheck` (ArgCheck.jl): argument validation and structural invariants.
- custom exceptions:
  - `ExchangeFailure`: exchange step cannot proceed under assumptions.
  - `ConvergenceFailure`: max iterations reached before tolerance criterion.
- domain-specific errors (for example relative mode non-vanishing assumptions) remain explicit where value context is important.

## 7. Piecewise and budget internals

The `src/piecewise/` subsystem has two major pathways.

### Fixed-degree piecewise

- adaptive bisection accepts/rejects by verified per-piece error.
- rejection reasons include non-finite objective, target miss, or inner-driver exceptions.

### Coefficient-budget piecewise

- per-piece degree cap via `max_coeffs`.
- policies:
  - `:max` (fixed highest degree),
  - `:min` (smallest local successful degree),
  - `:min_cost` (recursive global cost minimization with pruning).
- optional global coefficient cap via `total_coeffs`.

## 8. Performance notes

- Most runtime is in repeated driver calls and max-location objective evaluations.
- `GridThenOptim` can reduce required grid density for tighter local maxima.
- Scheme caching in budget mode avoids repeated `scheme_builder(d)` construction.
- Keep checks on API boundaries and avoid adding validation in inner numeric loops unless required.

## 9. Testing and validation

Current workflow:

```julia
using Pkg
Pkg.test()
```

The test suite is split by subsystem under `test/core/`, `test/schemes/`,
`test/exchange/`, `test/piecewise/`, and `test/interface/`.

Suggested contributor checks:

1. run full tests after algorithm or config-struct changes,
2. include at least one strategy-sensitive test when changing maximum search,
3. include budget-path tests when touching piecewise internals,
4. verify no method ambiguities are introduced by dispatch changes.

## 10. Extending the package safely

### Add a new search strategy

1. add a subtype of `SearchStrategy` in `src/exchange/search.jl`,
2. implement `locate_maximum` dispatch in `src/exchange/search.jl`,
3. add tests in `test/exchange/`,
4. update user docs with when to use it.

### Add a new solve policy

1. keep algorithm files backend-agnostic,
2. update `solve_dense_system` to expose optional algorithm selection,
3. add regression tests in modes that are numerically sensitive.

### Add a new evaluation scheme builder

1. ensure `EvalScheme` invariants are met,
2. validate `pi_i` output dimensions,
3. add end-to-end driver tests in `test/schemes/` and `test/exchange/` at multiple degrees.

## 11. Related implementation notes

The repository also contains integration-oriented notes:

- `docs/argcheck.md`
- `docs/optim.md`
- `docs/linearsolve.md`

Those files explain why each dependency was introduced and summarize adoption choices.

## 12. Generated documentation policy

`docs/build/` is generated output from Documenter.jl. Do not edit files in
that directory by hand; regenerate them from the package root with:

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

Source documentation lives in `docs/src/` and package docstrings. In a normal
git workflow, keep `docs/build/` ignored unless you intentionally deploy static
documentation from that directory.
