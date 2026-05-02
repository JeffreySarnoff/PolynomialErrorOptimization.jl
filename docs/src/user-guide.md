# User Guide

This guide is for the expert layer: users who need to choose explicit drivers,
search strategies, or piecewise policies.

## 1. Installation and setup

From Julia:

```julia
using Pkg
Pkg.develop(path = "/path/to/PolynomialErrorOptimization")
Pkg.instantiate()
```

Then load the package:

```julia
using PolynomialErrorOptimization

# Expert APIs are defined on the module but are not exported.
import PolynomialErrorOptimization: eval_approx_optimize,
  eval_approx_optimize_relative,
  eval_approx_optimize_relative_zero,
  approximate,
  approximate_abs,
  approximate_abs_budget,
  GridSearch,
  GridThenLocal,
  GridThenOptim,
  basis_info,
  solution_coefficients
```

## 2. Core concepts

- `EvalScheme`: describes how you evaluate the polynomial (Horner, Estrin, or custom expression tree). It determines `theta(a,t)`.
- `OptimResult`: result from top-level optimization drivers.
- `SearchStrategy`: how the algorithm finds the next worst-case point.
- `PiecewisePolyApprox`: callable piecewise approximation returned by adaptive drivers.
- `Approximation`: high-level wrapper returned by `approxfit`.

For most new workflows, start with the stable `approxfit` interface:

```julia
approx = approxfit(sin, (-3.0, 3.0); target = 1e-8)
@show error_bound(approx)
@show coeff_count(approx)
```

See [High-Level Interface](high-level-interface.md) for the recommended
workflow and [Choosing a Workflow](choosing-a-workflow.md) for the decision
guide.

## 3. Fixed-degree optimization

Use `eval_approx_optimize` when you want one degree-`n` polynomial over the full interval.

```julia
f = exp
n = 5
I = (-1.0, 1.0)
scheme = horner_scheme(n; u = 2.0^-53)

res = eval_approx_optimize(f, n, I, scheme;
    τ = 1e-3,
    max_iter = 100,
    strategy = GridSearch(4096),
    verbose = false)
```

Interpret the key outputs:

- `res.poly`: optimized polynomial.
- `res.total_error`: verified upper bound on the objective.
- `res.discrete_error`: final discrete lower-level quantity.
- `res.iterations`: exchange iterations used.

### Target and computation types

The coefficient type and the internal optimization type can be chosen
separately. At the expert-driver level, the computation type comes from the
scheme, while `target_type` controls returned coefficients:

```julia
scheme32_big = horner_scheme(5, BigFloat; u = eps(Float32) / 2)
res32 = eval_approx_optimize(sin, 5, (-1, 1), scheme32_big;
    target_type = Float32)
```

In the high-level interface, pass both explicitly:

```julia
approx32 = approxfit(sin, (-1.0, 1.0);
    target = 1e-6,
    degree = 5,
    piecewise = false,
    target_type = Float32,
    compute_type = BigFloat)
```

### Relative mode

Use `eval_approx_optimize_relative` if `f` does not vanish on the interval.

```julia
res_rel = eval_approx_optimize_relative(f, n, I, scheme; τ = 1e-3)
```

Use `eval_approx_optimize_relative_zero` when a known finite-order zero is part of the model assumptions.

For `eval_approx_optimize_relative_zero`, `res.poly` is stored in the
monomial basis even when the optimization solve itself used a shifted basis.
Use `basis_info(res)` to inspect that basis metadata and
`solution_coefficients(res)` to recover the coefficient vector in the original
solve basis.

This end-to-end expert example uses a known simple zero at `t_z = 0` and a
deterministic grid search over an interval that does not contain any other
zeros.

```julia
res_rz = eval_approx_optimize_relative_zero(
  t -> t * exp(t),
  4,
  (-0.35, 1.1),
  horner_scheme(4; u = 2.0^-53);
  t_z = 0.0,
  s = 1,
  τ = 1e-2,
  max_iter = 20,
  strategy = GridSearch(4097))

info = basis_info(res_rz)
coeffs = solution_coefficients(res_rz)

@show res_rz.total_error
@show res_rz.iterations
@show info.zero_order
@show info.solution_basis
@show coeffs
```

When `t_z != 0`, `basis_info(res_rz).solution_basis` records whether the
solve basis is shifted while `res_rz.poly` remains stored in the monomial
basis for evaluation and source generation.

## 4. Choosing a search strategy

In each exchange iteration the algorithm must locate the worst-case point
`tstar` — the point in the interval where the error objective is largest.
All three strategies start with the same coarse equispaced grid scan and
differ only in whether and how they refine that result.

### `GridSearch(M)`

Evaluates the error objective at `M` equally spaced points across the
interval and returns the grid point with the highest value.  No subsequent
refinement is performed.  Cost is exactly `M` objective evaluations per
exchange iteration.

- **Accuracy**: limited by grid spacing — error peaks narrower than one grid
  cell may be missed or underestimated.
- **Speed**: fastest; a single pass, no branches, no external optimizer.
- **When to use**: default starting point, reproducibility required, or when
  the error objective is smooth enough that grid density suffices.

### `GridThenLocal(M; bracket=3)`

Runs the same `M`-point grid scan, then performs a golden-section
maximization within a window of `±bracket` grid cells centred on the best
grid point.  The golden-section pass refines the location of the local peak
without extra grid evaluations.  Total cost is `M` objective evaluations plus
approximately 200 golden-section iterations within a narrow sub-interval —
roughly a 5–10 % wall-time increase over `GridSearch` at the same `M`.

- **Accuracy**: reliably better than `GridSearch` near smooth, single-peaked
  local maxima; rarely helps when the peak straddles multiple grid cells.
- **Speed**: low overhead; the refinement window is tiny and has no
  external dependencies.
- **When to use**: objective is mostly smooth and you want modest localization
  improvement without depending on `Optim.jl`.

### `GridThenOptim(M; bracket=3)`

Same grid scan followed by bounded Brent's method (`Optim.Brent()`) over the
`±bracket` window.  Brent's method combines parabolic interpolation with
golden-section bracketing and typically reaches tighter precision than
pure golden-section at the same number of function evaluations.

- **Accuracy**: sharpest localization of the three; parabolic interpolation
  accelerates convergence near smooth peaks.
- **Speed**: comparable to `GridThenLocal`; in practice 5–15 % slower than
  `GridSearch` at the same `M`, but often allows a smaller `M`, which more
  than compensates.
- **When to use**: error peaks are sharp, precise `tstar` localization
  noticeably reduces total exchange iteration count, and `Optim.jl` is
  acceptable.

Example:

```julia
strategy = GridThenOptim(5001; bracket = 4)
res = eval_approx_optimize(f, n, I, scheme; strategy = strategy)
```

Guidance:

- Start with `GridSearch` for reproducibility and fast iteration.
- Move to `GridThenLocal` or `GridThenOptim` when convergence stalls or you
  need sharper worst-case localization to reduce total iteration count.
- Increase `M` before increasing `bracket`: a denser grid improves the
  quality of the initial scan that both refinement methods depend on.
- `bracket = 3` (default) is almost always enough; `bracket = 4` or `5`
  helps only when the optimal point falls near the boundary of its grid cell.

## 5. Piecewise adaptive approximation

Use this when one global degree is insufficient or expensive.  The adaptive
driver bisects the interval recursively until each piece meets the per-piece
`target` or a structural limit (`max_depth`, `min_width`, `total_coeffs`) is
reached.

### Key parameters

- **`max_depth`**: maximum bisection recursion depth.  After `max_depth`
  halvings the narrowest possible piece has width `(b - a) / 2^max_depth`.
  Increase when the function has localized difficulty (for example a steep
  gradient or a near-singularity in a subinterval) that requires many
  bisections to resolve.  Typical starting values: `24` to `30`.

- **`min_width`**: minimum permitted piece width.  Bisection stops and the
  piece is rejected when the sub-interval falls below this floor, regardless
  of whether `target` is met.  Use `0.0` (default) to disable the floor;
  add a floor such as `1e-5` to `1e-3` to prevent over-fragmentation in
  near-singular or steeply varying regions.

- **`total_coeffs`**: global coefficient count cap summed over all accepted
  pieces.  `0` disables the cap.  A positive value enforces a hard model-size
  budget: the driver returns a partial result once the cumulative count would
  exceed this limit.  Fixed-degree pieces each cost exactly `n + 1`
  coefficients; budget-mode pieces cost between `1` and `max_coeffs`.

- **`degree_policy`** *(budget mode only)*: selects how the per-piece
  polynomial degree is chosen when a range of degrees is permitted.
  - `:max` — always attempt the highest permitted degree.  Fewest pieces,
    highest per-piece cost; simplest behaviour.
  - `:min` — use the smallest degree that meets `target` locally.  Locally
    sparse pieces; may use more pieces overall.
  - `:min_cost` — recursively minimizes total coefficient count using a
    branch-and-bound strategy.  Best global coefficient efficiency; highest
    per-piece computation cost.  Prefer when coefficient budget is the
    primary constraint.

- **`driver_max_iter`**: maximum number of inner exchange iterations allowed
  per piece.  If a single piece cannot converge within this budget it is
  bisected instead.  Too small forces unnecessary bisections; too large wastes
  time on genuinely hard pieces.  Start at `100`; increase to `150`–`300`
  for tighter targets or difficult functions.

- **`scheme`**: the `EvalScheme` that defines the polynomial evaluation model
  and the evaluation-error rows used during optimization.  The scheme must
  match the polynomial degree `n` (or `max_coeffs - 1` in budget mode).
  - `horner_scheme(n; u)` — standard Horner evaluation; best general default.
  - `fma_horner_scheme(n; u)` — Horner with fused multiply-add; use when
    the deployed evaluator will use FMA instructions.
  - `estrin_scheme(n; u)` — Estrin (pairwise) evaluation; for parallel or
    SIMD targets.
  The `u` parameter is the unit roundoff of the target floating-point type
  (typically `2.0^-53` for `Float64`, `2.0^-24` for `Float32`).

### Parameter interactions

- `max_depth` and `min_width` are complementary hard stops: a piece is
  rejected when either limit is hit before `target` is met.  Keep them
  consistent: `min_width` should be larger than `(b - a) / 2^max_depth`;
  otherwise `max_depth` is never the binding constraint.
- Tightening `target` while holding `driver_max_iter` fixed can cause
  increased bisection; raise `driver_max_iter` together with `target`.
- `total_coeffs` interacts with `degree_policy`: `:min_cost` is best at
  staying under a coefficient budget; `:max` is most likely to exhaust it
  quickly.

### Fixed degree per piece

```julia
pa = approximate_abs(sin, 4, (-3.0, 3.0), horner_scheme(4; u = 2.0^-53);
    target = 1e-8,
    max_depth = 30,
    min_width = 0.0,
    total_coeffs = 0,
    driver_max_iter = 100)

y = pa(0.7)
```

### Coefficient budget per piece

```julia
pa_budget = approximate_abs_budget(sin, 6, (-3.0, 3.0);
    target = 1e-8,
    degree_policy = :min_cost,
    total_coeffs = 80)
```

`degree_policy` options:

- `:max`: always use the largest allowed degree.
- `:min`: smallest local degree that meets target.
- `:min_cost`: globally cost-aware recursive choice.

### Structured rejection diagnostics

Piecewise fit attempts now carry a structured `FitAttemptReport` internally.
Each report records whether the attempt was accepted, the rejection kind,
interval, degree, mode, target, achieved error, and any caught exception.

The adaptive drivers still surface concise refusal messages such as
`reason: err > target`, but contributors and expert users can inspect the
structured report directly through the internal `_try_fit` path when debugging
piecewise behaviour or adding regressions.

### Unified API

Use `approximate(...)` when you want a single entry point.

```julia
scheme = horner_scheme(5; u = 2.0^-53)

pa1 = approximate(cos, (-2.0, 2.0);
    target = 1e-9,
    mode = :abs,
    n = 5,
    scheme = scheme)

pa2 = approximate(cos, (-2.0, 2.0);
    target = 1e-9,
    mode = :abs,
    max_coeffs = 7,
    degree_policy = :min)
```

## 6. Building custom evaluation schemes

You can define mixed-precision or custom evaluation DAGs via symbolic nodes.

```julia
e = Round(Add(VarA(0), Round(Mul(VarA(1), VarT()), 2.0^-24, 1)), 2.0^-53, 2)
θ = lin_eval_error(e)
ids_to_u = collect_rounding_us(e)
scheme = build_eval_scheme(θ, ids_to_u, 1, "custom-degree-1")
```

For standard workflows, prefer:

- `horner_scheme(n; u=...)`
- `fma_horner_scheme(n; u=...)`
- `estrin_scheme(n; u=...)`
- `fma_estrin_scheme(n; u=...)`

## 7. Practical tuning checklist

1. Start with a moderate degree and `GridSearch`.
2. Tighten `τ` only after strategy and degree are reasonable.
3. If many intervals are created, try budget mode with `:min` or `:min_cost`.
4. If exchange stalls, increase sampling resolution (`M`) and inspect `verbose=true` output.
5. Use `total_coeffs` to enforce hard complexity ceilings.

## 8. Parameter selection

The full parameter-selection playbook is now documented in its own part:
[Parameter Selection](parameter-selection.md).

## 9. Common errors and how to fix them

- `ArgumentError` about interval order:
  ensure `I = (left, right)` with `left < right`.
- `ArgumentError` about degree/scheme mismatch:
  ensure `scheme.n == n` in fixed-degree APIs.
- Relative mode domain errors:
  ensure `f` has no interior zero for `eval_approx_optimize_relative`.
- Budget/depth failure errors in piecewise mode:
  increase `max_depth`, relax `target`, or allow larger `max_coeffs`.
  The refusal message now preserves the immediate rejection reason from the
  structured attempt report.

## 10. Reproducibility tips

- Use fixed `GridSearch(M)` for deterministic baseline behavior.
- Keep package versions pinned when benchmarking.
- Record `strategy`, `τ`, `target`, and budget parameters with results.

## 11. Worked examples

For complete end-to-end recipes (including constrained setups for `cos` and
`acos`), see the dedicated [Recipes](examples.md) section.
