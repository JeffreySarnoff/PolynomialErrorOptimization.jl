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

Available strategies:

- `GridSearch(M)`: robust, deterministic baseline.
- `GridThenLocal(M; bracket=3)`: grid plus local bounded refinement.
- `GridThenOptim(M; bracket=3)`: grid plus `Optim.Brent()` refinement.

Example:

```julia
strategy = GridThenOptim(5001; bracket = 4)
res = eval_approx_optimize(f, n, I, scheme; strategy = strategy)
```

Guidance:

- Start with `GridSearch` for reproducibility.
- Move to `GridThenOptim` when tighter max-point localization helps convergence.
- Increase `M` before increasing `bracket`.

## 5. Piecewise adaptive approximation

Use this when one global degree is insufficient or expensive.

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
