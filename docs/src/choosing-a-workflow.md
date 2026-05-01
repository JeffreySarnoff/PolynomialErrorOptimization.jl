# Choosing a Workflow

This page is the decision guide for the package.

## Start with one question

Do you want the package to choose a reasonable workflow for you, or do you
need to control the driver directly?

- If you want the recommended path, use `approxfit`.
- If you already know the polynomial degree, evaluation scheme, and search policy, use an expert driver.

## Capability matrix

| Need | Recommended API |
| --- | --- |
| one obvious default workflow | `approxfit` |
| fixed global polynomial | `approxfit(...; degree=..., piecewise=false)` or `eval_approx_optimize` |
| piecewise approximation with simple defaults | `approxfit(...; max_coeffs=...)` |
| piecewise approximation with explicit degree policy | `approximate_abs_budget` or `approximate_rel_budget` |
| relative objective away from zeros | `fit_rel` or `eval_approx_optimize_relative` |
| known zero of finite order | `eval_approx_optimize_relative_zero` |
| explicit control of search strategy | expert drivers plus `GridSearch`, `GridThenLocal`, or `GridThenOptim` |
| custom mixed-precision evaluation model | symbolic AST plus `lin_eval_error` and `build_eval_scheme` |

## Single polynomial vs piecewise

Choose a single polynomial when:

- the interval is modest,
- one degree is acceptable across the whole interval,
- you want the simplest deployment shape.

Choose piecewise when:

- one global polynomial is too expensive,
- the function has varying difficulty across the interval,
- you need a hard coefficient budget.

## Absolute vs relative

Choose absolute mode when:

- the function crosses zero,
- the relevant engineering tolerance is absolute,
- you want the least restrictive problem formulation.

Choose relative mode when:

- the function stays away from zero on the interval,
- the application is scale-sensitive,
- you care about fractional rather than absolute error.

For known finite-order zeros, use the relative-zero expert driver instead of plain relative mode.

## Fixed degree vs coefficient budget

Choose fixed degree when:

- deployment already dictates a polynomial size,
- every piece must use the same degree,
- you want direct comparison against classical fixed-degree approximations.

Choose a coefficient budget when:

- you care about total storage or instruction budget,
- different intervals can justify different local degrees,
- you want `:min` or `:min_cost` piecewise policies.

## Recommended progression

1. Start with `approxfit(f, I; target=...)`.
2. Add `degree=...` or `max_coeffs=...` when the workflow needs to be explicit.
3. Drop to expert drivers only when you need search, scheme, or partition-policy control.
4. Use the symbolic layer only for custom evaluation-error models.
