# API Reference

This page lists the API by stability layer rather than by implementation file.

## Stable workflow layer

Use these first.

- `approxfit`
- `fit_abs`
- `fit_rel`
- `plan_fit`
- `recommend_parameters`
- `ObjectiveSpec`
- `ComplexitySpec`
- `PrecisionSpec`
- `SearchSpec`
- `FitPlan`
- `FitParameters`
- `Approximation`
- `error_bound`
- `coeff_count`
- `is_piecewise`
- `pieces`
- `horner_scheme`
- `fma_horner_scheme`
- `estrin_scheme`
- `fma_estrin_scheme`

`plan_fit` and the typed spec objects provide the structured planning path.
`FitParameters` remains the flattened executable bundle used by the current
keyword API. The high-level keywords separate `target_type` for returned
coefficients and the modeled default floating-point format from
`compute_type` for internal optimization arithmetic.

Everything in the stable workflow layer is exported from
`PolynomialErrorOptimization`.

## Expert layer

Use these when the stable workflow layer is too coarse.

Expert names remain defined on `PolynomialErrorOptimization` but are not
exported; access them via explicit imports or module-qualified calls.

### Fixed-degree drivers

- `eval_approx_optimize`
- `eval_approx_optimize_relative`
- `eval_approx_optimize_relative_zero`
- `OptimResult`
- `ResultBasis`
- `basis_info`
- `solution_coefficients`

`OptimResult.poly` is always a monomial-basis polynomial. For
`eval_approx_optimize_relative_zero`, `basis_info(result)` records whether the
optimization solve used a shifted basis and `solution_coefficients(result)`
returns that original coefficient vector explicitly.

### Piecewise drivers and results

- `approximate`
- `approximate_abs`
- `approximate_rel`
- `approximate_abs_budget`
- `approximate_rel_budget`
- `default_scheme_builder`
- `ApproxPiece`
- `PiecewisePolyApprox`
- `FitAttemptReport`

### Search and export tools

- `SearchStrategy`
- `GridSearch`
- `GridThenLocal`
- `GridThenOptim`
- `provide_source`
- `provide`
- `provide_file`
- `horner_eval`
- `fma_horner_eval`
- `estrin_eval`
- `fma_estrin_eval`

## Advanced and research layer

These APIs are useful for contributors and research workflows, but they are
not the recommended starting point for package users.

### Exchange building blocks

- `Index`
- `Signature`
- `EvalScheme`
- `AbstractMode`
- `AbsoluteMode`
- `RelativeMode`
- `RelativeZeroMode`
- `init_points`
- `solve_primal`
- `find_new_index`
- `exchange`

### Symbolic error-model API

- `ErrExpr`
- `VarT`
- `VarA`
- `Const`
- `Neg`
- `Add`
- `Mul`
- `FMA`
- `Round`
- `lin_eval_error`
- `build_eval_scheme`
- `collect_rounding_us`
- `horner_expr`
- `fma_horner_expr`
- `estrin_expr`
- `fma_estrin_expr`

### Exceptions

- `ExchangeFailure`
- `ConvergenceFailure`

## Canonical signatures

The canonical source of signatures and docstrings is the code in `src/`. In
particular, `src/interface.jl`, `src/exchange/driver.jl`, and `src/piecewise/public.jl`
define the user-facing contracts.
