# API Reference

This page lists the public API by category.

## Core domain types

- `Index`
- `Signature`
- `EvalScheme`
- `AbstractMode`
- `AbsoluteMode`
- `RelativeMode`
- `RelativeZeroMode`

## Driver result type

- `OptimResult`

`OptimResult{TargetT,ComputeT}` stores polynomial coefficients in `TargetT`
and verified error/search state in `ComputeT`.

## Search strategies

- `SearchStrategy`
- `GridSearch`
- `GridThenLocal`
- `GridThenOptim`

## Piecewise result types

- `ApproxPiece`
- `PiecewisePolyApprox`

## High-level interface

- `approxfit`
- `PolynomialErrorOptimization.fit` (unexported; prefer `approxfit`)
- `fit_abs`
- `fit_rel`
- `recommend_parameters`
- `FitParameters`
- `Approximation`
- `error_bound`
- `coeff_count`
- `is_piecewise`
- `pieces`

The high-level `approxfit`/`recommend_parameters` keywords include
`target_type` for returned coefficients and the default modeled
floating-point format, and `compute_type` for internal optimization
arithmetic.

## Top-level optimization drivers

- `eval_approx_optimize`
- `eval_approx_optimize_relative`
- `eval_approx_optimize_relative_zero`

## Piecewise approximation drivers

- `approximate`
- `approximate_abs`
- `approximate_rel`
- `approximate_abs_budget`
- `approximate_rel_budget`
- `default_scheme_builder`

## Standalone evaluator generation

- `provide_source`
- `provide`
- `provide_file`

## Evaluation scheme builders and evaluators

- `horner_scheme`
- `fma_horner_scheme`
- `estrin_scheme`
- `fma_estrin_scheme`
- `horner_eval`
- `fma_horner_eval`
- `estrin_eval`
- `fma_estrin_eval`

## Exchange algorithm building blocks

- `init_points`
- `solve_primal`
- `find_new_index`
- `exchange`

## Symbolic error-model API

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

## Exceptions

- `ExchangeFailure`
- `ConvergenceFailure`

## Where to find remaining signatures

The canonical source of signatures and docstrings is the source code in
`src/`. In particular, the high-level interface docstrings in
`src/interface.jl` are the source of truth for argument meanings and examples.

