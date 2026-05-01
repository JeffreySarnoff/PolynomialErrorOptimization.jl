"""
    PolynomialErrorOptimization

A Julia implementation of the exchange algorithm of

    Arzelier, Bréhard, Hubrecht, Joldeș (2024/2025),
    "An Exchange Algorithm for Optimizing both Approximation and
    Finite-Precision Evaluation Errors in Polynomial Approximations",
    HAL-04709615, ACM Transactions on Mathematical Software.

The package computes a degree-`n` polynomial that minimises the supremum
(over a real interval `I`) of the **sum** of approximation error and a
linearised model of finite-precision evaluation error:

    min_{a ∈ ℝⁿ⁺¹}   max_{t ∈ I}  ( |f(t) − p(t)|  +  θ(a, t) ),
    where p(t) = Σⱼ aⱼ tʲ.

The user-facing entry points are:

* `eval_approx_optimize`              — the absolute-error driver (Algorithm 3).
* `eval_approx_optimize_relative`     — the (P^rel) variant when `f` does
                                         not vanish on `I` (Section 5).
* `eval_approx_optimize_relative_zero` — the (P^rel2) variant when `f` has
                                         a zero of finite order in `I`.

Convenience evaluation-scheme builders:

* `horner_scheme(n; u)` — closed-form linearised error of paper Algorithm 1.
* `fma_horner_scheme(n; u)` — Horner model with one rounded FMA per step.
* `estrin_scheme(n; u)` — Estrin's parallel scheme, built via the symbolic
                          expression-tree pipeline.
* `fma_estrin_scheme(n; u)` — Estrin model with one rounded FMA per affine
                              combine.

For custom mixed-precision schemes, use the symbolic AST in
`eval_error.jl`:  build a tree of `Round`/`Add`/`Mul`/`FMA`/`Var…` nodes, run
`lin_eval_error` on it, then `build_eval_scheme` to obtain the runtime
`EvalScheme`.

# Quick example

```julia
using PolynomialErrorOptimization

f      = sin
n      = 6
I      = (-2.0, 2.0)
scheme = horner_scheme(n; u = 2.0^-12)        # toy precision (paper §6.1)
res    = eval_approx_optimize(f, n, I, scheme; τ = 0.01, verbose = true)

@show res.poly
@show res.total_error
@show res.iterations
```
"""
module PolynomialErrorOptimization

using ArgCheck: @argcheck
using LinearAlgebra
import LinearSolve
import Optim
import Polynomials
import Printf

# Internal source files (order matters: core types/helpers → formulation and
# search support → symbolic schemes → algorithm components → drivers).
include("core/types.jl")
include("core/numerics.jl")
include("core/constraints.jl")
include("core/linear_solve_backend.jl")
include("schemes/eval_error.jl")
include("schemes/horner.jl")
include("schemes/estrin.jl")
include("exchange/search.jl")
include("exchange/init_points.jl")
include("exchange/solve_primal.jl")
include("exchange/find_new_index.jl")
include("exchange/exchange.jl")
include("exchange/driver.jl")
include("piecewise/approximate.jl")
include("provide.jl")
include("interface.jl")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Core types
export Index, Signature, EvalScheme, OptimResult,
    AbstractMode, AbsoluteMode, RelativeMode, RelativeZeroMode,
    SearchStrategy, GridSearch, GridThenLocal, GridThenOptim

# Top-level drivers (paper Algorithm 3 + Section 5)
export eval_approx_optimize,
    eval_approx_optimize_relative,
    eval_approx_optimize_relative_zero

# Adaptive piecewise approximation (built on the drivers above)
export approximate,
    approximate_abs, approximate_rel,
    approximate_abs_budget, approximate_rel_budget,
    default_scheme_builder,
    PiecewisePolyApprox, ApproxPiece

# High-level convenience interface
export approxfit, fit_abs, fit_rel, recommend_parameters,
    FitParameters, Approximation,
    error_bound, coeff_count, is_piecewise, pieces

# Standalone evaluator generation
export provide_source, provide, provide_file

# Built-in evaluation schemes
export horner_scheme, fma_horner_scheme, estrin_scheme, fma_estrin_scheme,
    horner_eval, fma_horner_eval, estrin_eval, fma_estrin_eval

# Lower-level building blocks (paper Algorithms 4–7)
export init_points, solve_primal, find_new_index, exchange

# Symbolic evaluation-error machinery (paper Algorithm 2)
export ErrExpr, VarT, VarA, Const, Neg, Add, Mul, FMA, Round,
    lin_eval_error, build_eval_scheme,
    collect_rounding_us, horner_expr, fma_horner_expr, estrin_expr,
    fma_estrin_expr

# Exceptions
export ExchangeFailure, ConvergenceFailure

end # module PolynomialErrorOptimization
