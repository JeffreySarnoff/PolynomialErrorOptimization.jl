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

The recommended user-facing entry points are:

* `approxfit`                         — the stable high-level workflow.
* `fit_abs`, `fit_rel`               — mode-specific stable wrappers.
* `plan_fit` and `recommend_parameters` — explicit planning helpers.

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

# High-level convenience interface
export approxfit, fit_abs, fit_rel, recommend_parameters, plan_fit,
    ObjectiveSpec, ComplexitySpec, PrecisionSpec, SearchSpec, FitPlan,
    FitParameters, Approximation,
    error_bound, coeff_count, is_piecewise, pieces

# Standalone evaluator generation
export provide_source, provide, provide_file

# Built-in evaluation schemes
export horner_scheme, fma_horner_scheme, estrin_scheme, fma_estrin_scheme,
    horner_eval, fma_horner_eval, estrin_eval, fma_estrin_eval

end # module PolynomialErrorOptimization
