# PolynomialErrorOptimization.jl

`PolynomialErrorOptimization.jl` computes polynomials that balance two sources of error at once:

- function-approximation error,
- finite-precision polynomial evaluation error.

It implements the exchange algorithm from Arzelier, Brehard, Hubrecht, and Joldes for the objective

```math
\min_{a \in \mathbb{R}^{n+1}}\;\max_{t \in I}\;\left(|f(t)-p(t)| + \theta(a,t)\right),
```

where `p(t)` is a degree-`n` polynomial and `theta(a,t)` is the linearized rounding-error model induced by an evaluation scheme.

## What this documentation contains

- [High-Level Interface](high-level-interface.md): the recommended `fit` workflow, parameter recommendations, and result inspection helpers.
- [User Guide](user-guide.md): how to install, choose modes/strategies, and run fixed-degree and budgeted piecewise workflows.
- [Parameter Selection](parameter-selection.md): practical guidance for choosing `target`, `max_depth`, `min_width`, `total_coeffs`, `driver_max_iter`, and strategy settings.
- [Examples](examples.md): worked end-to-end approximation setups, including constrained `cos` and `acos` recipes.
- [Technical Guide](technical-guide.md): architecture, internals, numerical choices, and extension points.
- [API Reference](api.md): public API organized by category.

## Quick start

```julia
using PolynomialErrorOptimization

f = sin
n = 6
I = (-2.0, 2.0)
scheme = horner_scheme(n; u = 2.0^-53)

res = eval_approx_optimize(f, n, I, scheme; τ = 1e-3)

@show res.poly
@show res.total_error
@show res.iterations
```

## Build the docs locally

From the package root:

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

The generated site will be written to `docs/build`.
