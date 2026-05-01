# PolynomialErrorOptimization.jl

`PolynomialErrorOptimization.jl` is for fitting polynomials when plain approximation error is not the whole story. It optimizes against

- function-approximation error,
- modeled finite-precision evaluation error,

so the resulting polynomial and the intended evaluation scheme are designed together.

## Start here

For most users, the package should feel like one workflow:

```julia
using PolynomialErrorOptimization

approx = approxfit(sin, (-2.0, 2.0); target = 1e-4, effort = :fast, max_depth = 6)

@show error_bound(approx)
@show coeff_count(approx)
@show approx(0.25)
```

If that covers your use case, stay in the high-level interface. Move to the expert APIs only when you need fixed-degree control, piecewise partitioning policy, or explicit evaluation-scheme modeling.

## Audience layers

### Stable workflow layer

- `approxfit`
- `fit_abs`, `fit_rel`
- `recommend_parameters`
- `Approximation`
- `error_bound`, `coeff_count`, `is_piecewise`, `pieces`
- built-in scheme builders such as `horner_scheme` and `estrin_scheme`

### Expert layer

- fixed-degree drivers such as `eval_approx_optimize`
- piecewise drivers such as `approximate_abs_budget`
- search strategies such as `GridSearch` and `GridThenOptim`
- standalone evaluator generation via `provide_source` and `provide`

### Internal and research layer

- exchange substeps such as `init_points`, `solve_primal`, `find_new_index`, `exchange`
- symbolic error-expression nodes such as `Round`, `FMA`, and `VarA`
- low-level row construction and formulation helpers

## Choose the right page

- [High-Level Interface](high-level-interface.md): the recommended path.
- [Choosing a Workflow](choosing-a-workflow.md): pick single vs piecewise, absolute vs relative, and degree vs budget.
- [User Guide](user-guide.md): expert usage and parameter tuning.
- [Parameter Selection](parameter-selection.md): tuning guidance.
- [Recipes](examples.md): worked end-to-end examples.
- [Technical Guide](technical-guide.md): implementation overview.
- [Contributor Guide](contributor-guide.md): contributor workflow and redesign roadmap.
- [API Reference](api.md): API split by stability layer.

## Objective

The fixed-degree core solves

```math
\min_{a \in \mathbb{R}^{n+1}}\;\max_{t \in I}\;\left(|f(t)-p(t)| + \theta(a,t)\right),
```

where `p(t)` is a degree-`n` polynomial and `theta(a,t)` is the linearized rounding-error model induced by the chosen evaluation scheme.

## Local docs build

From the package root:

```julia
julia --project=docs docs/make.jl
```

Generated output is written to `docs/build`.
