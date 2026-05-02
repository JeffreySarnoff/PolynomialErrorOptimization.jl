# Recipes

This page collects worked recipes for common approximation tasks, with special
focus on constrained workflows for `cos` and `acos`.

## 0. Start with the high-level interface

```julia
using PolynomialErrorOptimization

approx = approxfit(cos, (-3.0, 3.0);
    target = 1e-8,
    effort = :balanced)

@show error_bound(approx)
@show coeff_count(approx)
@show approx(1.2)
```

When you need explicit control, use the lower-level examples below.

## 1. End-to-end examples for cos and acos

This section shows practical API patterns for developing approximations with
different constraints.

The expert APIs used below are available from the module but intentionally not
exported from the root namespace. Start expert workflows with explicit imports:

```julia
using PolynomialErrorOptimization

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

What each import provides:

- `eval_approx_optimize`: fixed-degree absolute-error optimization driver.
  Returns an `OptimResult` with the best polynomial and a verified error bound.
- `eval_approx_optimize_relative`: fixed-degree relative-error driver;
  minimizes `|f(t) - p(t)| / |f(t)|`.  Requires `f` to be strictly
  non-vanishing on the interval.
- `eval_approx_optimize_relative_zero`: relative-error driver for functions
  with a known finite-order zero.  The zero location `t_z` and its order `s`
  must be supplied; the driver avoids evaluating the objective at the zero.
- `approximate`: unified piecewise driver; dispatches to fixed-degree or
  budget-mode path based on keyword arguments (`n` vs `max_coeffs`).
- `approximate_abs`: piecewise fixed-degree driver for absolute-error
  minimization.  Takes an explicit `EvalScheme` as a positional argument.
- `approximate_abs_budget`: piecewise budget driver that optimizes across a
  range of per-piece degrees up to `max_coeffs - 1`, chosen by `degree_policy`.
- `GridSearch`, `GridThenLocal`, `GridThenOptim`: search strategy types.
  Passed as the `strategy` keyword to any fixed-degree or piecewise driver.
  See the User Guide for a detailed comparison of their performance tradeoffs.
- `basis_info`: extracts basis metadata from a `RelativeZeroMode` result.
  Returns a named tuple with `zero_order` (the order `s` of the known zero)
  and `solution_basis` (`:monomial` or `:shifted`, recording the internal
  linear system basis used by the solver).
- `solution_coefficients`: returns the coefficient vector in the basis in
  which the internal linear system was solved, which may differ from the
  monomial-basis polynomial stored in `res.poly`.

### 1.1 cos: fixed degree, absolute objective

```julia
f = cos
n = 6
I = (-2.0, 2.0)
scheme = horner_scheme(n; u = 2.0^-53)

res_abs = eval_approx_optimize(f, n, I, scheme;
    τ = 1e-3,
    max_iter = 120,
    strategy = GridThenOptim(6001; bracket = 4),
    verbose = false)

@show res_abs.total_error
@show res_abs.iterations
@show res_abs.poly
```

Key result fields:

- `res_abs.total_error`: verified upper bound on the maximum absolute error
  `max_{t ∈ I} |f(t) - p(t)|`.
- `res_abs.iterations`: number of exchange iterations the driver used before
  converging (or hitting `max_iter`).
- `res_abs.poly`: the optimized polynomial `p` stored as a `Polynomial`
  object in the monomial basis.

Use this when you want one global polynomial and explicit control over search
quality (`strategy`) and convergence budget (`max_iter`, `τ`).

### 1.2 cos: fixed degree, relative objective on a non-vanishing interval

`eval_approx_optimize_relative` requires that `f` does not vanish on `I`.
For `cos`, pick an interval away from `π/2 + kπ`.

```julia
f = cos
n = 5
I_rel = (0.0, 1.0)  # cos(t) > 0 on this interval
scheme = horner_scheme(n; u = 2.0^-53)

res_rel = eval_approx_optimize_relative(f, n, I_rel, scheme;
    τ = 1e-3,
    strategy = GridSearch(5001))

@show res_rel.total_error
```

### 1.3 cos: piecewise fixed-degree with structural constraints

This pattern is useful when a single global polynomial is too expensive.

```julia
pa_cos = approximate_abs(cos, 4, (-6.0, 6.0), horner_scheme(4; u = 2.0^-53);
    target = 1e-8,
    max_depth = 28,
    min_width = 1e-3,
    total_coeffs = 120,
    driver_max_iter = 100,
    strategy = GridThenLocal(4001; bracket = 3),
    verbose = false)

@show length(pa_cos.pieces)
@show pa_cos.worst_error
@show pa_cos(1.2)
```

Constraints shown here:

- geometric constraints: `max_depth`, `min_width`
- global complexity cap: `total_coeffs`
- per-piece solve controls: `driver_max_iter`, `strategy`

### 1.4 acos: fixed degree, absolute objective on interior domain

`acos` is finite on `[-1, 1]` but has endpoint sensitivity near `±1`, so many
workflows start on an interior interval.

```julia
f = acos
n = 7
I = (-0.95, 0.95)
scheme = horner_scheme(n; u = 2.0^-53)

res_acos_abs = eval_approx_optimize(f, n, I, scheme;
    τ = 5e-4,
    strategy = GridThenOptim(7001; bracket = 4))

@show res_acos_abs.total_error
```

### 1.5 acos: relative objective with non-vanishing constraint

`acos(t)` is strictly positive on `(-1, 1)`, so relative mode is valid on
intervals that exclude `t = 1`.

```julia
f = acos
n = 6
I_rel = (-0.9, 0.9)
scheme = estrin_scheme(n; u = 2.0^-53)

res_acos_rel = eval_approx_optimize_relative(f, n, I_rel, scheme;
    τ = 1e-3,
    strategy = GridSearch(6001))

@show res_acos_rel.total_error
```

### 1.6 relative-zero fixed-degree workflow

Use `eval_approx_optimize_relative_zero` when the zero location and order are
known ahead of time.

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
@show res_rz.poly
```

Key result fields:

- `res_rz.total_error`: verified upper bound on the relative-zero error
  `max_{t ∈ I} |f(t) - p(t)| / |t - t_z|^s`.
- `res_rz.iterations`: exchange iterations used.
- `res_rz.poly`: the optimized polynomial in the monomial basis (always,
  even when the internal solve used a shifted basis).
- `info.zero_order`: the order `s` of the known zero as recorded in the
  basis metadata (here `1` for a simple zero).
- `info.solution_basis`: `:monomial` when the solver used the standard basis,
  `:shifted` when it used a basis centred at `t_z`.
- `coeffs`: coefficient vector in the `solution_basis` basis; use this when
  you need the exact coefficients the solver produced before any basis change.

This recipe uses a deterministic grid and a known simple zero at the origin.
For nonzero `t_z`, inspect `basis_info(res_rz)` and
`solution_coefficients(res_rz)` before exporting coefficients.

### 1.7 acos: budget-constrained piecewise approximation

Use this when you need hard limits on per-piece and total polynomial cost.
`approximate_abs_budget` fits piecewise polynomials where each piece may use
a different degree up to `max_coeffs - 1`, chosen by `degree_policy`.

```julia
pa_acos_budget = approximate_abs_budget(acos, 8, (-0.99, 0.99);
    target = 2e-8,
    degree_policy = :min_cost,
    τ = 1e-3,
    max_depth = 30,
    min_width = 1e-4,
    total_coeffs = 140,
    driver_max_iter = 120,
    strategy = GridThenOptim(5001; bracket = 3),
    verbose = false)

@show length(pa_acos_budget.pieces)
@show pa_acos_budget.worst_error
```

Positional arguments:

- `acos`: the function to approximate.
- `8` (`max_coeffs`): per-piece coefficient cap; each piece uses at most 8
  coefficients, so the maximum per-piece degree is 7.
- `(-0.99, 0.99)`: the approximation interval.  `acos` has infinite
  derivatives at `±1` so this avoids the endpoint singularities.

Keyword arguments and how they interact:

- `target = 2e-8`: per-piece acceptance threshold.  A piece is accepted once
  its verified absolute error bound `≤ 2e-8`.  Tightening this forces more
  bisections and higher total coefficient use.
- `degree_policy = :min_cost`: globally cost-aware degree selection.  The
  driver tries all degrees 1–7 per piece and chooses the combination that
  minimizes total coefficients subject to `target`.  Produces the most compact
  model among the three policies at the cost of higher per-piece computation.
  (`:max` always uses degree 7; `:min` uses the smallest local degree that
  meets `target` without global coordination.)
- `τ = 1e-3`: inner exchange convergence tolerance.  Each per-piece exchange
  loop stops when the relative improvement falls below `τ`.  Larger values
  converge faster but leave the error bound looser relative to the true
  optimum; smaller values improve certification at the cost of more inner
  iterations.
- `max_depth = 30`: maximum bisection depth.  With interval width ≈ 1.98, the
  minimum piece width from depth alone is `1.98 / 2^30 ≈ 1.8e-9`, well below
  `min_width`, so `min_width` is the effective floor here.
- `min_width = 1e-4`: minimum piece width.  Bisection stops and the piece is
  rejected as too narrow if the sub-interval falls below `1e-4`.  This
  prevents over-fragmentation near the endpoint singularities while still
  allowing fine-grained coverage elsewhere.
- `total_coeffs = 140`: global coefficient cap.  The driver stops accepting
  new pieces once the cumulative count would exceed 140.  With `max_coeffs =
  8` this allows at most 17 to 35 pieces depending on per-piece degree, which
  is consistent with the expected piece count for this function and target.
- `driver_max_iter = 120`: per-piece inner exchange budget.  If a single
  piece does not converge within 120 iterations it is bisected.  Setting this
  above the expected convergence count (typically 20–60 for `acos` at this
  target) avoids premature bisection while still bounding per-piece cost.
- `strategy = GridThenOptim(5001; bracket = 3)`: 5001-point grid scan
  followed by bounded Brent refinement in a `±3`-cell window.  `acos` has a
  steep slope near `±0.99`, making sharp localization worthwhile here.
- `verbose = false`: suppresses per-piece diagnostic output.

Result fields:

- `pa_acos_budget.pieces`: vector of accepted pieces, each carrying its
  sub-interval and fitted polynomial.
- `pa_acos_budget.worst_error`: maximum verified error bound across all
  accepted pieces; the overall guaranteed accuracy of the approximation.

### 1.8 Unified API pattern for cos and acos

If you prefer one entry point, use `approximate`:

```julia
# cos with fixed degree
scheme_cos = horner_scheme(5; u = 2.0^-53)
pa_cos_u = approximate(cos, (-3.0, 3.0);
    target = 1e-8,
    mode = :abs,
    n = 5,
    scheme = scheme_cos,
    strategy = GridThenLocal(4001; bracket = 3))

# acos with coefficient budget
pa_acos_u = approximate(acos, (-0.98, 0.98);
    target = 5e-8,
    mode = :abs,
    max_coeffs = 7,
    degree_policy = :min,
    total_coeffs = 120,
    strategy = GridSearch(5001))
```

### 1.9 Separate target and computation types

Use `target_type` to choose the coefficient type returned by the fit and the
floating-point format modeled by the default evaluation-error scheme. Use
`compute_type` to choose the arithmetic used internally by the optimizer.

This example models Float32 polynomial evaluation and returns Float32
coefficients, but runs the exchange algorithm in `BigFloat`:

```julia
approx32_big = approxfit(sin, (-1.0, 1.0);
    target = 1e-6,
    degree = 5,
    piecewise = false,
    target_type = Float32,
    compute_type = BigFloat)

import Polynomials
@show typeof(error_bound(approx32_big))       # BigFloat
@show eltype(Polynomials.coeffs(approx32_big.model.poly))  # Float32
```

This example keeps the internal optimization arithmetic at `Float64` while
targeting Float32 coefficients and Float32 evaluation error:

```julia
approx32_64 = approxfit(cos, (-1.0, 1.0);
    target = 1e-6,
    degree = 5,
    piecewise = false,
    target_type = Float32,
    compute_type = Float64)
```

Julia's built-in 32-bit floating type is spelled `Float32`.

### 1.10 FMA Horner fitting and evaluation

Use `scheme = :horner_fma` when your deployed evaluator will use explicit
fused multiply-add Horner steps. This changes the optimization error model; it
is not just a different code-generation spelling.

```julia
approx_fma = approxfit(sin, (-1.0, 1.0);
    target = 1e-8,
    degree = 5,
    piecewise = false,
    scheme = :horner_fma)

@show approx_fma.parameters.eval_op  # :fma

src_fma = provide_source(approx_fma;
    name = :sin_fma_eval,
    check_domain = true)
```

At the expert-driver level, choose the FMA scheme explicitly:

```julia
scheme_fma = fma_horner_scheme(5; u = 2.0^-53)
res_fma = eval_approx_optimize(sin, 5, (-1.0, 1.0), scheme_fma;
    τ = 1e-3)

src_res_fma = provide_source(res_fma;
    name = :sin_fma_eval2,
    interval = (-1.0, 1.0),
    eval_op = :fma)
```

### 1.11 FMA Estrin fitting and evaluation

Use `scheme = :estrin_fma` when your deployed evaluator will use Estrin
grouping with explicit fused multiply-add affine combines. This preserves the
Estrin evaluation shape in generated source.

```julia
approx_estrin_fma = approxfit(sin, (-1.0, 1.0);
    target = 1e-8,
    degree = 5,
    piecewise = false,
    scheme = :estrin_fma)

@show approx_estrin_fma.parameters.eval_scheme  # :estrin
@show approx_estrin_fma.parameters.eval_op      # :fma

src_estrin_fma = provide_source(approx_estrin_fma;
    name = :sin_estrin_fma_eval,
    check_domain = true)
```

At the expert-driver level, choose the FMA Estrin scheme and request Estrin
source generation explicitly:

```julia
scheme_estrin_fma = fma_estrin_scheme(5; u = 2.0^-53)
res_estrin_fma = eval_approx_optimize(sin, 5, (-1.0, 1.0), scheme_estrin_fma;
    τ = 1e-3)

src_res_estrin_fma = provide_source(res_estrin_fma;
    name = :sin_estrin_fma_eval2,
    interval = (-1.0, 1.0),
    eval_scheme = :estrin,
    eval_op = :fma)
```

## 2. Function-specific tips for cos and acos

1. For `cos` relative mode, avoid intervals that cross zeros of `cos`.
2. For `acos`, prefer interior intervals first, then widen with piecewise
   constraints if needed.
3. For endpoint-heavy `acos` accuracy demands, budgeted piecewise mode with
   `:min_cost` usually gives better cost/error trade-offs than forcing one
   global high degree.

## 3. Export a standalone evaluator for another program

Use `provide_source` to generate a self-contained Julia function with no
dependency on `PolynomialErrorOptimization` or `Polynomials` at runtime.

```julia
using PolynomialErrorOptimization
import PolynomialErrorOptimization: approximate_abs, eval_approx_optimize

pa = approximate_abs(cos, 4, (-3.0, 3.0), horner_scheme(4; u = 2.0^-53);
    target = 1e-8,
    max_depth = 24)

# Emit standalone source (copy into another Julia project/file).
src = provide_source(pa; name = :cos_piecewise_eval, check_domain = true)
println(src)

# Optionally install directly into the current module/session.
out = provide(pa; name = :cos_piecewise_eval_runtime)
y = out.fn(1.2)
```

For single-polynomial outputs (`OptimResult`), provide an interval if you want
domain checking in the emitted function:

```julia
res = eval_approx_optimize(cos, 6, (-2.0, 2.0), horner_scheme(6; u = 2.0^-53))
src_single = provide_source(res;
    name = :cos_global_eval,
    interval = (-2.0, 2.0),
    check_domain = true)
```

The generated evaluator uses the approximation's target coefficient type by
default. Pass `eval_type` when you want a different arithmetic type in the
standalone function. Pass `eval_op = :fma` when you want explicit fused
multiply-add affine combines, and pass `eval_scheme = :estrin` when you want
Estrin-shaped evaluation instead of Horner-shaped evaluation:

```julia
src_single64 = provide_source(res;
    name = :cos_global_eval64,
    interval = (-2.0, 2.0),
    eval_type = Float64,
    eval_scheme = :estrin,
    eval_op = :fma)
```

### Exporting directly to a file

You can export the generated evaluator source directly to a file for one-step integration into another project using `provide_file`:

```julia
# Piecewise example
file_path = provide_file(pa, "cos_piecewise_eval.jl"; name = :cos_piecewise_eval, check_domain = true)
println("Wrote evaluator to: ", file_path)
println(read(file_path, String))

# Single-polynomial example
file_path2 = provide_file(res, "cos_global_eval.jl";
    name = :cos_global_eval,
    interval = (-2.0, 2.0),
    check_domain = true)
println("Wrote evaluator to: ", file_path2)
println(read(file_path2, String))
```
