# High-Level Interface

This is the stable workflow layer. Use it for the common question:

> Approximate this function on this interval to this tolerance, optionally
> under a coefficient budget.

It is built on top of the expert APIs. Stay here unless you specifically need:

- a fixed evaluation scheme and explicit search strategy,
- direct access to the exchange loop,
- piecewise degree-policy control,
- symbolic custom evaluation-error models.

## Minimal Inputs

The minimal call is:

```julia
using PolynomialErrorOptimization

approx = approxfit(sin, (-3.0, 3.0); target = 1e-3, effort = :fast, max_depth = 6)
```

You provide:

- `f`: the function to approximate.
- `I`: the interval `(left, right)`.
- one tolerance: `target`, `abs_tol`, or `rel_tol`.

By default this chooses an absolute-error, piecewise, Horner-based
approximation with balanced settings.

## Choosing the Tolerance

Use one of these:

```julia
approxfit(f, I; target = 1e-8)   # objective-space target
approxfit(f, I; abs_tol = 1e-8)  # absolute target
approxfit(f, I; rel_tol = 1e-8)  # relative tolerance
```

For `mode = :rel`, `rel_tol` is used directly as the relative target:

```julia
approx = approxfit(exp, (0.5, 2.5);
    mode = :rel,
    rel_tol = 1e-9)
```

For `mode = :abs`, `rel_tol` is converted to an absolute target by sampling
the scale of `f` on `I`.

## Common Workflows

Before choosing keywords, see [Choosing a Workflow](choosing-a-workflow.md) for the capability matrix.

### Let the package choose a coefficient budget

```julia
approx = approxfit(sin, (-3.0, 3.0);
    target = 1e-8,
    effort = :balanced)
```

This chooses `max_coeffs` from the effort preset and calls the budget-aware
piecewise driver.

### Fixed global polynomial

```julia
approx = approxfit(exp, (-1.0, 1.0);
    target = 1e-10,
    degree = 6,
    piecewise = false)
```

This calls the fixed-degree exchange driver and returns a single-polynomial
`Approximation`.

### FMA Horner model

Use `scheme = :horner_fma` when the deployed polynomial will be evaluated with
explicit fused multiply-add Horner steps. This uses an FMA-specific
evaluation-error model instead of the default separate multiply/add Horner
model.

```julia
approx = approxfit(sin, (-1.0, 1.0);
    target = 1e-8,
    degree = 5,
    piecewise = false,
    scheme = :horner_fma)
```

For high-level `Approximation` values fit with `scheme = :horner_fma`,
`provide_source(approx; ...)` emits `fma(...)` Horner steps by default. You can
override that with `eval_op = :muladd` if you explicitly want the old generated
source shape.

### FMA Estrin model

Use `scheme = :estrin_fma` when the deployed polynomial will be evaluated with
Estrin grouping and explicit fused multiply-add affine combines. Power
construction still uses rounded multiplication, while coefficient combines use
`fma(...)`.

```julia
approx = approxfit(sin, (-1.0, 1.0);
    target = 1e-8,
    degree = 5,
    piecewise = false,
    scheme = :estrin_fma)
```

For high-level `Approximation` values fit with `scheme = :estrin_fma`,
`provide_source(approx; ...)` emits Estrin-shaped source with `fma(...)`
combines by default. Expert-driver users can request the same source shape with
`provide_source(res; eval_scheme = :estrin, eval_op = :fma)`.

### Try global first, then piecewise if needed

```julia
approx = approxfit(sin, (-3.0, 3.0);
    target = 1e-8,
    degree = 5,
    piecewise = :auto)
```

With a degree provided and `piecewise = :auto`, `approxfit` first tries one global
polynomial. If its verified error bound is above `target`, it falls back to
fixed-degree piecewise approximation.

### Hard coefficient budget

```julia
approx = approxfit(acos, (-0.95, 0.95);
    target = 5e-8,
    max_coeffs = 7,
    total_coeffs = 120,
    effort = :accurate)
```

This calls the budget-aware piecewise API with a per-piece coefficient cap and
a global coefficient cap.

### Separate target and computation types

Use `target_type` for both the returned polynomial coefficient type and the
floating-point format modeled by the default Horner/Estrin error scheme
(`u = eps(target_type)/2`). Use `compute_type` for internal nodes, solves,
search objectives, and verified error bounds:

```julia
approx = approxfit(sin, (-1.0, 1.0);
    target = 1e-6,
    degree = 5,
    piecewise = false,
    target_type = Float32,
    compute_type = BigFloat)
```

This computes in `BigFloat`, models Float32 rounding through the default
scheme, and returns Float32 coefficients. The lower-level fixed-degree
drivers take their computation type from `EvalScheme{T}` and accept
`target_type` to control the returned coefficient type.

## Inspecting Results

`approxfit` returns an `Approximation`, which is callable:

```julia
y = approx(0.25)
```

Use helper functions to inspect the result independent of whether it is a
single polynomial or piecewise:

```julia
error_bound(approx)
coeff_count(approx)
is_piecewise(approx)
pieces(approx)       # returns nothing for a single polynomial
```

The wrapped expert result is available as:

```julia
approx.model
```

## Parameter Recommendations Without Running a Fit

Use `recommend_parameters` when you want to inspect or modify the choices
before running:

```julia
params = recommend_parameters(sin, (-3.0, 3.0);
    target = 1e-8,
    effort = :balanced)

approx = approxfit(sin, (-3.0, 3.0), params)
```

The helper requires the same minimal information as `approxfit`: `f`, `I`, and one
tolerance. Optional hints are:

- `mode = :abs` or `:rel`
- `effort = :fast`, `:balanced`, or `:accurate`
- `degree` for fixed-degree workflows
- `max_coeffs` and `total_coeffs` for budgeted workflows
- `scheme = :horner`, `:horner_fma`, `:estrin`, or `:estrin_fma`
- `target_type = Float64` or another floating type for returned coefficients
  and the default finite-precision error model
- `compute_type = target_type` or another floating type for internal
  optimization arithmetic

`recommend_parameters` samples `f` on `I` to estimate scale and variation.
That sampling is used to convert `rel_tol` in absolute mode, modestly raise
piecewise defaults for harder intervals, and reject relative-mode requests
whose sampled values cross or touch zero.

## Planning Before Execution

Use `plan_fit` when you want a structured plan object instead of just the
flattened `FitParameters` bundle.

### Keyword-driven planning

```julia
plan = plan_fit(sin, (-3.0, 3.0);
    target = 1e-3,
    effort = :fast,
    max_depth = 6)

approx = approxfit(sin, plan)
```

`FitPlan` records:

- the normalized interval,
- the typed objective,
- complexity, precision, and search specs,
- the executable `FitParameters`,
- which fields were inferred from defaults.

### Typed planning objects

Use the typed specs when the workflow should be explicit about intent rather
than expressed as a keyword bundle.

```julia
objective = ObjectiveSpec(:rel, 1e-8)
complexity = ComplexitySpec(degree = 5, piecewise = false)
precision = PrecisionSpec(target_type = Float32, compute_type = BigFloat)
search = SearchSpec(BigFloat;
    scheme = :horner_fma,
    effort = :fast,
    τ = 1e-3,
    max_depth = 8,
    driver_max_iter = 80,
    strategy = PolynomialErrorOptimization.GridSearch(4096))

plan = plan_fit(exp, (0.5, 1.5), objective;
    complexity = complexity,
    precision = precision,
    search = search)

approx = approxfit(exp, plan)
```

The keyword path and the typed path are equivalent front doors into the same
planning model. Use keywords for quick iteration and typed specs when the
problem definition should be explicit, inspectable, and easy to persist.

If you want to choose the workflow before choosing the parameters, start with
[Choosing a Workflow](choosing-a-workflow.md).

## Effort Presets

| Effort | Use when | Default behavior |
| --- | --- | --- |
| `:fast` | exploring shape and feasibility | smaller coefficient budget, lower iteration cap |
| `:balanced` | normal production starting point | moderate budget and conservative search |
| `:accurate` | tight tolerances or sharp features | larger budget, more iterations, `GridThenOptim` for fixed-degree search |

The presets are starting points. If a run fails due to depth or convergence
limits, increase `max_depth`, `driver_max_iter`, or `max_coeffs` before
tightening the target further.

## Exporting a Fitted Approximation

`provide_source` and `provide` accept `Approximation`:

```julia
src = provide_source(approx; name = :sin_eval)
out = provide(approx; name = :sin_eval_runtime)
```

For single-polynomial fits, `provide_source(approx; ...)` uses the interval
stored in the `Approximation` for domain checking.
