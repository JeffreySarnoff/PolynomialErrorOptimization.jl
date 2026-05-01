# FMA Support

This note describes the FMA support added to the package: polynomial
determination whose error model assumes fused multiply-add (FMA), and
polynomial evaluation that actually uses FMA.

The support is additive. The current non-FMA Horner/Estrin paths remain
available and keep their current defaults. FMA is opt-in through separate
scheme/evaluation choices, not a replacement for the existing behavior.

## Implemented Surface

- `fma_horner_scheme(n, T=Float64; u=eps(T)/2)` models one rounded FMA per
  Horner step.
- `fma_horner_expr(n; u=2.0^-53)` builds the matching symbolic error tree.
- `fma_horner_eval(p, t)` evaluates a coefficient vector with explicit `fma`.
- `fma_estrin_scheme(n, T=Float64; u=eps(T)/2)` models Estrin evaluation with
  one rounded FMA per affine combine.
- `fma_estrin_expr(n; u=2.0^-53)` builds the matching symbolic Estrin tree.
- `fma_estrin_eval(p, t)` evaluates a coefficient vector with Estrin grouping
  and explicit `fma` combines.
- `FMA(x, y, z)` represents an exact fused multiply-add expression in the
  symbolic rounding-error AST.
- `approxfit(...; scheme=:horner_fma)` fits with the FMA Horner model.
- `approxfit(...; scheme=:estrin_fma)` fits with the FMA Estrin model.
- `provide_source(...; eval_op=:fma)` emits explicit `fma(...)` affine
  combines. `provide_source(...; eval_scheme=:estrin, eval_op=:fma)` emits
  Estrin-shaped FMA evaluation. High-level `Approximation` values fit with
  `scheme=:horner_fma` or `scheme=:estrin_fma` use the matching export mode
  automatically unless overridden.

## Compatibility Goals

- `horner_scheme`, `horner_expr`, `horner_eval`, `estrin_scheme`,
  `estrin_expr`, `estrin_eval`, and `provide_source(...; eval_op=:muladd)`
  behavior unchanged.
- `scheme = :horner` remains the high-level default.
- FMA is a separate scheme: `fma_horner_scheme`, `fma_estrin_scheme`,
  `scheme = :horner_fma`, and `scheme = :estrin_fma`.
- Guaranteed-FMA generated evaluation is explicit:
  `provide_source(...; eval_op=:fma)`. Estrin-shaped generated evaluation is
  explicit for expert-driver results: `provide_source(...; eval_scheme=:estrin)`.
- Existing conservative workflows are preserved: users who do nothing get the
  same error model and evaluator source shape they had before FMA support.

## Background

The package already emits and uses Horner-style evaluators with `muladd` in a
few places:

- `horner_eval` uses `muladd(r, t, a[k])`.
- generated evaluators from `provide_source` use `muladd(y, t, coeffs[j])`.
- `monomial_dot` and `dot_view` also use `muladd` for local accumulation.

However, `muladd` is not the same API contract as FMA-aware polynomial
determination:

- `muladd(x, y, z)` permits the compiler/runtime to fuse, but does not by
  itself encode the package's evaluation-error model.
- `horner_scheme` currently models the standard Horner scheme as a rounded
  multiplication followed by a rounded addition. That is a two-rounding model
  per Horner step, matching the paper's non-FMA Horner analysis.
- FMA Horner has one rounded operation per step:

  ```julia
  r = fma(r, t, a[k])
  ```

  and therefore needs a different `EvalScheme`.

So FMA support is not just "call `muladd`". The fitting/determination path must
optimize against the same FMA error model that the exported evaluator uses,
while retaining the current non-FMA model as the default.

## Implementation Details

### 1. FMA Horner Evaluation Scheme

The public scheme builder alongside `horner_scheme` is:

```julia
fma_horner_scheme(n::Integer, ::Type{T}=Float64; u::Real=eps(T)/2) where T
```

This returns `EvalScheme{T}` for Horner evaluation performed as FMA:

```julia
r = a[n]
for k in (n-1):-1:0
    r = RN(r * t + a[k])
end
```

For degree `n`, this has `n` rounding-error rows, not the current `n + 1`
rows used by `horner_scheme`.

For `n == 0`, there are no arithmetic operations. Keep a single zero row for
compatibility with existing code paths, as `horner_scheme(0)` does today.

For `n > 0`, the row for the FMA at coefficient index `k`, where
`k = 0, ..., n-1`, is:

```text
u * (0, ..., 0, t^k, t^(k+1), ..., t^n)
```

That row represents the final-output contribution of the single rounding in:

```text
RN(a_k + t * (a_{k+1} + ...))
```

This is the FMA analogue of the current closed-form Horner rows. It is
implemented directly for speed and clarity, with tests against the symbolic
builder. `horner_scheme` is unchanged; callers must opt into this model.

### 2. Symbolic Error AST With FMA

The current AST has `Add`, `Mul`, and `Round`. It can model FMA as:

```julia
Round(Add(Mul(x, y), z), u, id)
```

but that representation is easy to confuse with the non-FMA tree:

```julia
Round(Add(Round(Mul(x, y), u, id_mul), z), u, id_add)
```

The explicit node is:

```julia
struct FMA <: ErrExpr
    x::ErrExpr
    y::ErrExpr
    z::ErrExpr
end
```

Then use:

```julia
Round(FMA(x, y, z), u, id)
```

Symbolic support includes:

- `Base.show` for `FMA`.
- `lin_eval_error` product/addition rule for `FMA`:

  ```text
  d(x*y + z) = y*dx + x*dy + dz
  ```

- `extract_aj_coeff` for `FMA`, preserving the existing linear-in-coefficients
  invariant.
- `depends_on_aj`, `strip_a`, and `compile_t` support for `FMA`.

This makes custom FMA-based evaluation schemes possible and gives a clean
symbolic reference for testing `fma_horner_scheme`. Existing `Add`/`Mul`/`Round`
trees remain the representation for non-FMA arithmetic.

### 3. `fma_horner_expr`

The public symbolic builder is:

```julia
fma_horner_expr(n; u = 2.0^-53) -> (ErrExpr, Dict{Int,Float64})
```

It builds:

```julia
r = VarA(n)
for k in (n-1):-1:0
    r = rnd(FMA(r, VarT(), VarA(k)))
end
```

where `rnd(e)` wraps the exact FMA expression in one `Round`.

This differs from the existing `horner_expr`, which uses:

```julia
r = rnd(simplify_add(rnd(simplify_mul(r, VarT())), VarA(k)))
```

The new symbolic expression gives an independent validation path:

```julia
θ = lin_eval_error(fma_horner_expr(n)[1])
sym = build_eval_scheme(θ, ids_to_u, n, "fma-horner-symbolic", T)
closed = fma_horner_scheme(n, T; u)
```

The combined envelopes from `sym` and `closed` match at sampled points.

### 4. FMA Estrin Evaluation Scheme

The Estrin FMA surface mirrors Horner FMA:

```julia
fma_estrin_scheme(n::Integer, ::Type{T}=Float64; u::Real=eps(T)/2) where T
fma_estrin_expr(n; u = 2.0^-53)
fma_estrin_eval(p::AbstractVector{<:Real}, t::Real)
```

Affine combines are represented as one rounded `FMA`:

```julia
Round(FMA(high, power, low), u, id)
```

The powers `t², t⁴, ...` remain rounded multiplications, matching Estrin's
power-building tree. This gives a distinct model from both non-FMA Estrin and
FMA Horner.

### 5. Explicit Evaluation Mode for Generated Evaluators

`provide_source` currently emits Horner evaluation using `muladd`. Keep that
as the default. To guarantee FMA semantics in addition to the current mode, use
the evaluation operation keyword:

```julia
provide_source(model; eval_type = <coefficient type>, eval_op = :muladd)
```

Supported values:

- `eval_op = :muladd`: current behavior.
- `eval_op = :fma`: emit `fma(y, t, coeffs[j])`.
- `eval_scheme = :horner`: emit Horner-shaped evaluation.
- `eval_scheme = :estrin`: emit Estrin-shaped evaluation.

For example:

```julia
y = fma(y, t, approx_T(coeffs[j]))
```

The same keyword flows through:

- `provide_source(::OptimResult)`
- `provide_source(::PiecewisePolyApprox)`
- `provide(::Approximation)`
- `provide_file`

This separates "evaluate with compiler-permitted multiply-add contraction" from
"evaluate with the language-level fused operation", without changing existing
generated code unless the caller opts in.

### 6. Public Runtime FMA Evaluator

The runtime evaluator alongside `horner_eval` is:

```julia
fma_horner_eval(p::AbstractVector{<:Real}, t::Real)
```

This mirrors `horner_eval`, but uses `fma` instead of `muladd`:

```julia
function fma_horner_eval(p::AbstractVector{<:Real}, t::Real)
    T = typeof(float(t))
    isempty(p) && return zero(T)
    @inbounds r = T(p[end])
    @inbounds for k in (length(p)-1):-1:1
        r = fma(r, T(t), T(p[k]))
    end
    return r
end
```

It is exported as user-facing functionality. `horner_eval` is unchanged.

### 7. High-Level Interface

The high-level interface accepts:

```julia
scheme = :horner | :horner_fma | :estrin | :estrin_fma
```

The `:horner_fma` and `:estrin_fma` options keep scheme selection as a single
enum. They include:

- `_FIT_SCHEMES` includes `:horner_fma` and `:estrin_fma`.
- `_scheme_for` calls `fma_horner_scheme` or `fma_estrin_scheme`.
- docs and examples cover the option.
- generated evaluators use the matching `eval_scheme` and `eval_op = :fma`
  when the approximation was fit with an FMA scheme, preserving the intended
  evaluation semantics.

`FitParameters` stores `eval_op::Symbol`, and `provide_source(::Approximation)`
uses that value unless the caller explicitly overrides `eval_op`.

### 8. FMA Metadata in Results

`OptimResult` stores `poly`, error bounds, and discretization state, but not
the evaluation operation used to build the scheme. If users pass an arbitrary
`EvalScheme`, there is no reliable way to know whether the result was intended
for FMA emission.

The implemented choice is to keep `OptimResult` unchanged and store `eval_op`
in the high-level `Approximation` path. This lets
`approxfit(...; scheme=:horner_fma)` and
`approxfit(...; scheme=:estrin_fma)` export matching FMA evaluators
automatically, while expert-driver users still pass
`provide_source(res; eval_op=:fma)` and, for Estrin,
`provide_source(res; eval_scheme=:estrin, eval_op=:fma)` explicitly.

## Correctness Requirements

The FMA fitting path is correct only if these three choices agree:

1. The `EvalScheme` used by the optimizer models FMA rounding.
2. The emitted/runtime evaluator actually uses FMA.
3. The target rounding unit `u` matches the intended hardware/format.

If the optimizer uses `fma_horner_scheme` but the deployed evaluator uses
separate multiply and add, the verified error bound is too optimistic.

If the optimizer uses regular `horner_scheme` but the deployed evaluator uses
FMA, the result is conservative but may be suboptimal because the optimizer
paid for rounding errors that do not occur.

## Validation Tests

Tests cover:

- `fma_horner_scheme(0)` shape and zero row.
- `fma_horner_scheme(n)` rows for small `n`, especially `n = 1, 2, 3`.
- symbolic `fma_horner_expr` envelope matches closed-form
  `fma_horner_scheme`.
- FMA Horner has fewer rows than standard Horner for `n > 0`.
- an end-to-end `eval_approx_optimize` smoke test with `fma_horner_scheme`.
- `provide_source(...; eval_op=:fma)` emits `fma(` and not `muladd(` in the
  Horner loop.
- generated evaluator with `eval_op=:fma` agrees with `fma_horner_eval`.
- high-level `approxfit(...; scheme=:horner_fma)` uses the FMA scheme and,
  exports with `eval_op=:fma`.
- `fma_estrin_scheme` and `fma_estrin_expr` agree symbolically.
- `fma_estrin_eval` matches ordinary polynomial values.
- high-level `approxfit(...; scheme=:estrin_fma)` uses the FMA Estrin scheme
  and exports with `eval_scheme=:estrin, eval_op=:fma`.

## Documentation Coverage

Updated:

- `README.md`: add short examples using `scheme = :horner_fma` and
  `scheme = :estrin_fma`.
- `docs/src/high-level-interface.md`: document scheme selection.
- `docs/src/examples.md`: add an FMA example.
- `docs/src/api.md`: list `fma_horner_scheme`, `fma_horner_expr`,
  `fma_horner_eval`, `fma_estrin_scheme`, `fma_estrin_expr`, and
  `fma_estrin_eval`.
- `provide_source` docstrings: document `eval_op`.

## Implementation Order Used

1. Implement `fma_horner_scheme` directly and test its rows.
2. Add `eval_op` to generated evaluators and `fma_horner_eval`.
3. Add symbolic `FMA` and `fma_horner_expr`.
4. Cross-check symbolic and closed-form FMA schemes.
5. Wire `:horner_fma` into `approxfit`.
6. Add automatic FMA export for high-level `Approximation`.
7. Add the parallel Estrin FMA scheme, evaluator, generated-source shape, and
   high-level `:estrin_fma` wiring.
8. Rebuild docs and run the full test suite.
