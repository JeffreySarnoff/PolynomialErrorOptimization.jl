# Using ArgCheck.jl in PolynomialErrorOptimization

## Why ArgCheck.jl is a good fit here

This repository has many explicit input checks and throws, especially in:

- src/exchange/solve_primal.jl
- src/exchange/exchange.jl
- src/exchange/find_new_index.jl
- src/core/numerics.jl
- src/exchange/driver.jl

Those checks are correct, but can become verbose and repetitive. ArgCheck.jl gives a concise, consistent way to express preconditions.

## What ArgCheck.jl improves

1. Clearer intent
- Preconditions read as direct assertions near function entry.

2. Less boilerplate
- Replaces repeated pattern: condition || throw(...).

3. Consistent diagnostics
- Standardized failure style across the codebase.

4. Easier maintenance
- Validation logic is shorter and easier to audit.

## Recommended usage style for this codebase

Use ArgCheck for argument and shape preconditions only.

Good candidates:

1. Dimension checks
- length(ω) == m
- length(y) == m
- length(a) == scheme.n + 1

2. Domain checks
- τ > 0
- I[1] < I[2]
- s in valid range for RelativeZeroMode

3. Scheme consistency checks
- scheme.n == n
- output vector lengths in in-place kernels

Keep explicit throw(DomainError(...)) where value-specific context matters.

## Migration pattern

### Before

~~~julia
length(ω) == m || throw(DimensionMismatch(
    "solve_primal: length(ω) = $(length(ω)) ≠ n+2 = $m"))
~~~

### After

~~~julia
@argcheck length(ω) == m DimensionMismatch(
    "solve_primal: length(ω) = $(length(ω)) ≠ n+2 = $m")
~~~

This keeps your existing error type and message while reducing ceremony.

## Suggested file-by-file adoption

### 1) src/exchange/solve_primal.jl

Use @argcheck for:

- length(ω) consistency in each mode branch.
- RelativeZeroMode expected m expression.

### 2) src/exchange/exchange.jl

Use @argcheck for:

- length(ω), length(y) checks.
- mode-dependent m checks.

Keep ExchangeFailure throws unchanged, since they represent algorithmic failure, not argument validation.

### 3) src/exchange/find_new_index.jl

Use @argcheck for:

- coefficient length checks.

### 4) src/exchange/driver.jl

Use @argcheck for:

- n >= 0
- τ > 0
- interval ordering
- max_iter >= 0
- scheme degree consistency

### 5) src/core/numerics.jl and src/core/constraints.jl

Use @argcheck for:

- α! output length and signature length
- dot_view dimension match

Keep DomainError in α_relzero when f(t) is zero, since the offending value is meaningful API feedback.

## Practical advantages in this project

1. Better readability in algorithm-heavy files
- solve_primal.jl and exchange.jl become easier to scan for actual linear-algebra logic.

2. Lower risk during refactors
- Centralized, compact checks reduce copy-paste divergence.

3. Cleaner tests and debugging
- More uniform failure behavior helps pinpoint invalid inputs quickly.

## Caveats

1. Do not replace all throws blindly
- Keep specialized exceptions for semantic failures (for example ExchangeFailure, ConvergenceFailure).

2. Preserve informative messages
- ArgCheck can keep custom messages and custom exception types; use that to avoid losing context.

3. Keep hot loops clean
- Put argument checks at boundaries, not in per-iteration inner loops unless necessary.

## Minimal integration example

~~~julia
# Project.toml
[deps]
ArgCheck = "dce04be8-c92d-5529-be00-80e4d2c0e197"
~~~

~~~julia
# in source files
using ArgCheck

function solve_primal(f, scheme::EvalScheme{T}, ω::Vector{Index{T}}) where {T<:AbstractFloat}
    m = scheme.n + 2
    @argcheck length(ω) == m DimensionMismatch(
        "solve_primal: length(ω) = $(length(ω)) ≠ n+2 = $m")
    # ...
end
~~~

## Suggested rollout

1. Phase 1
- Add ArgCheck.jl dependency.
- Convert checks in solve_primal.jl and exchange.jl only.
- Run full tests.

2. Phase 2
- Convert find_new_index.jl and driver.jl.

3. Phase 3
- Convert remaining structural checks in core helper files.
- Leave semantic exception paths unchanged.

## Bottom line

ArgCheck.jl is a strong readability and maintainability upgrade for this repository. It is best used for precondition checks, while preserving your current custom exceptions for algorithmic and domain-specific failure paths.
