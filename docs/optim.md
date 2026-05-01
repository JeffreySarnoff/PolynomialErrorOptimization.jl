# Using Optim.jl in PolynomialErrorOptimization

## Implementation status

Optim.jl is now integrated in this repository.

Implemented pieces:

- `Optim` added to `Project.toml` dependencies.
- `GridThenOptim` added as a `SearchStrategy` in `src/exchange/search.jl`.
- `locate_maximum(::GridThenOptim)` added in `src/exchange/search.jl`.
- `GridThenOptim` exported from `src/PolynomialErrorOptimization.jl`.
- Regression tests added in `test/runtests.jl`.

## Why Optim.jl is relevant here

This codebase already has a clear maximum-search abstraction:

- locate_maximum in src/exchange/search.jl
- strategy types like GridSearch and GridThenLocal
- find_new_index in src/exchange/find_new_index.jl built on that abstraction

That structure is ideal for an optional local-refinement backend without changing the main algorithm.

## Best fit in this repository

The strongest use case is one-dimensional bounded optimization in the maximum search step.

Current objective shape in find_new_index is:

- absolute mode: maximize total error envelope over t in I
- relative modes: maximize normalized total error over t in I

Because Optim.jl is a minimizer, the standard pattern is:

- define h(t) = -objective(t)
- minimize h on closed interval I
- convert back by astar = -minimum(h)

## Where it is integrated

### 1) Keep existing coarse grid pass

Grid scan is robust to nonsmooth points from absolute values and gives a reliable basin.

### 2) Optim-based local refinement strategy

The strategy is implemented as:

~~~julia
struct GridThenOptim <: SearchStrategy
    M::Int
    bracket::Int
end
~~~

In `locate_maximum` for this strategy:

1. Run the same coarse grid used by `GridThenLocal`.
2. Build a local interval around the best grid index using `bracket`.
3. Minimize `h(t) = -g(t)` with bounded Brent.
4. Convert back to a maximum and keep the better of sampled/refined values.

This preserves your existing design and testability.

## Optim method used

For 1D bounded local refinement, this code uses Brent on a bracketed interval.

Why:

1. Derivative-free.
2. Stable for nonsmooth or piecewise-smooth objectives.
3. Good fit for abs-heavy objective expressions in this project.

Core pattern used:

~~~julia
using Optim

function refine_with_optim(g, a, b)
    h(t) = -g(t)
    result = optimize(h, a, b, Brent())
    tstar = Optim.minimizer(result)
    astar = -Optim.minimum(result)
    return tstar, astar
end
~~~

## Practical advantages for this source code

### 1) Better peak localization at fixed grid cost

A coarse grid plus local refinement often matches the quality of a much denser grid while reducing evaluations.

### 2) Cleaner strategy layering

Your code already separates strategy from objective, so Optim.jl drops in naturally without touching solve_primal or exchange logic.

### 3) Easier experimentation and comparison

You can compare GridSearch, GridThenLocal, and GridThenOptim with minimal code churn.

### 4) Potentially tighter astar in difficult intervals

Sharper astar can reduce extra exchange iterations when convergence is sensitive to the max-search quality.

## Caveats and how to handle them

1. Objective nonsmoothness near sign changes:
- Keep grid-first behavior.
- Use local bounded refinement only after bracketing.

2. Local optimum risk:
- Restrict refinement to a bracket around the best sampled point.
- Keep the best sampled value and return max of sampled and refined if needed.

3. Determinism in tests:
- Keep deterministic settings and fixed strategy parameters.
- Retain existing grid-only strategies for strict reproducibility tests.

## Default policy and usage

`default_strategy(scheme)` remains `GridSearch(...)` for reproducibility.
`GridThenOptim` is opt-in at call sites.

Use it like this:

~~~julia
ωstar, astar = find_new_index(f, scheme, I, a;
    strategy=GridThenOptim(10_001; bracket=4))
~~~

Test coverage in `test/runtests.jl` verifies that this strategy:

- returns a point inside `I`,
- preserves signature consistency,
- does not underperform the coarse sampled objective in the tested setup.

## Best-practice guidance for this codebase

1. Keep `GridSearch` as default.
2. Use `GridThenOptim` for tighter maxima when extra local work is acceptable.
3. Choose `M` first, then tune `bracket`:
- start with `bracket = 3` or `4`,
- increase `M` for global resolution,
- increase `bracket` only when local basin capture is inadequate.
4. For strict reproducibility baselines, keep `GridSearch` in tests/benchmarks.

## Bottom line

Optim.jl is integrated where it helps most: maximum search. `GridThenOptim` gives a robust grid-first + Brent-local-refinement path that improves peak localization while preserving the existing algorithm structure and default behavior.
