# Contributor Guide

This page is for contributors working on package structure, API contracts, and validation.

## Architecture principles

The current package shape was delivered in stages. Keep these principles in
place when extending it.

### Phase 1: clarify the package surface

- define the intended stable user surface,
- document expert and internal APIs clearly,
- keep the README product-facing,
- split tests by subsystem,
- keep validation centered on `Pkg.test()` and the docs build.

### Phase 2: improve configuration and diagnostics

- introduce typed configuration and planning objects,
- keep the keyword and typed planning paths aligned,
- replace lossy rejection strings with structured reports,
- add tests and smoke checks that pin both the keyword and typed paths.

### Phase 3: harden boundaries

- make relative-zero representation explicit,
- narrow the root export story,
- promote stricter docs validation in CI.

The current phase-3 state keeps `OptimResult.poly` monomial-basis even for
relative-zero solves, exposes solve-basis metadata through `basis_info` and
`solution_coefficients`, and runs dedicated docs smoke tests in CI before the
Documenter build.

## Current audience split

- Stable workflow layer: `approxfit`, `fit_abs`, `fit_rel`, `recommend_parameters`, result helpers, and scheme builders.
- Expert layer: fixed-degree drivers, piecewise drivers, search strategies, and evaluator export.
- Advanced layer: exchange substeps, symbolic expression nodes, and low-level formulation internals.

Contributors should preserve that split when adding new features.

## Local validation

Run both from the package root:

```julia
using Pkg
Pkg.test()
```

```julia
julia --project=docs docs/make.jl
```

For the docs-backed examples alone, run:

```julia
julia --project=. -e 'include("test/common.jl"); include("test/docs/docs_tests.jl")'
```

## Test layout

- `test/core/`: low-level numerics and formulation invariants.
- `test/schemes/`: scheme construction and symbolic error models.
- `test/exchange/`: fixed-degree drivers and exchange-loop behavior.
- `test/piecewise/`: adaptive partitioning and budget behavior.
- `test/interface/`: high-level API and code generation.
- `test/docs/`: smoke tests for examples mirrored from the manual.

Add regressions at the narrowest layer that can prove the intended behavior.
For phase-2 changes specifically, keep typed-planning regressions in
`test/interface/` and structured diagnostic regressions in `test/piecewise/`.
For phase-3 relative-zero work, keep basis/representation regressions in
`test/exchange/` and docs-validation regressions in `test/docs/`.

## Documentation responsibilities

- put the recommended user path in the stable workflow docs,
- keep expert details in the user or technical guides,
- keep contributor and redesign material here rather than on the landing page,
- update the API page when a symbol changes layer.
