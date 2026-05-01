# Redesign Recommendations

## Context

This repository already has the shape of a serious numerical package rather than a toy prototype:

- the core exchange algorithm is split into coherent source files,
- the piecewise layer and the high-level interface exist,
- the test suite is substantial and currently passes,
- the package clearly aims to serve both paper-level reproducibility and practical approximation workflows.

That said, the first-pass design still reflects the structure of the paper more strongly than the structure of the product you likely want users to experience. The main redesign goal should be to preserve the algorithmic core while making the package easier to learn, safer to extend, and clearer about which APIs are stable versus expert-only.

## Review Summary

### What is already strong

- The implementation is decomposed by responsibility rather than kept in one monolithic file.
- The package has a credible high-level surface in `src/interface.jl`, which is the right direction for a user-facing package.
- Type separation between returned coefficients and internal arithmetic is a good design choice and worth keeping.
- The test suite exercises both algorithmic details and public workflows.

### Main design pressure points

1. The root module currently exports too many layers of abstraction at once.
2. The public configuration model relies heavily on `Symbol` values and mutually exclusive keyword combinations.
3. The package exposes paper-building blocks as peer-level public API alongside the intended end-user workflows.
4. Failure handling is algorithmically reasonable but not yet product-grade in how it communicates rejection and recovery.
5. The docs are informative, but the information architecture is still closer to a research note than a guided package experience.
6. Validation of documentation drift is weak because docs are not doctested and missing-doc checks are disabled.

## Evidence From The Current Codebase

### 1. The root namespace is overloaded

`src/PolynomialErrorOptimization.jl` exports all of the following from the top level:

- end-user workflows such as `approxfit`,
- expert drivers such as `eval_approx_optimize`,
- low-level exchange steps such as `init_points`, `solve_primal`, `find_new_index`, and `exchange`,
- symbolic AST nodes such as `ErrExpr`, `Round`, `Mul`, and `FMA`.

That makes the package flexible, but it also makes the stable package story ambiguous. A new user cannot tell which names are the intended entry points, which are advanced hooks, and which are research internals that may legitimately change.

### 2. The high-level API is doing too much argument interpretation

`src/interface.jl` is a strong start, but it currently validates and coordinates a large keyword matrix:

- `mode`, `effort`, `piecewise`, `scheme`, `degree_policy` are all `Symbol`-driven,
- `degree`, `max_coeffs`, and `total_coeffs` interact in nontrivial ways,
- fixed-degree and budget-driven flows share one surface but still require different mental models,
- special sentinel behavior such as `:auto` and `_AUTO_STRATEGY` is embedded in keyword interpretation.

This is workable for a first pass, but it does not scale well for maintenance or discoverability.

### 3. Piecewise rejection reasons are not first-class diagnostics yet

`src/piecewise/fitting.jl` currently catches many internal failures and compresses them to `(nothing, false, string(typeof(e)))` or short rejection strings such as `"err > target"`.

That is enough for internal recursion, but it loses information that would be valuable for:

- user-facing diagnostics,
- tuning guidance,
- debugging difficult functions,
- future telemetry or benchmarking.

### 4. Some mathematical edge cases leak through the surface awkwardly

The package already documents that `RelativeZeroMode` is only fully supported for `t_z = 0`, and `src/exchange/solve_primal.jl` warns when `t_z != 0` because the returned polynomial is effectively represented in a shifted basis.

This is a good example of where the implementation is honest, but the representation model is not yet aligned with the mathematical model. That is a redesign smell more than a bug.

### 5. The test suite is broad but concentrated

`test/runtests.jl` currently carries the whole test surface in one large file. The suite passes and covers a lot, but the structure makes it harder to:

- map tests back to subsystems,
- add focused regression cases,
- gate future refactors by behavior slice,
- grow documentation-backed tests.

### 6. Docs are useful, but they flatten audiences together

The README, home page, user guide, technical guide, and API page all contain good material, but the current doc set still presents:

- high-level workflows,
- research algorithm descriptions,
- extension points,
- full API inventories,

as near-equal peers.

In addition, `docs/make.jl` sets `doctest=false` and `checkdocs=:none`, which means the docs are intentionally not enforcing code-example correctness or docstring coverage.

## Recommended Redesign

## 1. Split the package into explicit audience layers

The package currently serves at least three audiences:

1. application users who want an approximation and do not care about the exchange loop,
2. expert numerical users who want to choose schemes, strategies, and modes,
3. contributors or researchers who want access to the raw algorithmic building blocks.

Those layers should exist explicitly in the API design.

### Layering proposal

Adopt a layered public surface such as:

- Stable workflow layer:
  - `approxfit`
  - `fit_abs`
  - `fit_rel`
  - result inspection helpers
  - evaluator export helpers
- Expert layer:
  - fixed-degree drivers
  - piecewise drivers
  - search strategies
  - scheme builders
- Internal or advanced layer:
  - exchange substeps
  - symbolic AST nodes
  - row constructors and low-level numerics

In Julia terms, this could be done either through submodules or through a stricter export policy with clearly documented namespaces. The important point is not the mechanism; it is making the stability boundary visible.

### Expected impact

- It reduces namespace noise.
- It clarifies what can change during refactors.
- It makes the docs much easier to organize.
- It lets you keep the research value without making the casual user pay the cognitive cost.

## 2. Replace keyword soup with typed problem specifications

The current interface is keyword-heavy because it is encoding several orthogonal concerns in one call:

- objective mode,
- degree or budget policy,
- arithmetic model,
- search policy,
- piecewise policy,
- heuristic effort level.

### Configuration proposal

Introduce small configuration types with clear responsibilities. For example:

- `ObjectiveSpec`: absolute, relative, relative-with-zero,
- `ComplexitySpec`: fixed degree, per-piece max coefficients, total coefficient cap,
- `PrecisionSpec`: target type, compute type, modeled evaluation type,
- `SearchSpec`: search strategy and convergence tolerance,
- `PartitionSpec`: fixed polynomial, adaptive piecewise, budget-aware piecewise.

Then let `approxfit` accept either a high-level convenience keyword path or a single structured spec object.

### Expected benefits

- Fewer invalid keyword combinations.
- Easier extension without signature sprawl.
- Better printed summaries and reproducibility records.
- Cleaner documentation because each concept has one home.

## 3. Make planning separate from execution

Right now the high-level API both interprets user intent and executes the algorithm. That is fine for small packages, but this package already has enough branching behavior that a separate planning phase would help.

### Planning proposal

Add an explicit planning object or function, for example:

- `plan = plan_fit(f, I; ...)`
- `result = solve(plan)`

The plan should record:

- which driver will be used,
- which degree or budget strategy was selected,
- which evaluation scheme will be built,
- which defaults were inferred versus explicitly requested.

This would make the package easier to debug and easier to surface in docs, benchmarks, and notebooks.

### Why planning should be explicit

The existing `recommend_parameters` already points in this direction. The redesign should push it one step further so that planning is a first-class stage rather than a hidden preprocessing step.

## 4. Promote diagnostics to structured results

The piecewise code currently uses rejection strings and exception-type names as control signals. That is serviceable for internal recursion, but too lossy for long-term usability.

### Diagnostics proposal

Introduce structured diagnostic types, for example:

- `FitAttempt`
- `RejectionReason`
- `ConvergenceReport`
- `PartitionReport`

Each failed or rejected attempt should carry structured fields such as:

- interval,
- degree,
- mode,
- achieved error,
- target,
- iteration count,
- failure kind,
- original exception where relevant.

### Expected benefits for diagnostics

- Better error messages without string parsing.
- Easier verbose logging.
- Easier future visualization of why intervals split.
- Better support for algorithm comparison and tuning.

## 5. Fix the representation mismatch for relative-zero workflows

The current relative-zero story works mathematically only in the centered case `t_z = 0`, while nonzero shifts leak through as warnings and basis caveats.

### Representation proposal

Redesign the polynomial representation layer so that the package can explicitly represent one of:

- monomial basis polynomials,
- shifted-basis polynomials,
- piecewise approximations carrying a local coordinate system.

If you do not want a full representation overhaul yet, at least introduce a result wrapper that records the basis and exposes conversion helpers explicitly.

### Expected impact on correctness and API clarity

This is exactly the kind of edge case that becomes expensive later if users start depending on current behavior. It is better to make the representation explicit before the API hardens.

## 6. Reorganize the docs around decisions, not just components

The current documentation is informative but mostly component-oriented. A better long-term structure would be decision-oriented.

### Documentation structure proposal

Use a documentation flow like this:

1. Landing page: what problem the package solves and who it is for.
2. Quick start: one recommended path using `approxfit`.
3. Choosing a workflow:
   - single polynomial vs piecewise,
   - absolute vs relative,
   - fixed degree vs coefficient budget,
   - Horner vs Estrin.
4. Practical recipes:
   - tight absolute target,
   - relative target away from zeros,
   - budget-limited piecewise approximation,
   - source generation.
5. Expert guide:
   - scheme construction,
   - search strategies,
   - low-level drivers.
6. Internals and contributor guide.
7. API reference.

### Near-term documentation changes

- Keep the README short and product-facing.
- Make the docs home page point users to the one recommended entry point.
- Move algorithm-number references and paper mapping deeper into the expert or technical sections.
- Add a capability matrix that tells users which API they should reach for.

## 7. Turn documentation examples into validation

`docs/make.jl` currently disables doctesting and missing-doc checks. That is understandable during rapid iteration, but it should not be the steady state.

### Documentation validation proposal

Move toward:

- doctested user-facing examples,
- at least one docs build in CI,
- some level of doc coverage enforcement for stable public APIs,
- generated example outputs or smoke examples for `provide_source`.

### Smallest useful next step

Even if full doctesting is too expensive immediately, introduce one or two smoke examples that must execute. That alone will catch documentation drift early.

## 8. Split tests by subsystem and contract

The existing test file is valuable, but its size now argues for reorganization.

### Test layout proposal

Split the test suite into files such as:

- `test/core/`
- `test/schemes/`
- `test/exchange/`
- `test/piecewise/`
- `test/interface/`
- `test/docs/`

And make sure each layer has both:

- invariant tests,
- user-facing behavior tests.

### Additional high-value test types

- round-trip tests for generated evaluators,
- regression tests for failure-reporting paths,
- docs examples as smoke tests,
- type-variation tests for `Float32`, `Float64`, and `BigFloat`,
- specific tests for basis handling in relative-zero mode.

## 9. Consider a smaller stable core export set

If you do not want a large structural refactor yet, the highest-leverage intermediate move is to define a smaller stable export set and treat the rest as advanced names.

### Suggested stable surface

- `approxfit`
- `fit_abs`
- `fit_rel`
- `recommend_parameters`
- `Approximation`
- `error_bound`
- `coeff_count`
- `is_piecewise`
- `pieces`
- scheme builders
- source generation helpers

Everything else can remain accessible, but not necessarily exported as equal-first-class names.

This would immediately make the package feel more intentional without deleting any capability.

## Suggested Migration Plan

## Phase 1: Clarify the package surface

- Define and document the intended stable user surface.
- Mark expert and internal APIs clearly in docs.
- Shorten the README to one recommended workflow plus links.
- Split tests into subsystem files.
- Add this redesign note to the contributor docs.

## Phase 2: Improve configuration and diagnostics

- Introduce typed configuration structs.
- Introduce structured rejection and convergence reports.
- Keep the keyword-driven path and the typed planning path aligned.
- Add docs-backed smoke tests.

## Phase 3: Harden mathematical and architectural boundaries

- Rework relative-zero result representation.
- Decide whether internal layers deserve explicit submodules.
- Narrow root exports.
- Promote documentation checks in CI.

## Recommended Priority Order

If only a few changes are feasible soon, the best order is:

1. clarify the stable user API,
2. simplify the configuration model,
3. improve diagnostics,
4. reorganize docs around user decisions,
5. reorganize tests,
6. only then consider deeper internal modularization.

## Bottom Line

The first-pass implementation is already technically credible. The main redesign need is not a new numerical core; it is a clearer contract between the package and its users.

In practice, that means:

- one obvious recommended workflow,
- a narrower stable public surface,
- structured configuration and diagnostics,
- docs that separate user tasks from research internals,
- tests and docs that validate the public story continuously.

That path keeps the current implementation value while making the package much easier to maintain and adopt.
