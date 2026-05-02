# Parameter Selection

This guide gives practical advice for selecting the main controls used by the
fixed-degree and piecewise APIs.

## 0. Quick reference values

Use this as a first pass before fine tuning.

| Parameter | Fast exploration | Balanced production | Aggressive accuracy |
| --- | --- | --- | --- |
| `target` | `1e-6` to `1e-8` | `1e-8` to `1e-10` | `1e-10` to `1e-12` |
| `τ` (convergence tolerance) | `1e-3` | `1e-3` to `5e-4` | `5e-4` to `1e-4` |
| `max_depth` | `20` to `26` | `24` to `30` | `28` to `34` |
| `min_width` | `0.0` | `1e-5` to `1e-4` | `1e-6` to `1e-5` |
| `driver_max_iter` | `80` to `120` | `100` to `180` | `180` to `300` |
| `strategy` | `GridSearch(2048..4096)` | `GridThenLocal` or `GridThenOptim` | `GridThenOptim(6000+)` |

These are starting ranges, not hard rules.

## 1. `target` (piecewise acceptance threshold)

`target` is the per-piece acceptance criterion on the verified bound
`total_error`.

- Smaller `target` means tighter approximation but more work (more bisections,
  more driver calls, and often more coefficients).
- Larger `target` means cheaper models and fewer pieces, but weaker accuracy.

Starting points:

- exploratory runs: `1e-6` to `1e-8`
- high-accuracy runs: `1e-9` to `1e-12`

Recommended workflow:

1. Start with a moderate `target` (for example `1e-8`).
2. Check piece count and `worst_error`.
3. Tighten by one order of magnitude only if needed.

How to derive a first value from a user-level tolerance:

1. Choose a characteristic Scale `S` for your function on Interval `I`
   (the approximation domain).
2. If you have a desired relative tolerance `eps_rel`, use
   `target ≈ eps_rel * S` as a first absolute target, where `S` is the
   characteristic magnitude of `f` on `I`.
3. Re-run and adjust after observing `worst_error` and piece count.

Good choices for Scale `S`:

- `maximum(abs.(f.(sample_grid)))` for absolute control.
- median or high quantile of `abs(f)` when outliers dominate.

Important interaction:

- If `target` is very small while `M` in `GridSearch(M)` is too small, the
  algorithm may chase sampling artifacts. Increase strategy resolution first.

## 2. `max_depth` and `min_width` (geometric constraints)

These two parameters cap how far bisection can go.

- `max_depth`: maximum recursion/bisection depth.
- `min_width`: minimum permitted interval width.

Use both together:

- If you expect local difficulty (for example near steep behavior), increase
  `max_depth` first.
- Use `min_width` to prevent over-fragmentation in tiny intervals.

Practical defaults:

- `max_depth = 24` to `32` for most smooth functions.
- `min_width = 0.0` initially, then add a floor such as `1e-5` to `1e-3`
  when piece splitting becomes too aggressive.

Depth-width consistency formulas:

- Let interval length be `L = I[2] - I[1]`.
- Smallest representable piece by depth alone is approximately `L / 2^max_depth`.
- If you need width floor `w`, choose `max_depth` so that `L / 2^max_depth <= w`.
- Equivalent rule: `max_depth >= ceil(log2(L / w))`.

Practical implication:

- If failures cite `min_width` before reaching target, lower `min_width` or
  raise `n` / `max_coeffs`.
- If failures cite `max_depth` often, increase `max_depth` by 2 to 4 first.

## 3. `total_coeffs` (global complexity cap)

`total_coeffs` caps the sum of coefficients across accepted pieces.

- `0` disables the cap.
- Positive values enforce hard model-size budgets.

How to pick it:

- If you have a memory/latency budget, set `total_coeffs` directly from that
  budget.
- Otherwise, run once with `total_coeffs = 0`, observe final usage, then set a
  cap slightly above the observed value.

Planning formulas:

- Fixed-degree piecewise (`approximate_abs`, `approximate_rel`):
  each accepted piece costs exactly `n + 1` coefficients.
  Estimated max pieces under cap is `floor(total_coeffs / (n + 1))`.
- Budget mode (`max_coeffs`): each piece costs between `1` and `max_coeffs`.
  Use pilot statistics (mean piece cost) to set realistic caps.

Sizing policy that usually works:

1. Run uncapped and record `C0 = sum(length(coeffs(piece.poly)))`.
2. Set a soft production cap around `1.05*C0` to `1.20*C0`.
3. Tighten only if deployment constraints require it.

## 4. `driver_max_iter` (inner exchange budget)

This controls the maximum iteration count of each inner fixed-degree solve.

- Too small: early failures and unnecessary bisections.
- Too large: wasted time on hard pieces.

Guidance:

- Start with `100`.
- Increase to `150-300` for tighter targets or harder intervals.
- If many pieces fail due to convergence limits, increase this before relaxing
  `target`.

Pilot-based setting:

1. Run with a generous budget (for example `300`) on a representative interval.
2. Record the iteration distribution from successful pieces.
3. Set `driver_max_iter` to about the 95th percentile plus a safety margin of
   10 to 20 iterations.

Interpretation signal:

- Frequent hits at `driver_max_iter` indicate either too strict `target`, too
  coarse `strategy`, or insufficient polynomial capacity (`n`/`max_coeffs`).

## 5. `strategy` details (maximum-search controls)

`strategy` controls how new worst-case indices are found.  In each strategy,
`M` is the Mesh point count — the number of equally spaced evaluation points
in the initial grid scan.

- `GridSearch(M)`: most robust baseline; returns the grid point with the
  highest error objective value.
- `GridThenLocal(M; bracket)`: grid scan followed by golden-section local
  refinement within `±bracket` cells of the best grid point.
- `GridThenOptim(M; bracket)`: grid scan followed by bounded Brent
  refinement (`Optim`) within `±bracket` cells; sharpest localization.

When to use each:

- Prefer `GridSearch` for strict reproducibility and low overhead.
- Prefer `GridThenLocal` when objective is mostly smooth and you want modest
  refinement without external optimizer behavior.
- Prefer `GridThenOptim` when objective maxima are sharp and exchange quality
  strongly affects runtime.

How to tune:

1. Choose a baseline `M` by interval width and function roughness.
2. Increase `M` first (for example `2048 -> 4096 -> 8192`).
3. Only then tune `bracket` (`3` or `4` is typically enough).

Useful baseline from package default:

- `default_strategy(scheme) = GridSearch(max(2048, 64 * (scheme.n + 2)))`.

Resolution heuristic for Mesh size `M`:

- If hard features occur on scale `ell` (smallest width you need to resolve),
  set `M` so grid spacing `h = (I[2]-I[1])/(M-1)` satisfies `h <= ell/4`.

## 6. Other important choices

- `τ` (tau, convergence tolerance): the relative threshold at which the
  exchange loop declares the solution converged.  Iteration stops when the
  ratio of the current error-bound improvement to the current error level
  falls below `τ`.
  - Start at `1e-3`; tighten to `1e-4` if the certified bound needs to
    be closer to the true optimum.
  - Tightening `τ` improves certification tightness but increases runtime,
    sometimes substantially near the Chebyshev optimum.
- `n` (fixed degree):
  - Lower `n` gives cheaper pieces but may force more bisection.
  - Higher `n` can reduce piece count but increase per-piece solve cost.
- `max_coeffs` (budget APIs):
  - Controls per-piece degree cap (`max_coeffs - 1`).
  - Increase when many pieces fail target at current per-piece cap.
- `degree_policy`:
  - `:max` for simpler behavior and fewer decisions.
  - `:min` for locally sparse pieces.
  - `:min_cost` for best global coefficient-efficiency when runtime is
    acceptable.
- `mode`:
  - `:abs` is the default robust choice.
  - `:rel` is useful when scale invariance matters and the function does not
    vanish on the interval.

Deeper guidance for `n` and `max_coeffs`:

- If piece count is very high but each piece converges easily, increase
  `n` or `max_coeffs`.
- If per-piece solves become expensive or unstable, reduce `n` and let
  bisection absorb complexity.
- In budget mode, start with `degree_policy = :min` for conservative cost,
  then move to `:min_cost` when total coefficient efficiency is critical.

## 7. A robust tuning sequence

1. Start with `mode = :abs`, moderate `target`, and `GridSearch`.
2. Tune `M` in `strategy` before tuning other knobs.
3. Set `max_depth` high enough that geometry is not the first limiter.
4. Add `total_coeffs` only after measuring unconstrained behavior.
5. Switch to budget mode (`max_coeffs`, `degree_policy`) when you need strict
   deploy-time complexity control.

## 8. Regime-specific defaults

These defaults are good first guesses by function behavior.

### 8.1 Smooth and slowly varying

- `target = 1e-8`
- `τ = 1e-3`
- `n = 4` to `6`
- `max_depth = 24`
- `strategy = GridSearch(4096)`

### 8.2 Oscillatory but smooth

- `target = 1e-8` to `1e-10`
- `τ = 1e-3`
- increase `M` aggressively (`6000+` often helpful)
- `strategy = GridThenOptim(M; bracket=3 or 4)`
- `max_depth = 26` to `30`

### 8.3 Endpoint-sensitive or steep-gradient regions

- begin on interior interval if mathematically acceptable
- `target = 1e-7` to `1e-9`
- `max_depth = 28` to `34`
- `min_width = 1e-6` to `1e-4`
- prefer piecewise budget mode with `degree_policy = :min_cost`

## 9. Troubleshooting by symptom

- Too many tiny pieces:
  increase `n` or `max_coeffs`, relax `target`, raise `min_width`.
- Frequent convergence-limit failures:
  raise `driver_max_iter`, improve `strategy` resolution, then revisit `target`.
- Good accuracy but too many coefficients:
  enable `total_coeffs`, switch to budget mode, try `degree_policy = :min`.
- Runtime too high:
  start by reducing `M` moderately, then relax `target` one decade.
