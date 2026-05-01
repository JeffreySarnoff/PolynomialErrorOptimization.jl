# PolynomialErrorOptimization.jl

A Julia implementation of the exchange algorithm of

> Arzelier, Bréhard, Hubrecht, Joldeș (2024/2025),
> *"An Exchange Algorithm for Optimizing both Approximation and Finite-Precision Evaluation Errors in Polynomial Approximations"*,
> HAL-04709615, ACM Transactions on Mathematical Software.

The package computes a degree-`n` polynomial that minimises the supremum (over a real interval `I`) of the **sum** of approximation error and a linearised model of finite-precision evaluation error:

```
min_{a ∈ ℝⁿ⁺¹}  max_{t ∈ I}  ( |f(t) − p(t)|  +  θ(a, t) )
```

where `p(t) = Σⱼ aⱼ tʲ` and `θ(a, t) = Σᵢ |πᵢ(t)ᵀ a|` is the linearised evaluation-error bound determined by the user-chosen polynomial evaluation scheme (e.g. Horner, Estrin) and floating-point precision.

## Testing status

The package includes a test suite in `test/runtests.jl`. Run it from the
package root with:

```julia
using Pkg
Pkg.test()
```

## Installation

This is not a registered package. Clone and `dev` it:

```julia
using Pkg
Pkg.develop(path = "/path/to/PolynomialErrorOptimization")
Pkg.instantiate()
```

Runtime dependencies are listed in `Project.toml`: `ArgCheck`, `LinearSolve`,
`Optim`, `Polynomials`, plus the `LinearAlgebra` and `Printf` standard
libraries. No LP solver is required — every sub-step of the exchange
algorithm is a square `(n+2)×(n+2)` linear solve.

## Quick example

```julia
using PolynomialErrorOptimization

approx = approxfit(sin, (-2.0, 2.0); target = 1e-8)

@show error_bound(approx)
@show coeff_count(approx)
@show approx(0.25)
```

For full control, use the lower-level drivers directly:

```julia
using PolynomialErrorOptimization

f      = sin
n      = 6
I      = (-2.0, 2.0)
scheme = horner_scheme(n; u = 2.0^-12)        # toy precision (paper §6.1)

res = eval_approx_optimize(f, n, I, scheme;
                           τ      = 0.01,
                           verbose = true)

@show res.poly                # coefficient type is Float64 by default
@show res.total_error         # verified upper bound on max |f-p| + θ(a,·)
@show res.iterations
```

`OptimResult.total_error` is `astar`, the verified upper bound — by Theorem 2 of the paper, it satisfies

```
ε⋆ ≤ res.total_error ≤ (1 + τ) · ε⋆
```

where `ε⋆` is the optimum of the (P_general) problem.

## Target and computation types

The package separates the type of the returned polynomial coefficients from
the type used internally by the exchange algorithm:

- `target_type` controls returned coefficients and the default rounding unit
  used by high-level scheme builders.
- `compute_type` controls internal nodes, dense solves, search objectives, and
  verified error bounds.

For example, to model Float32 evaluation while doing the optimization in
BigFloat and returning Float32 coefficients:

```julia
approx = approxfit(sin, (-1.0, 1.0);
    target = 1e-6,
    degree = 5,
    piecewise = false,
    target_type = Float32,
    compute_type = BigFloat)
```

At the expert-driver level, `compute_type` is the `EvalScheme` element type.
Use `target_type` to choose the result coefficient type:

```julia
scheme = horner_scheme(5, BigFloat; u = eps(Float32) / 2)
res = eval_approx_optimize(sin, 5, (-1, 1), scheme;
    target_type = Float32)
```

Generated standalone evaluators use the coefficient type for arithmetic by
default. Pass `eval_type` to `provide_source` or `provide` to evaluate the
stored coefficients in another floating-point type.

## Public API

### Top-level drivers

| Function | Paper | Purpose |
| --- | --- | --- |
| `eval_approx_optimize` | Algorithm 3 | Absolute-error mode (P_general). |
| `eval_approx_optimize_relative` | Section 5, eq. (15) | Relative error when `f` does not vanish on `I`. |
| `eval_approx_optimize_relative_zero` | Section 5, eq. (16) | Relative error when `f` has a known zero of finite order. **Only `t_z = 0` is currently supported.** |

### Building blocks (paper Algorithms 4–7)

```julia
ω, y    = init_points(scheme, I)              # Algorithm 4
ā, a    = solve_primal(f, scheme, ω)          # Algorithm 5
ωstar, astar  = find_new_index(f, scheme, I, a)     # Algorithm 6
ω, y    = exchange(scheme, ω, y, ωstar)          # Algorithm 7
```

### Evaluation schemes

```julia
scheme = horner_scheme(n; u = 2.0^-53)        # Horner, single FP precision
scheme = fma_horner_scheme(n; u = 2.0^-53)    # Horner with rounded FMA steps
scheme = estrin_scheme(n; u = 2.0^-53)        # Estrin, single FP precision
scheme = fma_estrin_scheme(n; u = 2.0^-53)    # Estrin with FMA combines
```

The high-level interface also accepts `scheme = :horner_fma` and
`scheme = :estrin_fma`:

```julia
approx_fma = approxfit(sin, (-1.0, 1.0);
    target = 1e-8,
    degree = 5,
    piecewise = false,
    scheme = :horner_fma)
```

For standalone source generation, pass `eval_op = :fma` to emit explicit
`fma(...)` affine combines, and pass `eval_scheme = :estrin` to emit
Estrin-shaped evaluation instead of Horner-shaped evaluation. High-level
`Approximation` values created with `scheme = :horner_fma` or
`scheme = :estrin_fma` preserve those choices automatically unless overridden.

For mixed-precision or non-standard schemes, build a symbolic expression tree and compile it:

```julia
using PolynomialErrorOptimization
# Hand-written expression tree:  RN(a₀ + RN(a₁·t, u₂), u₁)
e = Round(Add(VarA(0),
              Round(Mul(VarA(1), VarT()), 2.0^-24, 1)),  # mul rounded at u₂
          2.0^-53, 2)                                    # add rounded at u₁
θ        = lin_eval_error(e)                  # Algorithm 2: symbolic
ids_to_u = collect_rounding_us(e)
scheme   = build_eval_scheme(θ, ids_to_u, 1, "mixed-precision-degree-1")
```

### Result type

```julia
struct OptimResult{TargetT,ComputeT}
    poly                          # Polynomial{TargetT,:t}
    total_error::ComputeT         # astar = verified upper bound
    discrete_error::ComputeT      # ā = LP optimum at termination
    iterations::Int
    discretization::Vector{Index{ComputeT}}
    dual::Vector{ComputeT}
    converged::Bool
end
```

### Search strategies for `find_new_index`

```julia
GridSearch(M)                             # equispaced grid of M points
GridThenLocal(M; bracket = 3)             # grid + golden-section refinement
GridThenOptim(M; bracket = 3)             # grid + Optim.Brent() refinement
```

Default: `GridSearch(max(2048, 64*(n+2)))`.

## Algorithm map

```
                          ┌────────────────────────────────────┐
                          │   eval_approx_optimize  (Alg 3)   │
                          └─────────────┬──────────────────────┘
                                        │
                  ┌─────────────────────┼─────────────────────────┐
                  │                     │                         │
         init_points (Alg 4)   solve_primal (Alg 5)    find_new_index (Alg 6)
                  │                     │                         │
                  │                A · x = c(ω)         max_t obj(t)  +  σ from sign
                  │                                              │
                  └──── ω⁽⁰⁾, y⁽⁰⁾ ──┐         ┌──── (ωstar, astar) ┘
                                       ▼         ▼
                                exchange (Alg 7)
                                       │
                                       └──── (ω⁽ᵗ⁺¹⁾, y⁽ᵗ⁺¹⁾)

  Algorithm 2 (lin_eval_error)  ────►  builds  EvalScheme.π[i](t)
                                        used inside α(ω, scheme)
```

## Project layout

```
PolynomialErrorOptimization/
├── Project.toml
├── README.md
└── src/
    ├── PolynomialErrorOptimization.jl   # public module
    ├── core/                            # domain types, numeric kernels, constraints
    ├── schemes/                         # symbolic error machinery, Horner, Estrin
    ├── exchange/                        # Algorithms 3-7 and search strategies
    ├── piecewise/                       # adaptive piecewise approximation
    └── provide.jl                       # standalone evaluator generation
└── test/
    └── runtests.jl                      # unit tests for each algorithm
```

## Limitations

- **`RelativeZeroMode` only fully supported for `t_z = 0`.** For `t_z ≠ 0`, the polynomial returned is in the shifted basis `(t − t_z)ʲ` and the user must convert. The package emits an `@warn` in this case.
- **Higher-order error terms are not modelled.** `lin_eval_error` returns only the linear-in-`uᵢ` part of the rounding-error bound. The paper's Example C in Table 1 (mixed precision with `O(u²)` cross-terms) requires a future extension.
- **No stochastic error model.** The implementation uses the worst-case linearised model; switching to the Higham–Mary probabilistic model (paper ref. [31]) would be a separate pass through `eval_error.jl`.
- **Assumption 1 of the paper** (the dual is in the interior of the feasible set at every iteration) is not theoretically guaranteed for arbitrary evaluation schemes. The paper notes it never failed in practice; the implementation throws `ExchangeFailure` if it ever does.

## Citation

If you use this package, please cite the underlying paper:

```bibtex
@article{ArzelierBrehardHubrechtJoldes2025,
  author  = {Arzelier, Denis and Br{\'e}hard, Florent and Hubrecht, Tom and Jolde\c{s}, Mioara},
  title   = {An Exchange Algorithm for Optimizing both Approximation and
             Finite-Precision Evaluation Errors in Polynomial Approximations},
  journal = {ACM Trans. Math. Softw.},
  year    = {2025},
  doi     = {10.1145/3770066}
}
```

## License

This implementation is provided as-is for research/educational purposes; users should consult the original paper and the upstream Sollya-based reference implementation at <https://gitlab.laas.fr/mmjoldes/xatom> for production use.
