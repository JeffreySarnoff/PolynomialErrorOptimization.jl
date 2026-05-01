# PolynomialErrorOptimization.jl

`PolynomialErrorOptimization.jl` fits polynomials against a combined objective:

- approximation error,
- modeled finite-precision evaluation error.

The recommended entry point is:

```julia
using PolynomialErrorOptimization

approx = approxfit(sin, (-2.0, 2.0); target = 1e-8)

@show error_bound(approx)
@show coeff_count(approx)
@show approx(0.25)
```

## API layers

- Stable workflow layer: `approxfit`, `fit_abs`, `fit_rel`, `recommend_parameters`, `Approximation`, `error_bound`, `coeff_count`, `is_piecewise`, `pieces`, and the built-in scheme builders.
- Expert layer: fixed-degree drivers, piecewise drivers, search strategies, and standalone evaluator generation.
- Internal or research layer: exchange substeps, symbolic error-expression nodes, and low-level row/basis machinery.

If you are starting fresh, stay in the stable workflow layer until you need explicit control over degree policy, search, or evaluation-scheme construction.

## Installation

This package is not registered yet. From Julia:

```julia
using Pkg
Pkg.develop(path = "/path/to/PolynomialErrorOptimization")
Pkg.instantiate()
```

## Documentation map

- `docs/src/index.md`: landing page and package overview.
- `docs/src/high-level-interface.md`: recommended workflow.
- `docs/src/choosing-a-workflow.md`: decision guide for single vs piecewise, absolute vs relative, and degree vs budget.
- `docs/src/examples.md`: practical recipes.
- `docs/src/technical-guide.md`: internals and extension points.
- `docs/src/contributor-guide.md`: contributor workflow and redesign roadmap.
- `docs/src/api.md`: API split by stability layer.

## Validation

Run tests from the package root with:

```julia
using Pkg
Pkg.test()
```

Build the docs locally with:

```julia
julia --project=docs docs/make.jl
```

## Citation

If you use the package, cite the underlying paper by Arzelier, Bréhard, Hubrecht, and Joldeș:

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

This implementation is provided as-is for research/educational purposes; users should consult the original paper and the upstream Sollya-based reference implementation at <https://gitlab.laas.fr/mmjoldes/xatom> for production use.
