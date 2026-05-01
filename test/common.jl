using Test
using PolynomialErrorOptimization
import PolynomialErrorOptimization: Index, Signature, EvalScheme, OptimResult,
    AbstractMode, AbsoluteMode, RelativeMode, RelativeZeroMode,
    SearchStrategy, GridSearch, GridThenLocal, GridThenOptim,
    eval_approx_optimize, eval_approx_optimize_relative,
    eval_approx_optimize_relative_zero,
    ResultBasis, basis_info, solution_coefficients,
    approximate, approximate_abs, approximate_rel,
    approximate_abs_budget, approximate_rel_budget,
    default_scheme_builder, PiecewisePolyApprox, ApproxPiece, FitAttemptReport,
    ExchangeFailure, ConvergenceFailure,
    init_points, solve_primal, find_new_index, exchange,
    ErrExpr, VarT, VarA, Const, Neg, Add, Mul, FMA, Round,
    monomial_dot, dot_view, signum_int8, α, α!, c,
    horner_expr, fma_horner_expr, estrin_expr, fma_estrin_expr,
    lin_eval_error, build_eval_scheme, collect_rounding_us
using LinearAlgebra
using Polynomials

# ---------------------------------------------------------------------------
# IMPORTANT: do NOT alias the module via `const PEO = PolynomialErrorOptimization`.
# In some Julia 1.14 contexts (notably VS Code's testset evaluator), a
# qualified call through such an alias triggers a fresh package-import
# resolution and fails with "Package PolynomialErrorOptimization not found
# in current path". Importing specific functions (as above) is fine.
# ---------------------------------------------------------------------------

function sampled_total_error(a, f, scheme, I, M)
    tl, tr = I
    best = 0.0
    for k in 0:(M-1)
        t = tl + (tr - tl) * (k / (M - 1))
        v = abs(monomial_dot(a, t) - f(t))
        for i in 1:scheme.k
            v += abs(dot_view(a, scheme.π[i](t)))
        end
        best = max(best, v)
    end
    return best
end

function check_partition(pa, I; mode_target=nothing)
    @test length(pa.pieces) ≥ 1
    @test pa.pieces[1].a == Float64(I[1])
    @test pa.pieces[end].b == Float64(I[2])
    for i in 1:length(pa.pieces)-1
        @test pa.pieces[i].b == pa.pieces[i+1].a
        @test pa.pieces[i].a < pa.pieces[i].b
    end
    if mode_target !== nothing
        for p in pa.pieces
            @test p.result.total_error ≤ mode_target
        end
        @test pa.worst_error ≤ mode_target
    end
end