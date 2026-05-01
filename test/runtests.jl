using Test
using PolynomialErrorOptimization
import PolynomialErrorOptimization: monomial_dot, dot_view, signum_int8, α, α!, c,
    horner_expr, lin_eval_error, build_eval_scheme
using LinearAlgebra
using Polynomials

# ---------------------------------------------------------------------------
# IMPORTANT: do NOT alias the module via `const PEO = PolynomialErrorOptimization`.
# In some Julia 1.14 contexts (notably VS Code's testset evaluator), a
# qualified call through such an alias triggers a fresh package-import
# resolution and fails with "Package PolynomialErrorOptimization not found
# in current path". Importing specific functions (as above) is fine.
# ---------------------------------------------------------------------------

@testset "PolynomialErrorOptimization" begin

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

    @testset "dispatch hygiene and inference smoke tests" begin
        @test isempty(Test.detect_ambiguities(PolynomialErrorOptimization;
            recursive=false))
        @test @inferred(monomial_dot([1.0, 2.0, 3.0], 0.5)) ≈ 2.75
        @test @inferred(dot_view([1.0, 2.0], [3.0, 4.0])) ≈ 11.0
        @test @inferred(signum_int8(-1.0)) === Int8(-1)
        @test @inferred(horner_scheme(2, Float32)) isa EvalScheme{Float32}
        @test @inferred(fma_horner_scheme(2, Float32)) isa EvalScheme{Float32}
        @test @inferred(fma_estrin_scheme(2, Float32)) isa EvalScheme{Float32}
    end

    # ===========================================================================
    @testset "utils: monomial_dot, dot_view, signum_int8" begin
        @test monomial_dot(Float64[], 0.5) == 0.0
        @test monomial_dot([1.0], 100.0) == 1.0
        @test monomial_dot([1.0, 2.0, 3.0], 0.0) == 1.0
        @test monomial_dot([1.0, 2.0, 3.0], 1.0) ≈ 6.0
        @test monomial_dot([1.0, 2.0, 3.0], 2.0) ≈ 1 + 4 + 12

        @test dot_view([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) ≈ 32.0

        @test signum_int8(0.0) === Int8(0)
        @test signum_int8(1.5) === Int8(1)
        @test signum_int8(-1e-300) === Int8(-1)
    end

    # ===========================================================================
    @testset "horner_scheme: shape and special cases" begin
        # n = 0: single zero row
        s0 = horner_scheme(0; u=2.0^-53)
        @test s0.n == 0
        @test s0.k == 1
        @test s0.π[1](0.5) == [0.0]

        # n = 1: rows are (u, ut) and (0, ut)
        u = 2.0^-12
        s1 = horner_scheme(1; u=u)
        @test s1.n == 1
        @test s1.k == 2
        @test s1.π[1](0.7) ≈ [u, u * 0.7]
        @test s1.π[2](0.7) ≈ [0.0, u * 0.7]

        # n = 3: cross-check Example 1 of the paper
        u = 2.0^-12
        s3 = horner_scheme(3; u=u)
        @test s3.n == 3 && s3.k == 4
        t = 0.4
        # π₁ = (u, ut, ut², ut³)
        @test s3.π[1](t) ≈ [u, u * t, u * t^2, u * t^3]
        # π₂ = (0, 2ut, 2ut², 2ut³)
        @test s3.π[2](t) ≈ [0.0, 2u * t, 2u * t^2, 2u * t^3]
        # π₃ = (0, 0, 2ut², 2ut³)
        @test s3.π[3](t) ≈ [0.0, 0.0, 2u * t^2, 2u * t^3]
        # π₄ = (0, 0, 0, ut³)
        @test s3.π[4](t) ≈ [0.0, 0.0, 0.0, u * t^3]
    end

    # ===========================================================================
    @testset "fma_horner_scheme: shape and symbolic match" begin
        s0 = fma_horner_scheme(0; u=2.0^-53)
        @test s0.n == 0
        @test s0.k == 1
        @test s0.π[1](0.5) == [0.0]

        u = 2.0^-12
        s1 = fma_horner_scheme(1; u=u)
        @test s1.n == 1
        @test s1.k == 1
        @test s1.π[1](0.7) ≈ [u, u * 0.7]

        s3 = fma_horner_scheme(3; u=u)
        t = 0.4
        @test s3.n == 3
        @test s3.k == 3
        @test s3.π[1](t) ≈ [u, u * t, u * t^2, u * t^3]
        @test s3.π[2](t) ≈ [0.0, u * t, u * t^2, u * t^3]
        @test s3.π[3](t) ≈ [0.0, 0.0, u * t^2, u * t^3]
        @test s3.k < horner_scheme(3; u=u).k

        e, ids_to_u = fma_horner_expr(4; u=u)
        θ = lin_eval_error(e)
        sym_scheme = build_eval_scheme(θ, ids_to_u, 4, "fma-horner-symbolic")
        closed_scheme = fma_horner_scheme(4; u=u)
        for t in (-0.9, -0.3, 0.1, 0.7, 0.99)
            a = randn(5)
            sym_θ = sum(abs(dot_view(a, sym_scheme.π[i](t)))
                        for i in 1:sym_scheme.k)
            closed_θ = sum(abs(dot_view(a, closed_scheme.π[i](t)))
                           for i in 1:closed_scheme.k)
            @test sym_θ ≈ closed_θ rtol = 1e-12 atol = 1e-14
        end
    end

    # ===========================================================================
    @testset "fma_estrin_scheme: shape and symbolic match" begin
        s0 = fma_estrin_scheme(0; u=2.0^-53)
        @test s0.n == 0
        @test s0.k == 0

        u = 2.0^-12
        s1 = fma_estrin_scheme(1; u=u)
        @test s1.n == 1
        @test s1.k == 1
        @test s1.π[1](0.7) ≈ [u, u * 0.7]
        @test s1.k < estrin_scheme(1; u=u).k

        s4 = fma_estrin_scheme(4; u=u)
        @test s4.n == 4
        @test s4.k < estrin_scheme(4; u=u).k

        e, ids_to_u = fma_estrin_expr(5; u=u)
        θ = lin_eval_error(e)
        sym_scheme = build_eval_scheme(θ, ids_to_u, 5, "fma-estrin-symbolic")
        expr_scheme = fma_estrin_scheme(5; u=u)
        for t in (-0.9, -0.3, 0.1, 0.7, 0.99)
            a = randn(6)
            sym_θ = sum(abs(dot_view(a, sym_scheme.π[i](t)))
                        for i in 1:sym_scheme.k)
            expr_θ = sum(abs(dot_view(a, expr_scheme.π[i](t)))
                         for i in 1:expr_scheme.k)
            @test sym_θ ≈ expr_θ rtol = 1e-12 atol = 1e-14
        end
    end

    # ===========================================================================
    @testset "α / α! / c: row construction" begin
        n = 3
        scheme = horner_scheme(n; u=2.0^-12)
        σ = Int8[1, 0, 0, 0, 0]   # σ₀ = +1, all error rows masked
        ω = Index(0.5, σ)

        row = α(ω, scheme)
        @test length(row) == n + 2
        @test row[1] == 1.0          # error variable column
        # Without error rows, row[2..n+2] = (1, t, t², t³)
        @test row[2:end] ≈ [1.0, 0.5, 0.25, 0.125]

        # With σ₀ = -1
        σm = Int8[-1, 0, 0, 0, 0]
        rowm = α(Index(0.5, σm), scheme)
        @test rowm[2:end] ≈ -[1.0, 0.5, 0.25, 0.125]

        # In-place version produces same result
        out = zeros(n + 2)
        α!(out, ω, scheme)
        @test out ≈ row

        # c uses σ₀
        @test c(ω, t -> 7.0) ≈ 7.0
        @test c(Index(0.5, σm), t -> 7.0) ≈ -7.0
    end

    # ===========================================================================
    @testset "init_points: Lemma 1 properties" begin
        for n in 0:6
            scheme = horner_scheme(n; u=2.0^-12)
            I = (-1.0, 1.0)
            ω, y = init_points(scheme, I)
            @test length(ω) == n + 2
            @test length(y) == n + 2
            # Lemma 1: yⱼ > 0 for all j
            @test all(y .> 0)

            # Lemma 1 also: A · y = e₁ where A[:,j] = α(ωⱼ).
            m = n + 2
            A = Matrix{Float64}(undef, m, m)
            for j in 1:m
                A[:, j] = α(ω[j], scheme)
            end
            @test A * y ≈ [1.0; zeros(m - 1)] atol = 1e-10
            # Chebyshev nodes lie in I
            for ωⱼ in ω
                @test -1 - 1e-12 ≤ ωⱼ.t ≤ 1 + 1e-12
            end
        end
    end

    # ===========================================================================
    @testset "solve_primal: Lemma 2 — recovers Remez when θ ≡ 0" begin
        # Using a degenerate scheme (all π's zero), solve_primal on the
        # alternating-Chebyshev discretisation should reproduce the classical
        # discrete Remez interpolation for f(t) on those nodes.
        n = 4
        f = sin
        π_zero = [(t::Float64) -> zeros(Float64, n + 1) for _ in 1:1]
        scheme_zero = EvalScheme(n, 1, π_zero, "zero-error")
        I = (-1.0, 1.0)

        ω, _ = init_points(scheme_zero, I)
        ā, a = solve_primal(f, scheme_zero, ω)

        # By construction, |a^T π₀(tⱼ) − f(tⱼ)| = ā for all j (with σ₀ alternating).
        for ωⱼ in ω
            residual = monomial_dot(a, ωⱼ.t) - f(ωⱼ.t)
            @test abs(abs(residual) - ā) < 1e-10
        end
        @test ā > 0
    end

    # ===========================================================================
    @testset "find_new_index: maximum and signature reconstruction" begin
        n = 3
        scheme = horner_scheme(n; u=2.0^-30)
        I = (-1.0, 1.0)
        f = exp
        a = Float64[1.0, 1.0, 0.5, 1.0/6.0]   # Taylor coefficients of exp at 0

        # Use a fine grid so the brute-force sampled max and find_new_index's
        # internal max use the same resolution.
        M = 50_001
        ωstar, astar = find_new_index(f, scheme, I, a; strategy=GridSearch(M))

        # Sampled-objective brute-force max at the same resolution.
        sampled = sampled_total_error(a, f, scheme, I, M)
        @test astar ≥ sampled - 1e-10
        @test astar ≤ sampled + 1e-10   # same grid → values must match

        # Signature consistency: σ₀ = -sign(aᵀπ₀(tstar) - f(tstar))
        v0 = monomial_dot(a, ωstar.t) - f(ωstar.t)
        @test ωstar.σ[1] == -signum_int8(v0)

        # Grid + Optim local refinement should not underperform the sampled
        # coarse-grid objective used above.
        ωopt, aopt = find_new_index(f, scheme, I, a;
            strategy=GridThenOptim(10_001; bracket=4))
        @test aopt ≥ sampled - 1e-10
        @test I[1] ≤ ωopt.t ≤ I[2]
        v0opt = monomial_dot(a, ωopt.t) - f(ωopt.t)
        @test ωopt.σ[1] == -signum_int8(v0opt)
    end

    # ===========================================================================
    @testset "exchange: Lemma 4 properties" begin
        n = 3
        scheme = horner_scheme(n; u=2.0^-12)
        I = (-1.0, 1.0)
        f = sin
        ω, y = init_points(scheme, I)
        ā, a = solve_primal(f, scheme, ω)
        ωstar, _ = find_new_index(f, scheme, I, a)

        ω_new, y_new = exchange(scheme, ω, y, ωstar)
        @test length(ω_new) == n + 2
        @test length(y_new) == n + 2
        # All dual entries non-negative
        @test all(y_new .≥ -1e-12)

        # Basis property: A_new is invertible
        m = n + 2
        A_new = Matrix{Float64}(undef, m, m)
        for j in 1:m
            A_new[:, j] = α(ω_new[j], scheme)
        end
        @test rank(A_new) == m
        # Dual feasibility: A_new * y_new = e₁
        @test A_new * y_new ≈ [1.0; zeros(m - 1)] atol = 1e-9
    end

    # ===========================================================================
    @testset "lin_eval_error: matches closed-form Horner (Example 1)" begin
        # Build the symbolic Horner tree, compute its linearised error, and
        # compare to horner_scheme entry-by-entry.
        n = 4
        u = 2.0^-12
        e, ids_to_u = horner_expr(n; u=u)
        θ = lin_eval_error(e)
        sym_scheme = build_eval_scheme(
            θ, ids_to_u, n, "horner-symbolic")
        closed_scheme = horner_scheme(n; u=u)

        # The two schemes need not have rows in the same order, but their
        # *combined* error envelope must agree at sampled points.
        for t in (-0.9, -0.3, 0.1, 0.7, 0.99)
            a = randn(n + 1)
            sym_θ = sum(abs(dot_view(a, sym_scheme.π[i](t)))
                        for i in 1:sym_scheme.k)
            closed_θ = sum(abs(dot_view(a, closed_scheme.π[i](t)))
                           for i in 1:closed_scheme.k)
            # The closed-form bound is paper eq. (7); the symbolic bound is the
            # exact linearised expression. The closed form is an upper bound on
            # the true linearised error, which is what `lin_eval_error` returns,
            # so closed_θ should be ≥ sym_θ.
            @test closed_θ ≥ sym_θ * (1 - 1e-9)
        end
    end

    # ===========================================================================
    @testset "eval_approx_optimize: end-to-end on sin / [-1, 1]" begin
        # Sin on [-1, 1] with degree 5 and tiny rounding unit: total-error should
        # approach the classical Remez approximation error, since evaluation
        # error becomes negligible at u = 2^-53.
        n = 5
        scheme = horner_scheme(n; u=2.0^-53)
        res = eval_approx_optimize(sin, n, (-1.0, 1.0), scheme;
            τ=1e-3, max_iter=50)
        @test res.converged
        @test res.iterations ≤ 50
        # At u = 2^-53 the bound should be very small (under 1e-3 for n=5).
        @test res.total_error < 1e-3
        # Verify the polynomial actually approximates sin
        @test abs(res.poly(0.5) - sin(0.5)) < res.total_error
        @test abs(res.poly(-0.7) - sin(-0.7)) < res.total_error
    end

    # ===========================================================================
    @testset "eval_approx_optimize: Airy-toy (paper §6.1)" begin
        # The paper uses Ai but here we use a smooth surrogate: f(t) = sin(3t)
        # on [-2, 2], n = 6, u = 2^-12. We verify the algorithm terminates and
        # produces a polynomial whose total error is finite and consistent with
        # the discrete optimum.
        #
        # Pass an explicit GridSearch(M) so the driver's internal sup search and
        # the post-hoc verification sampler walk the same grid points.
        n = 6
        scheme = horner_scheme(n; u=2.0^-12)
        f = t -> sin(3t)
        I = (-2.0, 2.0)
        M = 20_001
        res = eval_approx_optimize(f, n, I, scheme;
            τ=0.01, max_iter=100,
            strategy=GridSearch(M))
        @test res.converged
        @test res.discrete_error ≤ res.total_error
        @test res.total_error ≤ (1 + 0.01) * res.discrete_error + 1e-12

        # Same-grid verification sampler.
        a = Polynomials.coeffs(res.poly)
        @test sampled_total_error(a, f, scheme, I, M) ≤ res.total_error + 1e-10
    end

    # ===========================================================================
    @testset "runtime evaluators match Polynomials evaluation" begin
        a = [1.0, -2.0, 0.5, 0.0, 3.0]
        p = Polynomials.Polynomial(a, :t)
        for t in (-1.5, 0.0, 0.3, 1.7)
            @test horner_eval(a, t) ≈ p(t)
            @test fma_horner_eval(a, t) ≈ p(t)
            @test estrin_eval(a, t) ≈ p(t)
            @test fma_estrin_eval(a, t) ≈ p(t)
        end
    end

    # ===========================================================================
    @testset "eval_approx_optimize: FMA Horner smoke test" begin
        n = 5
        scheme = fma_horner_scheme(n; u=2.0^-53)
        res = eval_approx_optimize(sin, n, (-1.0, 1.0), scheme;
            τ=1e-3, max_iter=50)
        @test res.converged
        @test res.total_error < 1e-3
        @test abs(res.poly(0.5) - sin(0.5)) < res.total_error
    end

    # ===========================================================================
    @testset "eval_approx_optimize: FMA Estrin smoke test" begin
        n = 5
        scheme = fma_estrin_scheme(n; u=2.0^-53)
        res = eval_approx_optimize(sin, n, (-1.0, 1.0), scheme;
            τ=1e-3, max_iter=50)
        @test res.converged
        @test res.total_error < 1e-3
        @test abs(res.poly(0.5) - sin(0.5)) < res.total_error
    end

    # ===========================================================================
    @testset "eval_approx_optimize_relative: smoke test" begin
        # exp on [0.5, 1.5] never vanishes, so RelativeMode is well-defined.
        n = 4
        scheme = horner_scheme(n; u=2.0^-53)
        res = eval_approx_optimize_relative(exp, n, (0.5, 1.5), scheme;
            τ=1e-3, max_iter=50)
        @test res.converged
        # The relative error returned should be a small positive number.
        @test res.total_error > 0
        @test res.total_error < 1e-3
    end

    # ===========================================================================
    @testset "ConvergenceFailure with too-tight max_iter" begin
        n = 6
        scheme = horner_scheme(n; u=2.0^-12)
        f = t -> sin(3t)
        I = (-2.0, 2.0)
        @test_throws ConvergenceFailure eval_approx_optimize(
            f, n, I, scheme; τ=1e-12, max_iter=1)
    end

    # ===========================================================================
    # approximate.jl — adaptive piecewise drivers
    # ===========================================================================
    #
    # These tests exercise the four piecewise drivers (`approximate_abs`,
    # `approximate_rel`, `approximate_abs_budget`, `approximate_rel_budget`),
    # the unified `approximate` dispatcher, the three `degree_policy` options,
    # and the `total_coeffs` cap.
    #
    # Most assertions are structural (partition coverage, soundness of the
    # acceptance criterion, dispatch routing) rather than numeric, since the
    # numeric behaviour is the parent driver's responsibility and the inner
    # `total_error` field is already a verified upper bound.

    # Structural sanity check shared by every accepting test below.
    function _check_partition(pa, I; mode_target=nothing)
        @test length(pa.pieces) ≥ 1
        @test pa.pieces[1].a == Float64(I[1])
        @test pa.pieces[end].b == Float64(I[2])
        for i in 1:length(pa.pieces)-1
            @test pa.pieces[i].b == pa.pieces[i+1].a
            @test pa.pieces[i].a < pa.pieces[i].b
        end
        # Acceptance soundness: every piece's verified bound is ≤ target.
        # The driver's own acceptance check is exact `≤ target`, so we
        # require exact `≤` here too.
        if mode_target !== nothing
            for p in pa.pieces
                @test p.result.total_error ≤ mode_target
            end
            @test pa.worst_error ≤ mode_target
        end
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_abs: end-to-end on sin / [-3, 3]" begin
        target = 1e-6
        scheme = horner_scheme(4; u=2.0^-53)
        pa = approximate_abs(sin, 4, (-3.0, 3.0), scheme;
            target=target, τ=1e-2, max_depth=12)
        @test pa isa PiecewisePolyApprox
        @test pa.mode === :abs
        @test pa.max_n == 4
        @test pa.target == target
        _check_partition(pa, (-3.0, 3.0); mode_target=target)
        # Spot-check the callable dispatch agrees with sin to within `target`.
        for t in (-2.7, -1.0, 0.0, 0.5, 1.9, 2.999)
            @test abs(pa(t) - sin(t)) ≤ target
        end
        # Boundary points should not throw.
        @test pa(-3.0) isa Float64
        @test pa(3.0) isa Float64
        # Out-of-domain throws.
        @test_throws DomainError pa(3.5)
        @test_throws DomainError pa(-3.5)
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_abs: constants (n = 0)" begin
        # A constant fit on a small enough interval around a smooth function
        # should converge — possibly with many pieces.
        target = 1e-2
        scheme = horner_scheme(0; u=2.0^-53)
        pa = approximate_abs(cos, 0, (-1.0, 1.0), scheme;
            target=target, τ=0.1, max_depth=20)
        @test pa.max_n == 0
        _check_partition(pa, (-1.0, 1.0); mode_target=target)
        # Each piece's polynomial is a degree-0 constant.
        for p in pa.pieces
            @test length(Polynomials.coeffs(p.result.poly)) == 1
        end
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_rel: smoke test on exp / [0.5, 2.5]" begin
        # exp never vanishes on (0, ∞); RelativeMode is well-defined.
        target = 1e-6
        scheme = horner_scheme(5; u=2.0^-53)
        pa = approximate_rel(exp, 5, (0.5, 2.5), scheme;
            target=target, τ=1e-2, max_depth=10)
        @test pa.mode === :rel
        _check_partition(pa, (0.5, 2.5); mode_target=target)
        # Verified bound is on the *relative* error |f-p|+θ over |f|, so the
        # absolute error |pa(t) - exp(t)| ≤ target * exp(t).
        for t in (0.6, 1.0, 1.7, 2.4)
            @test abs(pa(t) - exp(t)) ≤ target * exp(t)
        end
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_abs_budget :max policy" begin
        target = 1e-6
        pa = approximate_abs_budget(sin, 5, (-3.0, 3.0);
            target=target,
            degree_policy=:max,
            τ=1e-2, max_depth=12)
        @test pa.max_n == 4   # max_coeffs - 1
        _check_partition(pa, (-3.0, 3.0); mode_target=target)
        # Under :max every piece is fit at the full budget degree.
        for p in pa.pieces
            @test length(Polynomials.coeffs(p.result.poly)) == 5
        end
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_abs_budget :min policy uses ≤ :max coefficients" begin
        target = 1e-4
        common = (target=target, τ=1e-2, max_depth=12)
        pa_max = approximate_abs_budget(sin, 6, (-2.0, 2.0);
            degree_policy=:max, common...)
        pa_min = approximate_abs_budget(sin, 6, (-2.0, 2.0);
            degree_policy=:min, common...)
        _check_partition(pa_max, (-2.0, 2.0); mode_target=target)
        _check_partition(pa_min, (-2.0, 2.0); mode_target=target)
        # :min may use lower per-piece degrees than :max — its per-piece coeff
        # count is ≤ max_coeffs, with the same upper bound. We can't assert
        # strict inequality without committing to a specific bisection schedule.
        for p in pa_min.pieces
            @test length(Polynomials.coeffs(p.result.poly)) ≤ 6
        end
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_abs_budget :min_cost ≤ :max total coeffs" begin
        # :min_cost is a globally-aware variant: at each piece it compares
        # accept-here vs bisect-and-recurse, picking whichever spends fewer
        # coefficients in total. So total cost(:min_cost) ≤ total cost(:max).
        target = 1e-4
        common = (target=target, τ=1e-2, max_depth=10)
        pa_max = approximate_abs_budget(sin, 6, (-2.0, 2.0);
            degree_policy=:max, common...)
        pa_minc = approximate_abs_budget(sin, 6, (-2.0, 2.0);
            degree_policy=:min_cost, common...)
        _check_partition(pa_minc, (-2.0, 2.0); mode_target=target)
        total_max = sum(length(Polynomials.coeffs(p.result.poly))
                        for p in pa_max.pieces)
        total_minc = sum(length(Polynomials.coeffs(p.result.poly))
                         for p in pa_minc.pieces)
        @test total_minc ≤ total_max
    end

    # ---------------------------------------------------------------------------
    @testset "approximate_abs_budget: :min_cost ≤ :min total coeffs" begin
        # :min_cost is designed to minimise total coefficients across the
        # partition; in particular it should never spend more total
        # coefficients than the locally-greedy :min policy.
        target = 1e-4
        common = (target=target, τ=1e-2, max_depth=10)
        pa_min = approximate_abs_budget(sin, 6, (-1.0, 1.0);
            degree_policy=:min, common...)
        pa_minc = approximate_abs_budget(sin, 6, (-1.0, 1.0);
            degree_policy=:min_cost, common...)
        total_min = sum(length(Polynomials.coeffs(p.result.poly))
                        for p in pa_min.pieces)
        total_minc = sum(length(Polynomials.coeffs(p.result.poly))
                         for p in pa_minc.pieces)
        @test total_minc ≤ total_min
    end

    # ---------------------------------------------------------------------------
    @testset "total_coeffs cap is enforced" begin
        target = 1e-6
        scheme = horner_scheme(4; u=2.0^-53)
        # Generous cap → success.
        pa = approximate_abs(sin, 4, (-3.0, 3.0), scheme;
            target=target, τ=1e-2,
            total_coeffs=1000, max_depth=12)
        total = sum(length(Polynomials.coeffs(p.result.poly)) for p in pa.pieces)
        @test total ≤ 1000
        # Tiny cap → must error. With max_n = 4, even the very first accepted
        # piece costs 5; total_coeffs = 4 cannot fit a single piece.
        @test_throws ErrorException approximate_abs(
            sin, 4, (-3.0, 3.0), scheme;
            target=target, τ=1e-2,
            total_coeffs=4, max_depth=12)
    end

    # ---------------------------------------------------------------------------
    @testset "total_coeffs cap interacts with :min_cost" begin
        target = 1e-4
        # Generous cap — should succeed.
        pa = approximate_abs_budget(sin, 6, (-2.0, 2.0);
            target=target,
            degree_policy=:min_cost,
            τ=1e-2, max_depth=10,
            total_coeffs=200)
        total = sum(length(Polynomials.coeffs(p.result.poly)) for p in pa.pieces)
        @test total ≤ 200
        _check_partition(pa, (-2.0, 2.0); mode_target=target)
        # Tiny positive cap that can't fit even one coefficient → error.
        # (total_coeffs = 0 means "no cap"; we want a concrete too-small cap.)
        @test_throws ErrorException approximate_abs_budget(
            sin, 6, (-2.0, 2.0);
            target=target, degree_policy=:min_cost,
            τ=1e-2, max_depth=4, total_coeffs=1)
    end

    # ---------------------------------------------------------------------------
    @testset "min_width / max_depth refusal paths" begin
        target = 1e-12   # impossibly tight at low max_depth
        scheme = horner_scheme(2; u=2.0^-53)
        # max_depth = 0 means we get one shot at the whole interval; a tight
        # target with low degree should fail to subdivide.
        @test_throws ErrorException approximate_abs(
            sin, 2, (-3.0, 3.0), scheme;
            target=target, τ=1e-3, max_depth=0)

        # min_width too coarse: refuse to bisect below width 5 when interval is
        # only width 6, so we get at most one bisection. Very tight target
        # at low degree → still fail.
        @test_throws ErrorException approximate_abs(
            sin, 2, (-3.0, 3.0), scheme;
            target=target, τ=1e-3,
            max_depth=30, min_width=5.0)
    end

    # ---------------------------------------------------------------------------
    @testset "unified approximate: dispatch routing" begin
        target = 1e-6
        scheme = horner_scheme(4; u=2.0^-53)

        # Fixed-degree absolute.
        pa = approximate(sin, (-3.0, 3.0);
            target=target, n=4, scheme=scheme,
            τ=1e-2, max_depth=12)
        _check_partition(pa, (-3.0, 3.0); mode_target=target)
        @test pa.mode === :abs
        @test pa.max_n == 4

        # Fixed-degree relative.
        pa_rel = approximate(exp, (0.5, 2.5);
            target=target, mode=:rel,
            n=5, scheme=horner_scheme(5; u=2.0^-53),
            τ=1e-2, max_depth=10)
        _check_partition(pa_rel, (0.5, 2.5); mode_target=target)
        @test pa_rel.mode === :rel

        # Budget absolute.
        pa_bud = approximate(sin, (-3.0, 3.0);
            target=target, max_coeffs=5,
            degree_policy=:max,
            τ=1e-2, max_depth=12)
        _check_partition(pa_bud, (-3.0, 3.0); mode_target=target)
        @test pa_bud.mode === :abs
        @test pa_bud.max_n == 4

        # Budget relative with :min_cost.
        pa_brel = approximate(exp, (0.5, 2.5);
            target=target, mode=:rel,
            max_coeffs=7, degree_policy=:min_cost,
            τ=1e-2, max_depth=10)
        _check_partition(pa_brel, (0.5, 2.5); mode_target=target)
        @test pa_brel.mode === :rel
    end

    # ---------------------------------------------------------------------------
    @testset "unified approximate: argument validation" begin
        scheme = horner_scheme(4; u=2.0^-53)
        # Neither n nor max_coeffs.
        @test_throws ArgumentError approximate(sin, (-1.0, 1.0); target=1e-3)
        # Both n and max_coeffs.
        @test_throws ArgumentError approximate(
            sin, (-1.0, 1.0); target=1e-3, n=4, max_coeffs=5,
            scheme=scheme)
        # n given but no scheme.
        @test_throws ArgumentError approximate(
            sin, (-1.0, 1.0); target=1e-3, n=4)
        # max_coeffs given but caller passed scheme (degree-specific) instead
        # of scheme_builder — should reject.
        @test_throws ArgumentError approximate(
            sin, (-1.0, 1.0); target=1e-3, max_coeffs=5, scheme=scheme)
        # Bad mode.
        @test_throws ArgumentError approximate(
            sin, (-1.0, 1.0); target=1e-3, mode=:nonsense,
            n=4, scheme=scheme)
    end

    # ---------------------------------------------------------------------------
    @testset "high-level fit interface" begin
        params = recommend_parameters(sin, (-1.0, 1.0);
            target=1e-3, effort=:fast)
        @test params.max_coeffs !== nothing
        @test params.target == 1e-3
        @test params.mode === :abs
        @test occursin("FitParameters(", sprint(show, params))

        single = approxfit(sin, (-1.0, 1.0);
            target=1e-3, degree=3, piecewise=:auto,
            effort=:fast)
        @test single isa Approximation
        @test !is_piecewise(single)
        @test error_bound(single) ≤ 1e-3
        @test coeff_count(single) == 4
        @test abs(single(0.25) - sin(0.25)) ≤ 1e-3

        fma_single = approxfit(sin, (-1.0, 1.0);
            target=1e-3, degree=3, piecewise=false,
            scheme=:horner_fma, effort=:fast,
            strategy=GridSearch(4096))
        @test fma_single.parameters.scheme === :horner_fma
        @test fma_single.parameters.eval_scheme === :horner
        @test fma_single.parameters.eval_op === :fma
        fma_src = provide_source(fma_single; name=:fit_fma_eval)
        @test occursin("fma(y, t,", fma_src)
        @test !occursin("muladd(y, t,", fma_src)

        fma_estrin_single = approxfit(sin, (-1.0, 1.0);
            target=1e-3, degree=3, piecewise=false,
            scheme=:estrin_fma, effort=:fast,
            strategy=GridSearch(4096))
        @test fma_estrin_single.parameters.scheme === :estrin_fma
        @test fma_estrin_single.parameters.eval_scheme === :estrin
        @test fma_estrin_single.parameters.eval_op === :fma
        fma_estrin_src = provide_source(fma_estrin_single; name=:fit_fma_estrin_eval)
        @test occursin("fma(", fma_estrin_src)
        @test occursin("Vector{fit_fma_estrin_eval_T}", fma_estrin_src)
        @test !occursin("muladd(", fma_estrin_src)

        pw = approxfit(exp, (0.5, 1.5);
            rel_tol=1e-3, mode=:rel, max_coeffs=4,
            effort=:fast, max_depth=6)
        @test pw isa Approximation
        @test is_piecewise(pw)
        @test error_bound(pw) ≤ 1e-3
        @test pieces(pw) isa Vector

        src = provide_source(single; name=:fit_single_eval)
        @test occursin("function fit_single_eval", src)
        @test occursin("outside approximation interval", src)

        @test_throws ArgumentError recommend_parameters(
            sin, (-1.0, 1.0); target=1e-3, abs_tol=1e-3)
    end

    # ---------------------------------------------------------------------------
    @testset "target_type and compute_type are independent" begin
        scheme = horner_scheme(3, BigFloat; u=eps(Float32) / 2)
        res = eval_approx_optimize(sin, 3, (-1.0, 1.0), scheme;
            τ=1e-2, max_iter=40, strategy=GridSearch(4096),
            target_type=Float32)
        @test res isa OptimResult{Float32,BigFloat}
        @test eltype(Polynomials.coeffs(res.poly)) === Float32
        @test typeof(res.total_error) === BigFloat
        @test typeof(res.discretization[1].t) === BigFloat

        a = approxfit(sin, (-1.0, 1.0);
            target=1e-2, degree=3, piecewise=false,
            target_type=Float32, compute_type=BigFloat,
            effort=:fast, driver_max_iter=40,
            strategy=GridSearch(4096))
        @test a isa Approximation{Float32,BigFloat}
        @test a.parameters.target_type === Float32
        @test a.parameters.compute_type === BigFloat
        @test eltype(Polynomials.coeffs(a.model.poly)) === Float32
        @test typeof(error_bound(a)) === BigFloat
        @test typeof(a.interval[1]) === BigFloat

        pa = approximate_abs_budget(sin, 4, (-1.0, 1.0);
            target=1e-2, compute_type=BigFloat, target_type=Float32,
            degree_policy=:max, τ=1e-2, max_depth=4,
            driver_max_iter=40, strategy=GridSearch(4096))
        @test pa isa PiecewisePolyApprox{Float32,BigFloat}
        @test eltype(Polynomials.coeffs(pa.pieces[1].result.poly)) === Float32
        @test typeof(pa.worst_error) === BigFloat
        @test typeof(pa.pieces[1].a) === BigFloat

        src = provide_source(a; name=:typed_eval)
        @test occursin("const typed_eval_T = Float32", src)
        @test occursin("const typed_eval_coeff_T = Float32", src)

        src64 = provide_source(a; name=:typed_eval64, eval_type=Float64)
        @test occursin("const typed_eval64_T = Float64", src64)
        @test occursin("const typed_eval64_coeff_T = Float32", src64)
        mod64 = Module(:ProvidedTypedEval64Sandbox)
        Core.eval(mod64, Meta.parse(src64))
        f64 = getfield(mod64, :typed_eval64)
        @test Base.invokelatest(f64, 0.25f0) isa Float64

        src_fma = provide_source(a; name=:typed_eval_fma, eval_op=:fma)
        @test occursin("fma(y, t,", src_fma)
        @test !occursin("muladd(y, t,", src_fma)
        src_estrin_fma = provide_source(a; name=:typed_eval_estrin_fma,
            eval_scheme=:estrin, eval_op=:fma)
        @test occursin("fma(", src_estrin_fma)
        @test occursin("Vector{typed_eval_estrin_fma_T}", src_estrin_fma)
        @test !occursin("muladd(", src_estrin_fma)
        mod_estrin = Module(:ProvidedTypedEstrinEvalSandbox)
        Core.eval(mod_estrin, Meta.parse(src_estrin_fma))
        f_estrin = getfield(mod_estrin, :typed_eval_estrin_fma)
        coeffs = Polynomials.coeffs(a.model.poly)
        @test Base.invokelatest(f_estrin, 0.25f0) ≈ fma_estrin_eval(coeffs, 0.25f0)
        @test_throws ArgumentError provide_source(a; name=:typed_eval_bad, eval_op=:bad)
        @test_throws ArgumentError provide_source(a; name=:typed_eval_bad_scheme,
            eval_scheme=:bad)
    end

    # ---------------------------------------------------------------------------
    @testset "scheme_builder validation" begin
        target = 1e-6
        # A builder that returns the wrong-degree scheme must be rejected.
        bad_builder = d -> horner_scheme(d + 1; u=2.0^-53)
        @test_throws ArgumentError approximate_abs_budget(
            sin, 4, (-1.0, 1.0);
            target=target, scheme_builder=bad_builder)
        # A builder that returns the wrong type must be rejected.
        nonscheme_builder = d -> 42
        @test_throws ArgumentError approximate_abs_budget(
            sin, 4, (-1.0, 1.0);
            target=target, scheme_builder=nonscheme_builder)
    end

    # ---------------------------------------------------------------------------
    @testset "degree_policy validation" begin
        @test_throws ArgumentError approximate_abs_budget(
            sin, 4, (-1.0, 1.0);
            target=1e-6, degree_policy=:bogus)
    end

    # ---------------------------------------------------------------------------
    @testset "PiecewisePolyApprox callable: piece location" begin
        target = 1e-4
        scheme = horner_scheme(3; u=2.0^-53)
        pa = approximate_abs(sin, 3, (-2.0, 2.0), scheme;
            target=target, τ=1e-2, max_depth=10)
        # Sample many points; every one should land in some piece and the
        # returned polynomial should agree with sin to within `target`.
        # We add a small slack because `pa(t)` evaluates via Polynomials.jl
        # rather than the exact Horner scheme that θ models, so the actual
        # eval error can differ slightly from the θ bound.
        for t in range(-2.0, 2.0; length=41)
            y = pa(t)
            @test abs(y - sin(t)) ≤ target + 1e-12
        end
    end

    # ---------------------------------------------------------------------------
    @testset "provide_source / provide: standalone evaluator generation" begin
        # --- Piecewise approximation -> standalone function source ---
        target = 1e-6
        scheme = horner_scheme(4; u=2.0^-53)
        pa = approximate_abs(sin, 4, (-3.0, 3.0), scheme;
            target=target, τ=1e-2, max_depth=12)

        src_pa = provide_source(pa; name=:sin_pw_eval, check_domain=true)
        @test occursin("function sin_pw_eval", src_pa)
        @test occursin("sin_pw_eval_bounds", src_pa)
        @test occursin("sin_pw_eval_coeffs", src_pa)

        mod_pa = Module(:ProvidedPiecewiseSandbox)
        Core.eval(mod_pa, Meta.parse(src_pa))
        f_pa = getfield(mod_pa, :sin_pw_eval)

        for t in range(-3.0, 3.0; length=31)
            @test f_pa(t) ≈ pa(t) atol = 1e-12 rtol = 0
        end
        @test_throws DomainError f_pa(3.5)

        # --- Single polynomial (OptimResult) -> standalone function source ---
        res = eval_approx_optimize(sin, 5, (-1.0, 1.0), horner_scheme(5; u=2.0^-53);
            τ=1e-3, max_iter=80)

        src_single = provide_source(res;
            name=:sin_poly_eval,
            interval=(-1.0, 1.0),
            check_domain=true)
        @test occursin("function sin_poly_eval", src_single)

        mod_single = Module(:ProvidedSingleSandbox)
        Core.eval(mod_single, Meta.parse(src_single))
        f_single = getfield(mod_single, :sin_poly_eval)

        for t in range(-1.0, 1.0; length=25)
            @test f_single(t) ≈ res.poly(t) atol = 1e-12 rtol = 0
        end
        @test_throws DomainError f_single(1.2)

        # --- provide(...) installs function and returns fn + source ---
        install_mod = Module(:ProvideInstallSandbox)
        out = provide(pa; name=:sin_pw_eval2, module_=install_mod)
        @test occursin("function sin_pw_eval2", out.source)
        @test Base.invokelatest(out.fn, 0.5) ≈ pa(0.5) atol = 1e-12 rtol = 0

        # Existing symbol in target module should require force=true.
        @test_throws ArgumentError provide(pa; name=:sin_pw_eval2, module_=install_mod)
    end

end  # @testset PolynomialErrorOptimization
