@testset "solve_primal: Lemma 2 — recovers Remez when θ ≡ 0" begin
    n = 4
    f = sin
    π_zero = [(t::Float64) -> zeros(Float64, n + 1) for _ in 1:1]
    scheme_zero = EvalScheme(n, 1, π_zero, "zero-error")
    I = (-1.0, 1.0)

    ω, _ = init_points(scheme_zero, I)
    ā, a = solve_primal(f, scheme_zero, ω)

    for ωⱼ in ω
        residual = monomial_dot(a, ωⱼ.t) - f(ωⱼ.t)
        @test abs(abs(residual) - ā) < 1e-10
    end
    @test ā > 0
end

@testset "find_new_index: maximum and signature reconstruction" begin
    n = 3
    scheme = horner_scheme(n; u=2.0^-30)
    I = (-1.0, 1.0)
    f = exp
    a = Float64[1.0, 1.0, 0.5, 1.0/6.0]

    M = 50_001
    ωstar, astar = find_new_index(f, scheme, I, a; strategy=GridSearch(M))
    sampled = sampled_total_error(a, f, scheme, I, M)
    @test astar ≥ sampled - 1e-10
    @test astar ≤ sampled + 1e-10

    v0 = monomial_dot(a, ωstar.t) - f(ωstar.t)
    @test ωstar.σ[1] == -signum_int8(v0)

    ωopt, aopt = find_new_index(f, scheme, I, a;
        strategy=GridThenOptim(10_001; bracket=4))
    @test aopt ≥ sampled - 1e-10
    @test I[1] ≤ ωopt.t ≤ I[2]
    v0opt = monomial_dot(a, ωopt.t) - f(ωopt.t)
    @test ωopt.σ[1] == -signum_int8(v0opt)
end

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
    @test all(y_new .≥ -1e-12)

    m = n + 2
    A_new = Matrix{Float64}(undef, m, m)
    for j in 1:m
        A_new[:, j] = α(ω_new[j], scheme)
    end
    @test rank(A_new) == m
    @test A_new * y_new ≈ [1.0; zeros(m - 1)] atol = 1e-9
end

@testset "eval_approx_optimize: end-to-end on sin / [-1, 1]" begin
    n = 5
    scheme = horner_scheme(n; u=2.0^-53)
    res = eval_approx_optimize(sin, n, (-1.0, 1.0), scheme;
        τ=1e-3, max_iter=50)
    @test res.converged
    @test res.iterations ≤ 50
    @test res.total_error < 1e-3
    @test abs(res.poly(0.5) - sin(0.5)) < res.total_error
    @test abs(res.poly(-0.7) - sin(-0.7)) < res.total_error
end

@testset "eval_approx_optimize: Airy-toy (paper §6.1)" begin
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

    a = Polynomials.coeffs(res.poly)
    @test sampled_total_error(a, f, scheme, I, M) ≤ res.total_error + 1e-10
end

@testset "eval_approx_optimize: FMA Horner smoke test" begin
    n = 5
    scheme = fma_horner_scheme(n; u=2.0^-53)
    res = eval_approx_optimize(sin, n, (-1.0, 1.0), scheme;
        τ=1e-3, max_iter=50)
    @test res.converged
    @test res.total_error < 1e-3
    @test abs(res.poly(0.5) - sin(0.5)) < res.total_error
end

@testset "eval_approx_optimize: FMA Estrin smoke test" begin
    n = 5
    scheme = fma_estrin_scheme(n; u=2.0^-53)
    res = eval_approx_optimize(sin, n, (-1.0, 1.0), scheme;
        τ=1e-3, max_iter=50)
    @test res.converged
    @test res.total_error < 1e-3
    @test abs(res.poly(0.5) - sin(0.5)) < res.total_error
end

@testset "eval_approx_optimize_relative: smoke test" begin
    n = 4
    scheme = horner_scheme(n; u=2.0^-53)
    res = eval_approx_optimize_relative(exp, n, (0.5, 1.5), scheme;
        τ=1e-3, max_iter=50)
    @test res.converged
    @test res.total_error > 0
    @test res.total_error < 1e-3

    neg_res = eval_approx_optimize_relative(t -> -exp(t), n, (0.5, 1.5), scheme;
        τ=1e-3, max_iter=50)
    @test neg_res.converged
    @test neg_res.total_error > 0
    @test neg_res.total_error < 1e-3
end

@testset "relative-zero initialization and preflight are zero-safe" begin
    f = t -> t^2 * (1 + t)
    scheme = horner_scheme(4; u=2.0^-53)
    mode = RelativeZeroMode{Float64}(0.0, 2)

    ω, _ = init_points(scheme, (-0.5, 1.2), mode; f=f)
    @test all(!iszero(node.t) for node in ω)

    ωstar, astar = find_new_index(f, scheme, (-0.5, 1.2),
        Float64[0.0, 0.0, 1.0, 1.0, 0.0], mode;
        strategy=GridSearch(257))
    @test !iszero(ωstar.t)
    @test isfinite(astar)

    @test_throws DomainError eval_approx_optimize_relative_zero(
        t -> t^2 * (1 + t),
        4,
        (-1.0, 0.8),
        scheme;
        t_z=0.0,
        s=2,
        τ=1.0,
        max_iter=5,
        strategy=GridSearch(257))
end

@testset "eval_approx_optimize_relative_zero: smoke test" begin
    f = t -> t^2 * (1 + t)
    n = 3
    scheme = horner_scheme(n; u=2.0^-53)
    res = eval_approx_optimize_relative_zero(f, n, (-0.5, 1.0), scheme;
        t_z=0.0,
        s=2,
        τ=1.0,
        max_iter=20,
        strategy=GridSearch(257))
    @test res.converged
    @test res.iterations ≤ 20
    @test res.total_error < 1e-12

    info = basis_info(res)
    @test info.coefficient_basis === :monomial
    @test info.solution_basis === :monomial
    @test info.shift == 0.0
    @test info.zero_order == 2

    coeffs = solution_coefficients(res)
    @test coeffs[1] == 0.0
    @test coeffs[2] == 0.0
    for t in (-0.4, 0.25, 0.9)
        @test abs(res.poly(t) - f(t)) < 1e-12
    end
end

@testset "eval_approx_optimize_relative_zero: transcendental smoke test" begin
    f = t -> t * exp(t)
    n = 4
    scheme = horner_scheme(n; u=2.0^-53)
    res = eval_approx_optimize_relative_zero(f, n, (-0.35, 1.1), scheme;
        t_z=0.0,
        s=1,
        τ=1e-2,
        max_iter=20,
        strategy=GridSearch(4097))
    @test res.converged
    @test res.iterations ≤ 20
    @test res.total_error < 2e-3

    info = basis_info(res)
    @test info.coefficient_basis === :monomial
    @test info.solution_basis === :monomial
    @test info.zero_order == 1

    for t in (-0.2, 0.4, 1.0)
        @test abs(res.poly(t) - f(t)) ≤ res.total_error
    end
end

@testset "relative-zero representation is basis-explicit" begin
    n = 2
    t_z = 0.8
    shifted = Float64[0.0, 1.5, -0.25]
    mode = RelativeZeroMode{Float64}(t_z, 1)
    poly = PolynomialErrorOptimization._result_polynomial(shifted, mode, Float64)
    basis = PolynomialErrorOptimization._result_basis(mode, Float64)
    res = OptimResult{Float64,Float64}(
        poly,
        0.0,
        0.0,
        0,
        Index{Float64}[],
        Float64[],
        true,
        basis,
        shifted)

    info = basis_info(res)
    @test info isa ResultBasis{Float64}
    @test info.coefficient_basis === :monomial
    @test info.solution_basis === :shifted
    @test info.shift == t_z
    @test info.zero_order == 1

    recovered = solution_coefficients(res)
    monomial = Polynomials.coeffs(res.poly)
    @test recovered == shifted
    @test length(monomial) == n + 1
    @test recovered[1] == 0.0

    shifted_eval(t) = sum(recovered[j+1] * (t - t_z)^j for j in 0:n)
    for t in (0.6, 0.8, 1.0, 1.2, 1.4)
        @test res.poly(t) ≈ shifted_eval(t) atol = 1e-12 rtol = 1e-12
    end

    src = provide_source(res; name=:relzero_eval, interval=(0.3, 1.6), check_domain=true)
    mod = Module(:RelativeZeroProvideSandbox)
    Core.eval(mod, Meta.parse(src))
    fn = getfield(mod, :relzero_eval)
    @test Base.invokelatest(fn, 1.2) ≈ res.poly(1.2) atol = 1e-12 rtol = 0
end

@testset "ConvergenceFailure with too-tight max_iter" begin
    n = 6
    scheme = horner_scheme(n; u=2.0^-12)
    f = t -> sin(3t)
    I = (-2.0, 2.0)
    @test_throws ConvergenceFailure eval_approx_optimize(
        f, n, I, scheme; τ=1e-12, max_iter=1)
end