@testset "documentation smoke tests" begin
    home = approxfit(sin, (-2.0, 2.0);
        target=1e-4, effort=:fast, max_depth=6)
    @test home isa Approximation
    @test error_bound(home) ≤ 1e-4
    @test coeff_count(home) ≥ 1
    @test abs(home(0.25) - sin(0.25)) ≤ 1e-4

    guide = approxfit(sin, (-3.0, 3.0);
        target=1e-3, effort=:fast, max_depth=6)
    @test guide isa Approximation
    @test error_bound(guide) ≤ 1e-3

    plan = plan_fit(sin, (-3.0, 3.0);
        target=1e-3, effort=:fast, max_depth=6)
    @test plan isa FitPlan
    @test plan.parameters.mode === :abs
    replay = approxfit(sin, plan)
    @test replay isa Approximation
    @test replay.parameters.mode === plan.parameters.mode
    @test replay.parameters.target == plan.parameters.target

    expert = PolynomialErrorOptimization.eval_approx_optimize(
        sin,
        5,
        (-1.0, 1.0),
        horner_scheme(5; u=2.0^-53);
        τ=1e-3,
        max_iter=50,
        strategy=PolynomialErrorOptimization.GridSearch(4096))
    @test expert isa OptimResult
    @test expert.converged
    @test expert.total_error < 1e-3

    rz_result = PolynomialErrorOptimization.eval_approx_optimize_relative_zero(
        t -> t * exp(t),
        4,
        (-0.35, 1.1),
        horner_scheme(4; u=2.0^-53);
        t_z=0.0,
        s=1,
        τ=1e-2,
        max_iter=20,
        strategy=PolynomialErrorOptimization.GridSearch(4097))
    info = PolynomialErrorOptimization.basis_info(rz_result)
    shifted = PolynomialErrorOptimization.solution_coefficients(rz_result)
    @test rz_result.converged
    @test info.coefficient_basis === :monomial
    @test info.solution_basis === :monomial
    @test info.shift == 0.0
    @test info.zero_order == 1
    @test shifted[1] == 0.0
    @test rz_result.total_error < 2e-3
    @test rz_result.poly(0.0) == 0.0
end