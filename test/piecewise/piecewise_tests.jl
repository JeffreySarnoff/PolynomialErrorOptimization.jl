@testset "approximate_abs: end-to-end on sin / [-3, 3]" begin
    target = 1e-6
    scheme = horner_scheme(4; u=2.0^-53)
    pa = approximate_abs(sin, 4, (-3.0, 3.0), scheme;
        target=target, τ=1e-2, max_depth=12)
    @test pa isa PiecewisePolyApprox
    @test pa.mode === :abs
    @test pa.max_n == 4
    @test pa.target == target
    check_partition(pa, (-3.0, 3.0); mode_target=target)
    for t in (-2.7, -1.0, 0.0, 0.5, 1.9, 2.999)
        @test abs(pa(t) - sin(t)) ≤ target
    end
    @test pa(-3.0) isa Float64
    @test pa(3.0) isa Float64
    @test_throws DomainError pa(3.5)
    @test_throws DomainError pa(-3.5)
end

@testset "approximate_abs: constants (n = 0)" begin
    target = 1e-2
    scheme = horner_scheme(0; u=2.0^-53)
    pa = approximate_abs(cos, 0, (-1.0, 1.0), scheme;
        target=target, τ=0.1, max_depth=20)
    @test pa.max_n == 0
    check_partition(pa, (-1.0, 1.0); mode_target=target)
    for p in pa.pieces
        @test length(Polynomials.coeffs(p.result.poly)) == 1
    end
end

@testset "approximate_rel: smoke test on exp / [0.5, 2.5]" begin
    target = 1e-6
    scheme = horner_scheme(5; u=2.0^-53)
    pa = approximate_rel(exp, 5, (0.5, 2.5), scheme;
        target=target, τ=1e-2, max_depth=10)
    @test pa.mode === :rel
    check_partition(pa, (0.5, 2.5); mode_target=target)
    for t in (0.6, 1.0, 1.7, 2.4)
        @test abs(pa(t) - exp(t)) ≤ target * exp(t)
    end
end

@testset "approximate_abs_budget :max policy" begin
    target = 1e-6
    pa = approximate_abs_budget(sin, 5, (-3.0, 3.0);
        target=target,
        degree_policy=:max,
        τ=1e-2, max_depth=12)
    @test pa.max_n == 4
    check_partition(pa, (-3.0, 3.0); mode_target=target)
    for p in pa.pieces
        @test length(Polynomials.coeffs(p.result.poly)) == 5
    end
end

@testset "approximate_abs_budget :min policy uses ≤ :max coefficients" begin
    target = 1e-4
    common = (target=target, τ=1e-2, max_depth=12)
    pa_max = approximate_abs_budget(sin, 6, (-2.0, 2.0);
        degree_policy=:max, common...)
    pa_min = approximate_abs_budget(sin, 6, (-2.0, 2.0);
        degree_policy=:min, common...)
    check_partition(pa_max, (-2.0, 2.0); mode_target=target)
    check_partition(pa_min, (-2.0, 2.0); mode_target=target)
    for p in pa_min.pieces
        @test length(Polynomials.coeffs(p.result.poly)) ≤ 6
    end
end

@testset "approximate_abs_budget :min_cost ≤ :max total coeffs" begin
    target = 1e-4
    common = (target=target, τ=1e-2, max_depth=10)
    pa_max = approximate_abs_budget(sin, 6, (-2.0, 2.0);
        degree_policy=:max, common...)
    pa_minc = approximate_abs_budget(sin, 6, (-2.0, 2.0);
        degree_policy=:min_cost, common...)
    check_partition(pa_minc, (-2.0, 2.0); mode_target=target)
    total_max = sum(length(Polynomials.coeffs(p.result.poly))
                    for p in pa_max.pieces)
    total_minc = sum(length(Polynomials.coeffs(p.result.poly))
                     for p in pa_minc.pieces)
    @test total_minc ≤ total_max
end

@testset "approximate_abs_budget: :min_cost ≤ :min total coeffs" begin
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

@testset "total_coeffs cap is enforced" begin
    target = 1e-6
    scheme = horner_scheme(4; u=2.0^-53)
    pa = approximate_abs(sin, 4, (-3.0, 3.0), scheme;
        target=target, τ=1e-2,
        total_coeffs=1000, max_depth=12)
    total = sum(length(Polynomials.coeffs(p.result.poly)) for p in pa.pieces)
    @test total ≤ 1000
    @test_throws ErrorException approximate_abs(
        sin, 4, (-3.0, 3.0), scheme;
        target=target, τ=1e-2,
        total_coeffs=4, max_depth=12)
end

@testset "total_coeffs cap interacts with :min_cost" begin
    target = 1e-4
    pa = approximate_abs_budget(sin, 6, (-2.0, 2.0);
        target=target,
        degree_policy=:min_cost,
        τ=1e-2, max_depth=10,
        total_coeffs=200)
    total = sum(length(Polynomials.coeffs(p.result.poly)) for p in pa.pieces)
    @test total ≤ 200
    check_partition(pa, (-2.0, 2.0); mode_target=target)
    @test_throws ErrorException approximate_abs_budget(
        sin, 6, (-2.0, 2.0);
        target=target, degree_policy=:min_cost,
        τ=1e-2, max_depth=4, total_coeffs=1)
end

@testset "min_width / max_depth refusal paths" begin
    target = 1e-12
    scheme = horner_scheme(2; u=2.0^-53)
    err_depth = try
        approximate_abs(
            sin, 2, (-3.0, 3.0), scheme;
            target=target, τ=1e-3, max_depth=0)
        nothing
    catch e
        e
    end
    @test err_depth isa ErrorException
    @test occursin("reason: err > target", sprint(showerror, err_depth))

    err_width = try
        approximate_abs(
            sin, 2, (-3.0, 3.0), scheme;
            target=target, τ=1e-3,
            max_depth=30, min_width=5.0)
        nothing
    catch e
        e
    end
    @test err_width isa ErrorException
    @test occursin("reason: err > target", sprint(showerror, err_width))
end

@testset "structured fit attempt diagnostics" begin
    abs_scheme = horner_scheme(0; u=2.0^-53)
    abs_cfg = PolynomialErrorOptimization.FitConfig(
        1e-12, 1e-3, 40, GridSearch(4096), Float64)
    res_abs, ok_abs, report_abs = PolynomialErrorOptimization._try_fit(
        sin, 0, (-3.0, 3.0), abs_scheme, AbsoluteMode(), abs_cfg)
    @test !ok_abs
    @test report_abs isa PolynomialErrorOptimization.FitAttemptReport
    @test report_abs.kind === :target_miss
    @test report_abs.interval == (-3.0, 3.0)
    @test report_abs.degree == 0
    @test report_abs.mode === :abs
    @test report_abs.target == 1e-12
    @test report_abs.achieved_error == res_abs.total_error
    @test report_abs.exception === nothing
    @test sprint(show, report_abs) == "err > target"

    rel_scheme = horner_scheme(2; u=2.0^-53)
    rel_cfg = PolynomialErrorOptimization.FitConfig(
        1e-6, 1e-3, 40, GridSearch(4096), Float64)
    _, ok_rel, report_rel = PolynomialErrorOptimization._try_fit(
        sin, 2, (-1.0, 1.0), rel_scheme, RelativeMode(), rel_cfg)
    @test !ok_rel
    @test report_rel isa PolynomialErrorOptimization.FitAttemptReport
    @test report_rel.kind === :driver_exception
    @test report_rel.mode === :rel
    @test report_rel.exception isa DomainError
    @test occursin("DomainError", sprint(show, report_rel))
end

@testset "unified approximate: dispatch routing" begin
    target = 1e-6
    scheme = horner_scheme(4; u=2.0^-53)

    pa = approximate(sin, (-3.0, 3.0);
        target=target, n=4, scheme=scheme,
        τ=1e-2, max_depth=12)
    check_partition(pa, (-3.0, 3.0); mode_target=target)
    @test pa.mode === :abs
    @test pa.max_n == 4

    pa_rel = approximate(exp, (0.5, 2.5);
        target=target, mode=:rel,
        n=5, scheme=horner_scheme(5; u=2.0^-53),
        τ=1e-2, max_depth=10)
    check_partition(pa_rel, (0.5, 2.5); mode_target=target)
    @test pa_rel.mode === :rel

    pa_bud = approximate(sin, (-3.0, 3.0);
        target=target, max_coeffs=5,
        degree_policy=:max,
        τ=1e-2, max_depth=12)
    check_partition(pa_bud, (-3.0, 3.0); mode_target=target)
    @test pa_bud.mode === :abs
    @test pa_bud.max_n == 4

    pa_brel = approximate(exp, (0.5, 2.5);
        target=target, mode=:rel,
        max_coeffs=7, degree_policy=:min_cost,
        τ=1e-2, max_depth=10)
    check_partition(pa_brel, (0.5, 2.5); mode_target=target)
    @test pa_brel.mode === :rel
end

@testset "unified approximate: argument validation" begin
    scheme = horner_scheme(4; u=2.0^-53)
    @test_throws ArgumentError approximate(sin, (-1.0, 1.0); target=1e-3)
    @test_throws ArgumentError approximate(
        sin, (-1.0, 1.0); target=1e-3, n=4, max_coeffs=5,
        scheme=scheme)
    @test_throws ArgumentError approximate(
        sin, (-1.0, 1.0); target=1e-3, n=4)
    @test_throws ArgumentError approximate(
        sin, (-1.0, 1.0); target=1e-3, max_coeffs=5, scheme=scheme)
    @test_throws ArgumentError approximate(
        sin, (-1.0, 1.0); target=1e-3, mode=:nonsense,
        n=4, scheme=scheme)
end