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
    @test pieces(single) === nothing
    @test error_bound(single) ≤ 1e-3
    @test coeff_count(single) == 4
    @test abs(single(0.25) - sin(0.25)) ≤ 1e-3

    abs_fit = fit_abs(sin, (-1.0, 1.0);
        target=1e-3, degree=3, piecewise=false,
        effort=:fast)
    @test abs_fit isa Approximation
    @test abs_fit.parameters.mode === :abs

    rel_fit = fit_rel(exp, (0.5, 1.5);
        rel_tol=1e-3, degree=3, piecewise=false,
        effort=:fast)
    @test rel_fit isa Approximation
    @test rel_fit.parameters.mode === :rel

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

@testset "root export surface" begin
    exported = Set(names(PolynomialErrorOptimization))

    for sym in (:approxfit, :fit_abs, :fit_rel, :plan_fit, :recommend_parameters,
        :ObjectiveSpec, :ComplexitySpec, :PrecisionSpec, :SearchSpec,
        :FitPlan, :FitParameters, :Approximation,
        :error_bound, :coeff_count, :is_piecewise, :pieces,
        :provide_source, :provide, :provide_file,
        :horner_scheme, :fma_horner_scheme, :estrin_scheme, :fma_estrin_scheme)
        @test sym in exported
    end

    for sym in (:eval_approx_optimize, :eval_approx_optimize_relative,
        :eval_approx_optimize_relative_zero, :approximate_abs,
        :approximate_abs_budget, :GridSearch, :Index, :exchange,
        :ResultBasis, :basis_info, :solution_coefficients)
        @test !(sym in exported)
        @test isdefined(PolynomialErrorOptimization, sym)
    end
end

@testset "stable workflow documentation smoke tests" begin
    readme_fit = approxfit(sin, (-2.0, 2.0);
        target=1e-4, effort=:fast, max_depth=6)
    @test readme_fit isa Approximation
    @test error_bound(readme_fit) ≤ 1e-4
    @test coeff_count(readme_fit) ≥ 1

    docs_fit = approxfit(sin, (-3.0, 3.0);
        target=1e-3, effort=:balanced, max_depth=6)
    @test docs_fit isa Approximation

    params = recommend_parameters(sin, (-3.0, 3.0);
        target=1e-3, effort=:balanced)
    replay = approxfit(sin, (-3.0, 3.0), params)
    @test replay isa Approximation
    @test replay.parameters.mode === params.mode
end

@testset "typed fit planning" begin
    plan = plan_fit(sin, (-2.0, 2.0);
        target=1e-4, effort=:fast, max_depth=6)
    @test plan isa FitPlan
    @test plan.objective.mode === :abs
    @test plan.objective.target == plan.parameters.target
    @test plan.interval == (-2.0, 2.0)
    @test plan.inferred.degree
    @test plan.inferred.max_coeffs
    @test plan.inferred.piecewise
    @test occursin("FitPlan(", sprint(show, plan))

    replay = approxfit(sin, plan)
    @test replay isa Approximation
    @test replay.parameters.target == plan.parameters.target
    @test replay.parameters.max_depth == plan.parameters.max_depth

    objective = ObjectiveSpec(:rel, 1e-3)
    complexity = ComplexitySpec(degree=3, piecewise=false)
    precision = PrecisionSpec(target_type=Float32, compute_type=BigFloat)
    search = SearchSpec(BigFloat;
        scheme=:horner_fma,
        effort=:fast,
        τ=1e-2,
        max_depth=6,
        driver_max_iter=40,
        strategy=GridSearch(4096))
    typed = plan_fit(exp, (0.5, 1.5), objective;
        complexity=complexity,
        precision=precision,
        search=search)

    @test typed isa FitPlan{Float32,BigFloat}
    @test typed.objective.mode === :rel
    @test typed.complexity.degree == 3
    @test typed.complexity.piecewise === false
    @test typed.search.scheme === :horner_fma
    @test typed.search.effort === :fast
    @test typed.parameters.target_type === Float32
    @test typed.parameters.compute_type === BigFloat
    @test !typed.inferred.degree
    @test !typed.inferred.τ
    @test !typed.inferred.max_depth
    @test !typed.inferred.driver_max_iter

    fit_typed = approxfit(exp, typed)
    @test fit_typed isa Approximation{Float32,BigFloat}
    @test fit_typed.parameters.mode === :rel
    @test fit_typed.parameters.scheme === :horner_fma
    @test !is_piecewise(fit_typed)

    @test_throws ArgumentError ObjectiveSpec(:bad, 1e-3)
    @test_throws ArgumentError ComplexitySpec(degree=3, max_coeffs=4)
    @test_throws ArgumentError SearchSpec(max_depth=-1)
end

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

@testset "scheme_builder validation" begin
    target = 1e-6
    bad_builder = d -> horner_scheme(d + 1; u=2.0^-53)
    @test_throws ArgumentError approximate_abs_budget(
        sin, 4, (-1.0, 1.0);
        target=target, scheme_builder=bad_builder)
    nonscheme_builder = d -> 42
    @test_throws ArgumentError approximate_abs_budget(
        sin, 4, (-1.0, 1.0);
        target=target, scheme_builder=nonscheme_builder)
end

@testset "degree_policy validation" begin
    @test_throws ArgumentError approximate_abs_budget(
        sin, 4, (-1.0, 1.0);
        target=1e-6, degree_policy=:bogus)
end

@testset "PiecewisePolyApprox callable: piece location" begin
    target = 1e-4
    scheme = horner_scheme(3; u=2.0^-53)
    pa = approximate_abs(sin, 3, (-2.0, 2.0), scheme;
        target=target, τ=1e-2, max_depth=10)
    for t in range(-2.0, 2.0; length=41)
        y = pa(t)
        @test abs(y - sin(t)) ≤ target + 1e-12
    end
end

@testset "provide_source / provide: standalone evaluator generation" begin
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

    install_mod = Module(:ProvideInstallSandbox)
    out = provide(pa; name=:sin_pw_eval2, module_=install_mod)
    @test occursin("function sin_pw_eval2", out.source)
    @test Base.invokelatest(out.fn, 0.5) ≈ pa(0.5) atol = 1e-12 rtol = 0

    @test_throws ArgumentError provide(pa; name=:sin_pw_eval2, module_=install_mod)
end