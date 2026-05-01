@testset "horner_scheme: shape and special cases" begin
    s0 = horner_scheme(0; u=2.0^-53)
    @test s0.n == 0
    @test s0.k == 1
    @test s0.π[1](0.5) == [0.0]

    u = 2.0^-12
    s1 = horner_scheme(1; u=u)
    @test s1.n == 1
    @test s1.k == 2
    @test s1.π[1](0.7) ≈ [u, u * 0.7]
    @test s1.π[2](0.7) ≈ [0.0, u * 0.7]

    s3 = horner_scheme(3; u=u)
    @test s3.n == 3 && s3.k == 4
    t = 0.4
    @test s3.π[1](t) ≈ [u, u * t, u * t^2, u * t^3]
    @test s3.π[2](t) ≈ [0.0, 2u * t, 2u * t^2, 2u * t^3]
    @test s3.π[3](t) ≈ [0.0, 0.0, 2u * t^2, 2u * t^3]
    @test s3.π[4](t) ≈ [0.0, 0.0, 0.0, u * t^3]
end

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

@testset "lin_eval_error: matches closed-form Horner (Example 1)" begin
    n = 4
    u = 2.0^-12
    e, ids_to_u = horner_expr(n; u=u)
    θ = lin_eval_error(e)
    sym_scheme = build_eval_scheme(θ, ids_to_u, n, "horner-symbolic")
    closed_scheme = horner_scheme(n; u=u)

    for t in (-0.9, -0.3, 0.1, 0.7, 0.99)
        a = randn(n + 1)
        sym_θ = sum(abs(dot_view(a, sym_scheme.π[i](t)))
                    for i in 1:sym_scheme.k)
        closed_θ = sum(abs(dot_view(a, closed_scheme.π[i](t)))
                       for i in 1:closed_scheme.k)
        @test closed_θ ≥ sym_θ * (1 - 1e-9)
    end
end

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