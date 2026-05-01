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

@testset "α / α! / c: row construction" begin
    n = 3
    scheme = horner_scheme(n; u=2.0^-12)
    σ = Int8[1, 0, 0, 0, 0]
    ω = Index(0.5, σ)

    row = α(ω, scheme)
    @test length(row) == n + 2
    @test row[1] == 1.0
    @test row[2:end] ≈ [1.0, 0.5, 0.25, 0.125]

    σm = Int8[-1, 0, 0, 0, 0]
    rowm = α(Index(0.5, σm), scheme)
    @test rowm[2:end] ≈ -[1.0, 0.5, 0.25, 0.125]

    out = zeros(n + 2)
    α!(out, ω, scheme)
    @test out ≈ row

    @test c(ω, t -> 7.0) ≈ 7.0
    @test c(Index(0.5, σm), t -> 7.0) ≈ -7.0
end

@testset "init_points: Lemma 1 properties" begin
    for n in 0:6
        scheme = horner_scheme(n; u=2.0^-12)
        I = (-1.0, 1.0)
        ω, y = init_points(scheme, I)
        @test length(ω) == n + 2
        @test length(y) == n + 2
        @test all(y .> 0)

        m = n + 2
        A = Matrix{Float64}(undef, m, m)
        for j in 1:m
            A[:, j] = α(ω[j], scheme)
        end
        @test A * y ≈ [1.0; zeros(m - 1)] atol = 1e-10
        for ωⱼ in ω
            @test -1 - 1e-12 ≤ ωⱼ.t ≤ 1 + 1e-12
        end
    end
end