include("common.jl")

@testset "PolynomialErrorOptimization" begin
    include("core/core_tests.jl")
    include("schemes/scheme_tests.jl")
    include("exchange/exchange_tests.jl")
    include("piecewise/piecewise_tests.jl")
    include("interface/interface_tests.jl")
    include("docs/docs_tests.jl")
end
