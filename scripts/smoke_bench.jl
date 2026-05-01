using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using PolynomialErrorOptimization

cases = [
    ("sin budget fast", sin, (-2.0, 2.0),
        (; target=1e-6, max_coeffs=5, effort=:fast)),
    ("exp relative budget", exp, (0.5, 1.5),
        (; rel_tol=1e-5, mode=:rel, max_coeffs=4, effort=:fast, max_depth=8)),
    ("cos fixed global", cos, (-1.0, 1.0),
        (; target=1e-4, degree=4, piecewise=false, effort=:fast)),
]

println("PolynomialErrorOptimization smoke benchmark")
for (label, f, I, kwargs) in cases
    local approx
    seconds = @elapsed approx = approxfit(f, I; kwargs...)
    println(rpad(label, 22),
        " time=", round(seconds; digits=3), "s",
        " error=", error_bound(approx),
        " coeffs=", coeff_count(approx),
        " piecewise=", is_piecewise(approx))
end
