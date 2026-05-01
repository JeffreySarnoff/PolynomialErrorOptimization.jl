using Documenter
using PolynomialErrorOptimization

makedocs(
    sitename="PolynomialErrorOptimization.jl",
    modules=[PolynomialErrorOptimization],
    remotes=nothing,
    doctest=false,
    checkdocs=:none,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical=nothing,
        edit_link=nothing,
        repolink=nothing,
    ),
    pages=[
        "Home" => "index.md",
        "High-Level Interface" => "high-level-interface.md",
        "User Guide" => "user-guide.md",
        "Parameter Selection" => "parameter-selection.md",
        "Examples" => "examples.md",
        "Technical Guide" => "technical-guide.md",
        "API Reference" => "api.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo="github.com/JeffreySarnoff/PolynomialErrorOptimization.jl.git",
        devbranch="main",
    )
end
