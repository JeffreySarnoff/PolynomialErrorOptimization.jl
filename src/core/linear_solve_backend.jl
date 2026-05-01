"""
    linear_solve_backend.jl

Internal linear-system backend wrappers.

This keeps solver-policy choices in one place so call sites in
`init_points.jl`, `solve_primal.jl`, and `exchange.jl` stay focused on the
algorithm logic.
"""

"""
    solve_dense_system(A, b; alg = LinearSolve.LUFactorization()) -> x

Solve a dense linear system `A * x = b` via LinearSolve.jl.

The default algorithm matches the project’s current use case (small dense
systems) while allowing future algorithm overrides from one central helper.
"""
function solve_dense_system(A::AbstractMatrix{T},
    b::AbstractVector{T};
    alg=LinearSolve.LUFactorization()) where T<:AbstractFloat
    @argcheck size(A, 1) == size(A, 2) DimensionMismatch(
        "solve_dense_system: A must be square, got size(A) = $(size(A))")
    @argcheck length(b) == size(A, 1) DimensionMismatch(
        "solve_dense_system: length(b) = $(length(b)) must match size(A,1) = $(size(A, 1))")

    prob = LinearSolve.LinearProblem(A, b)
    sol = LinearSolve.solve(prob, alg)
    return sol.u
end
