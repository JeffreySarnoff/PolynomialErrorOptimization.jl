# ---------------------------------------------------------------------------
# Internal: shared adaptive bisection
# ---------------------------------------------------------------------------

# A pending subinterval to try fitting on. Stored on a stack (vector used
# as LIFO) so that pieces come out in left-to-right order if we always push
# the right child first and then the left child.
struct PendingInterval{T<:AbstractFloat}
    a::T
    b::T
    depth::Int
end

struct FitConfig{TargetT<:AbstractFloat,T<:AbstractFloat}
    target::T
    τ::T
    driver_max_iter::Int
    strategy::SearchStrategy
    target_type::Type{TargetT}
    function FitConfig(target::T, τ::T, driver_max_iter::Integer,
        strategy::SearchStrategy,
        target_type::Type{TargetT}=T) where {TargetT<:AbstractFloat,T<:AbstractFloat}
        @argcheck target > zero(T) ArgumentError("target must be > 0, got $target")
        @argcheck τ > zero(T) ArgumentError("τ must be > 0, got $τ")
        @argcheck driver_max_iter ≥ 0 ArgumentError(
            "driver_max_iter must be ≥ 0, got $driver_max_iter")
        return new{TargetT,T}(target, τ, Int(driver_max_iter), strategy, target_type)
    end
end

struct BisectConfig{TargetT<:AbstractFloat,T<:AbstractFloat}
    mode::AbstractMode
    max_n::Int
    target::T
    τ::T
    max_depth::Int
    min_width::T
    total_coeffs::Int
    target_type::Type{TargetT}
    verbose::Bool
    function BisectConfig(mode::AbstractMode,
        max_n::Integer,
        target::T,
        τ::T,
        max_depth::Integer,
        min_width::T,
        total_coeffs::Integer,
        target_type::Type{TargetT},
        verbose::Bool) where {TargetT<:AbstractFloat,T<:AbstractFloat}
        @argcheck mode isa AbsoluteMode || mode isa RelativeMode ArgumentError(
            "mode must be AbsoluteMode() or RelativeMode(), got $(typeof(mode))")
        @argcheck max_n ≥ 0 ArgumentError("max_n must be ≥ 0, got $max_n")
        @argcheck target > zero(T) ArgumentError("target must be > 0, got $target")
        @argcheck τ > zero(T) ArgumentError("τ must be > 0, got $τ")
        @argcheck max_depth ≥ 0 ArgumentError("max_depth must be ≥ 0, got $max_depth")
        @argcheck min_width ≥ zero(T) ArgumentError("min_width must be ≥ 0, got $min_width")
        @argcheck total_coeffs ≥ 0 ArgumentError(
            "total_coeffs must be ≥ 0, got $total_coeffs")
        return new{TargetT,T}(mode, Int(max_n), target, τ, Int(max_depth), min_width,
            Int(total_coeffs), target_type, verbose)
    end
end

struct FixedApproxConfig{TargetT<:AbstractFloat,T<:AbstractFloat}
    mode::AbstractMode
    target::T
    τ::T
    max_depth::Int
    min_width::T
    total_coeffs::Int
    driver_max_iter::Int
    strategy::SearchStrategy
    target_type::Type{TargetT}
    verbose::Bool
    function FixedApproxConfig(mode::AbstractMode,
        target::T,
        τ::T,
        max_depth::Integer,
        min_width::T,
        total_coeffs::Integer,
        driver_max_iter::Integer,
        strategy::SearchStrategy,
        target_type::Type{TargetT},
        verbose::Bool) where {TargetT<:AbstractFloat,T<:AbstractFloat}
        @argcheck mode isa AbsoluteMode || mode isa RelativeMode ArgumentError(
            "mode must be AbsoluteMode() or RelativeMode(), got $(typeof(mode))")
        @argcheck target > zero(T) ArgumentError("target must be > 0, got $target")
        @argcheck τ > zero(T) ArgumentError("τ must be > 0, got $τ")
        @argcheck max_depth ≥ 0 ArgumentError("max_depth must be ≥ 0, got $max_depth")
        @argcheck min_width ≥ zero(T) ArgumentError("min_width must be ≥ 0, got $min_width")
        @argcheck total_coeffs ≥ 0 ArgumentError(
            "total_coeffs must be ≥ 0, got $total_coeffs")
        @argcheck driver_max_iter ≥ 0 ArgumentError(
            "driver_max_iter must be ≥ 0, got $driver_max_iter")
        return new{TargetT,T}(mode, target, τ, Int(max_depth), min_width,
            Int(total_coeffs), Int(driver_max_iter), strategy, target_type,
            verbose)
    end
end

struct BudgetApproxConfig{TargetT<:AbstractFloat,T<:AbstractFloat}
    mode::AbstractMode
    target::T
    scheme_builder::Function
    degree_policy::Symbol
    τ::T
    max_depth::Int
    min_width::T
    total_coeffs::Int
    driver_max_iter::Int
    strategy
    target_type::Type{TargetT}
    verbose::Bool
    function BudgetApproxConfig(mode::AbstractMode,
        target::T,
        scheme_builder::Function,
        degree_policy::Symbol,
        τ::T,
        max_depth::Integer,
        min_width::T,
        total_coeffs::Integer,
        driver_max_iter::Integer,
        strategy,
        target_type::Type{TargetT},
        verbose::Bool) where {TargetT<:AbstractFloat,T<:AbstractFloat}
        @argcheck mode isa AbsoluteMode || mode isa RelativeMode ArgumentError(
            "mode must be AbsoluteMode() or RelativeMode(), got $(typeof(mode))")
        @argcheck target > zero(T) ArgumentError("target must be > 0, got $target")
        @argcheck degree_policy === :max || degree_policy === :min ||
                  degree_policy === :min_cost ArgumentError(
            "degree_policy must be :max, :min, or :min_cost; got :$degree_policy")
        @argcheck τ > zero(T) ArgumentError("τ must be > 0, got $τ")
        @argcheck max_depth ≥ 0 ArgumentError("max_depth must be ≥ 0, got $max_depth")
        @argcheck min_width ≥ zero(T) ArgumentError("min_width must be ≥ 0, got $min_width")
        @argcheck total_coeffs ≥ 0 ArgumentError(
            "total_coeffs must be ≥ 0, got $total_coeffs")
        @argcheck driver_max_iter ≥ 0 ArgumentError(
            "driver_max_iter must be ≥ 0, got $driver_max_iter")
        return new{TargetT,T}(mode, target, scheme_builder, degree_policy, τ,
            Int(max_depth), min_width, Int(total_coeffs), Int(driver_max_iter),
            strategy, target_type, verbose)
    end
end
