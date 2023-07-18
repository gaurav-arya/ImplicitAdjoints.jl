# nb: nothing in this file should really be in this package, but it's necessary for now for things to work

# TODO: figure out how to use rrule instead
@adjoint function Base.adjoint(A::LinearMap)
      pullback(Ā) = (Ā.lmap, )
      return A', pullback
end	

# TODO: change to clunkier but more Julian subtyping
function Identity(size)
    LinearMap(x -> x, x -> x, size, size)
end

## rrules for LinearMaps

# CompositeMap's

rrule(::typeof(Base.adjoint), A::LinearMap) = A', (∂A -> (NoTangent(), ∂A.lmap))

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(*), A::CompositeMap, x::AbstractVector)
    function safemul(A::CompositeMap, x::AbstractVector)
        n = length(A.maps)
        for i in 1:n
            x = A.maps[i] * x
        end
        x
    end
    rrule_via_ad(config, safemul, A, x)
end

# TODO: generalize to adjoint using metaprogramming, as in https://github.com/Jutho/LinearMaps.jl/blob/master/src/blockmap.jl
function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(*), At::TransposeMap{<:Any, <:CompositeMap}, x::AbstractVector)
    function safemul(At::TransposeMap{<:Any, <:CompositeMap}, x::AbstractVector)
        A = At.lmap
        n = length(A.maps)
        for i in n:-1:1
            x = A.maps[i]' * x
        end
        x
    end
    rrule_via_ad(config, safemul, A, x)
end

# TODO: investigate list comprehensions efficiency in backwards pass
# TODO: Ideally, I'd like to implement A' * ∂y rule for all LinearMap's; 
# but I don't see how to avoid the undesirable default of 
# having not implemented w.r.t. the map itself
function rrule(::typeof(*), A::Union{BlockMap, TransposeMap{<:Any, <:BlockMap}}, x::AbstractVector)
    pullback(∂y) = NoTangent(), NoTangent(), A' * ∂y
    # @not_implemented("Derivative of BlockMap multiplication w.r.t. the map itself is not implemented") # TODO: use this, while avoiding error when Zygote backpropagates not implemented for no reason
    A * x, pullback
end
