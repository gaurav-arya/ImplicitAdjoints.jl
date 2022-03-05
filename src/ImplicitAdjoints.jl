module ImplicitAdjoints

    export genlasso, L1, TV

    using ChainRulesCore
    import ChainRulesCore.rrule
    using LinearAlgebra
    import LinearAlgebra: mul! 
    using IterativeSolvers: cg, powm
    using LinearMaps
    using LinearMaps: CompositeMap, TransposeMap, BlockMap, AdjointMap
    import LinearMaps._unsafe_mul!
    using Zygote: @adjoint # TODO: don't want this

    include("linear_maps.jl")
    include("regularizers.jl")
    include("fista.jl")
    include("lasso.jl")

end
