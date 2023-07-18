module ImplicitAdjoints

    export genlasso, fake_genlasso, L1, TV
    export FISTA
    export run_reconstruction_test
    export Identity
    export proximal, transform_op, projection_op, support, L1, TV, L1Project, TVProject, Gradient

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
