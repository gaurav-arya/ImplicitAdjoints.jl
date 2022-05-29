using ImplicitAdjoints
using Test
using Random: randperm
using Statistics: mean
using Zygote
using FiniteDifferences

@testset "ImplicitAdjoints.jl" begin
    n, p = 5, 10
    S = 2
    α = 0.1
    β = 0.01
    G = randn(n, p)
    reg = L1(p)
    u = zeros(p)
    u[randperm(p)[1:S]] .= 1
    η = randn(n)

    function f(G, α, β)
        y = G * u
        y += 0.05 * mean(abs.(y)) * η
        uest, info = genlasso(G, y, α, β, 1000, 1e-12, reg)
        return sum((u - uest).^2)
    end

    ∂f = gradient(f, G, α, β)
    ∂f_fdm = grad(central_fdm(3, 1, max_range=1e-4), f, G, α, β)

    @show ∂f
    @show ∂f_fdm
    @test all(isapprox(∂f[i], ∂f_fdm[i], rtol=1e-3) for i in 1:3) # check gradients ∂f/∂G, ∂f/∂α, ∂f/∂β
end
