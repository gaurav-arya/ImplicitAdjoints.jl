# ImplicitAdjoints [WIP]

Contains a sensitivity analysis for the generalized Lasso problem, see example below. Eventually the adjoint method should become a macro / utility function for helping define an rrule, and the sensitivity analysis here can be simplified to only form the linear operators required by that.

```julia
using Random: randperm
using Statistics: mean
using Zygote
using ImplicitAdjoints
using FiniteDifferences

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

println(all(isapprox(∂f[i], ∂f_fdm[i], rtol=1e-3) for i in 1:3)) # check gradients ∂f/∂G, ∂f/∂α, ∂f/∂β
# true
```
