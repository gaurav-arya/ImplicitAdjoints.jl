# ImplicitAdjoints [WIP]

A work-in-progress package for *matrix-free* adjoint sensitivity analysis, integrated into the [Julia AD ecosystem](https://juliadiff.org/ChainRulesCore.jl/stable/). Initially, it contains a sensitivity analysis specifically for the generalized Lasso problem. Example:

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

Right now, the code contains the precise amount necessary for a matrix-free sensitivity analysis of Lasso to work (in the above example, `G` can be replaced by an arbitrary linear operator). `G` and `G'` are assumed to support the non-allocating `mul!` operation, which is used in the iterative solvers, as well as the out-of-place `*` operation, which is used in sensitivity analysis and assumed to be compatible with standard AD. Both `L1` and `TV` regularizer types are implemented. Some possible future goals:

- Provide a interface for easily defining matrix-free adjoint sensitivity rules, to avoid [reinventing the wheel in each package](https://discourse.julialang.org/t/ad-step-for-iterative-solution/26404/4).
- Define rules for common problems, including the Lasso.

The [jaxopt](https://github.com/google/jaxopt) Python library may be worth looking at. 

Notes on how this package relates to the ecosystem:

- Part of this package performs type piracy on `LinearMaps` to get things to work with AD. That stuff should be polished and moved to `LinearMaps`. 
- Something missing from this package, but is very useful in practice when incorporating `genlasso` into pipelines, is good tooling for defining matrix-free operators with `LinearMaps` (e.g. for defining the measurement matrix of an imaging system). This might be best as a separate package.
- Ideally rules should be solver independent; a forward implementation of `genlasso` probably shouldn't be here, and the regularizer interface should be simplified accordingly.
- Need to think about how this package will interact with `IterativeSolvers`.




