# ImplicitAdjoints [WIP]

A work-in-progress package for *matrix-free* adjoint sensitivity analysis, integrated into the [Julia AD ecosystem](https://juliadiff.org/ChainRulesCore.jl/stable/). Initially, it contains a matrix-free sensitivity analysis specifically for the LASSO problem. Example:

Some possible future goals:

- Provide a interface for easily defining matrix-free adjoint sensitivity rules, to avoid [reinventing the wheel in each package](https://discourse.julialang.org/t/ad-step-for-iterative-solution/26404/11)
- Define rules (ideally, solver-indepenent) for common problems such as LASSO.




