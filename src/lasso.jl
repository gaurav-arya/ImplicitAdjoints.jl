"""
    genlasso(G, y, α, β, iters, tol, reg::Regularizer)

    Minimize |Gu - y|² + α|Ψu|₁ + β|u|²

    # Arguments
    - `G`: measurement matrix.
    - `y`: measured vector.
    - `α`: scaling of ℓ₁ term |Ψu|₁ 
    - `β`: scaling of ℓ₂ term |u|²
    - `reg::Regularizer`: regularizer interface, which defines Ψ and
        other necessary operators for the forward + backward solve.

    # Output
    - 'u': recovered vector. 
    - 'info': solver info.
"""
genlasso(G, y, α, β, iters, tol, reg::Regularizer) = fista(G, y, α, β, iters, tol, reg)

function fake_genlasso(u, G, y, α, β, iters, tol, reg::Regularizer)
    
    supp = support(u, reg)
    P = projection_op(supp, reg)
    Ψ = transform_op(reg)

    # set up and solve linear system
    A = P * G' * G * P' + β * Identity(size(P)[1])
    b = P * G' * y - 1 / 2 * α * (P * Ψ' * sign.(Ψ * u))

    x, ch = cg(A, b; maxiter=iters ÷ 10, log=true)

    P' * x
end

# Linear system satisfied by solution u is A*(P*u) = b.
function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(genlasso), G, y, α, β, iters, tol, reg::Regularizer)

    u, info = genlasso(G, y, α, β, iters, tol, reg)

    # Takes in dL/du and returns dL/d(args).
    function pullback(∂out)

        ## Stuff that sets up the linear system, logic done by user (makes A and P for the original args, as well as functions make_APu and make_b that are locally valid near the original args).
        
        supp = support(u, reg)
        P = projection_op(supp, reg)
        A = P * G' * G * P' + β * Identity(size(P)[1])
        Ψ = transform_op(reg)
        # CompositeMaps may / may not be differentiable, so use right associativity to be safe
        make_APu(G, y, α, β) = P * (G' * (G * u)) + β * (P * u)
        make_b(G, y, α, β) = P * (G' * y) - 1 / 2 * α * (P * (Ψ' * sign.(Ψ * u)))

        ## Stuff that does the adjoint method (could automate?)

        # We define ϵ = b - APu.
        # λ ⋅ dϵ/d(args) gives the backpropagated adjoint variable.
        ϵ(args...) = make_b(args...) - make_APu(args...)

        ∂u = ∂out[1]
        ∂u_supp = P * ∂u
        λ = cg(A, ∂u_supp; maxiter=iters)

        args_pullback = rrule_via_ad(config, (args...) -> ϵ(args...), G, y, α, β)[2]
        ∂args = args_pullback(λ)

        x = ∂args..., NoTangent(), NoTangent(), NoTangent()
        x
    end
    
    (u, info), pullback
end
