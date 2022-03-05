
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
genlasso(G, y, α, β, iters, tol, reg::Regularizer) = FISTA(G, y, α, β, iters, tol, reg)

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

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(genlasso), G, y, α, β, iters, tol, reg::Regularizer)

    u, info = genlasso(G, y, α, β, iters, tol, reg)

    # ensure gradient accuracy
    #ufake = fake_reconstruct(u, G, y, α, β, iters, tol, reg)
    #@assert sum((u - ufake).^2) / sum(u.^2) < 1e-3

    # Takes in dL/du and returns dL/d(args).
    function pullback(∂out)

        supp = support(u, reg)
        P = projection_op(supp, reg)
        Ψ = transform_op(reg)

        # We define ϵ = b - APu.
        # λ ⋅ dϵ/d(args) gives the backpropagated adjoint variable.
        function ϵ(G, y, α, β)
            # CompositeMaps may / may not be differentiable, so use right associativity to be safe
            APu = P * (G' * (G * u)) + β * (P * u)
            b = P * (G' * y) - 1 / 2 * α * (P * (Ψ' * sign.(Ψ * u)))
            b - APu
        end

        A = P * G' * G * P' + β * Identity(size(P)[1])

        args_pullback = rrule_via_ad(config, (G, y, α, β) -> ϵ(G, y, α, β), G, y, α, β)[2]

        ∂u = ∂out[1]
        ∂u_supp = P * ∂u
        λ = cg(A, ∂u_supp; maxiter=iters)
        ∂args = args_pullback(λ)
        x = ∂args..., NoTangent(), NoTangent(), NoTangent()
        x
    end
    
    (u, info), pullback
end
