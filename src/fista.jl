# uses the Fast Iterative Soft-Thresholding Algorithm (FISTA) to
# minimize f(x) + g(x) = ½(|Ax - y|² + β|x|²) + ½α|Ψx|₁

fista(A, y, α, β, iters, tol, reg::Regularizer) = fista(A, y, α, β, iters, tol, reg::Regularizer, zeros(size(A)[2]))

# TODO: structure similarly to IterativeSolvers 
function fista(A, y, α, β, iters, tol, reg::Regularizer, xstart)
    n, p = size(A)
    
    maxeig = abs(powm(A' * A, maxiter=100)[1]) # TODO: handle number of eigiters
    L = maxeig + β # Lipschitz constant of f (strongly convex term)
    η = 1 / (L * lipschitz_scale(reg))

    x = xstart[:]
    z = x[:] 
    xold = similar(x)
    res = similar(y) 
    grad = similar(x)
    t = 1
    
    iters_done = iters
    xupdates = Float64[] 
    convdists = Float64[]
    evals = Float64[]

    Ψ = transform_op(reg)
    Fold = Inf

    for i = 1:iters

        xold .= x

        res .= mul!(res, A, z) .- y # TODO: maybe use five-arg mul! instead. (but IterativeSolvers sticks to three-arg)
        grad .= mul!(grad, A', res) .+ β .* z 

        x .= z .- η .* grad
        proximal!(x, 1/2 * η * α, reg)

        restart = dot(z .- x, x .- xold)
        restart > 0 && (t = 1)

        told = t
        t = 1/2 * (1 + √(1 + 4t^2))
        
        z .= x .+ (told - 1)/t .* (x .- xold)

        xupdate = norm(z .- xold) / norm(xold)
        append!(xupdates, xupdate)

        if xupdate < tol
            iters_done = i
            break
        end
    end

    x, (;iters=iters_done, final_tol=norm(x .- xold) / norm(x), xupdates) 
end
