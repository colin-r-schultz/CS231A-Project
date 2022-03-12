using LinearAlgebra, JuMP, Ipopt

function camera(X, K, R, T)
    P = (X * R' .+ T') * K'
    p = P[:, 1:2]' ./ P[:, 3]'
    return transpose(p)
end

function bundle_adjustment(image_points, K)
    """
    image_points - M x N x 2
    """
    M, N, _ = size(image_points)

    model = Model(Ipopt.Optimizer)

    @variable(model, R[1:M, 1:3, 1:3])
    @variable(model, T[1:M, 1:3])
    @variable(model, X[1:N, 1:3])
    @variable(model, P[1:M, 1:N, 1:3])
    @variable(model, p[1:M, 1:N, 1:2])

    @objective(model, Min, sum((image_points - p).^2))

    @constraint(model, [i = 1:M], P[i, :, :] .== (X * R[i, :, :]' .+ T[i, :]') * K')
    @NLconstraint(model, [i = 1:M, j = 1:N, k = 1:2], p[i, j, k] == P[i, j, k] / P[i, j, 3])
    @constraint(model, X[1, :] .== 0.0)
    @constraint(model, X[2, :] .== [1.0, 0.0, 0.0])
    # @constraint(model, [i = 1:M], det(R[i, :, :]) == 1)
    @NLconstraint(model, [i = 1:M], 
        R[i, 1, 1]*(R[i, 2, 2]*R[i, 3, 3]-R[i, 2, 3]*R[i, 3, 2])
        - R[i, 1, 2]*(R[i, 2, 1]*R[i, 3, 3]-R[i, 2, 3]*R[i, 3, 1])
        + R[i, 1, 3]*(R[i, 2, 1]*R[i, 3, 2]-R[i, 2, 2]*R[i, 3, 1]) == 1.0)
    @constraint(model, [i = 1:M, j = 1:3], sum(R[i, j, :].^2) == 1.0)
    @constraint(model, [i = 1:M, j = 1:3], sum(R[i, :, j].^2) == 1.0)

    set_start_value.(R, rand(M, 3, 3))
    set_start_value.(T, rand(M, 3))
    set_start_value.(X, rand(N, 3))
    set_start_value.(P, rand(M, N, 3))
    set_start_value.(p, rand(M, N, 2))
    MOI.set(model, MOI.RawParameter("max_iter"), 10000)
    optimize!(model)
    return value.(R), value.(T), value.(X)
end

