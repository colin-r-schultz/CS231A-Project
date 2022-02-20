using LinearAlgebra, JuMP, Ipopt

function bundle_adjustment(image_points, K)
    """
    image_points - M x N x 2
    """
    M, N, _ = size(image_points)

    model = Model(Ipopt.Optimizer)

    @variable(model, R[1:M, 1:3, 1:3])
    @variable(model, T[1:M, 1:3])
    @variable(model, X[1:N, 1:3])

    @expression(model, X_projected_homogenous[i = 1:M],
        (X * R[i, :, :]' .+ T[i]') * K'
    )
    @NLexpression(model, X_projected[i = 1:M],
        X_projected_homogenous[i, :, 1:2] ./ X_projected_homogenous[i, :, 3]
    )

    @objective(model, Min, sum((image_points - X_projected).^2))

    @constraint(model, X[1, :] .== 0)
    @constraint(model, X[2, :] == [1 0 0])
    @NLconstraint(model, [i = 1:M], det(R[i]) == 1)

    optimize!(model)
    return value.(R), value.(T), value.(X)
end

