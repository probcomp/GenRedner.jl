using Gen
using PyCall
using GenRedner: get_depth_renderer

plt = pyimport("matplotlib.pyplot")

const depth_renderer = get_depth_renderer(100)

function construct_simple_scene()

    vertices = vcat(

        # first triangle
        [-0.25, -0.25, 1.0]', # 1
        [0.25, -0.25, 1.0]', # 2
        [0.0, 0.25, 1.0]', # 3

        # second triangle
        [-1, -1, 1.5]', # 4
        [1, -1, 1.5]', # 5
        [0.0, 1, 1.5]', # 6

        # two triangles forming a background
        [-2, -2, 2.0]', # 7
        [-2, 2, 2.0]', # 8
        [2, 2, 2.0]', # 9
        [2, -2, 2.0]' # 10
    )

    indices = vcat(
        [1, 2, 3]', # first triangle
        [4, 5, 6]', # second triangle
        [7, 8, 9]', # background triangle 1
        [9, 10, 7]' # background triangle 2
    )

    return (vertices, indices)
end

function construct_corrupted_scene()
    (vertices, indices) = construct_simple_scene()
    corrupted_vertices = copy(vertices)
    corrupted_vertices[1,:] += [-0.1, 0.0, 0.0]
    return corrupted_vertices, indices
end

# get scene
vertices, indices = construct_simple_scene()
corrupted_vertices, corrupted_indices = construct_corrupted_scene()

# focal lengths 
fx = 48.0
fy = 48.0

# principal point
cx = 32.0
cy = 24.0

# image resolution
width = 64
height = 48

intrinsics = [fx, fy, cx, cy]
dims = [width, height]

function forward_example()

    # run forward three times and plot using matplotlib
    plt.figure(figsize=(18, 27))

    for replicate in [1, 2, 3]

        trace = simulate(depth_renderer, (vertices, indices, intrinsics, dims))
        points = get_retval(trace)


        plt.subplot(3, 2, (replicate-1) * 2 + 1)
        plt.imshow(points[:,:,3], cmap="Greys", vmin=0, vmax=2.0, origin="lower")
        plt.title("depth image")

        plt.subplot(3, 2, (replicate-1) * 2 + 2, projection="3d")
        ax = plt.gca()
        ax.scatter(points[:,:,1], points[:,:,2], points[:,:,3], s=2)
        ax.scatter([0.0], [0.0], [0.0], color="red", s=10, marker="o")
        ax.quiver([0.0], [0.0], [0.0], [1.0], [0.0], [0.0], length=0.5, color="red")
        ax.quiver([0.0], [0.0], [0.0], [0.0], [1.0], [0.0], length=0.5, color="green")
        ax.quiver([0.0], [0.0], [0.0], [0.0], [0.0], [1.0], length=0.5, color="blue")
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-0, 2])
        plt.title("point cloud")

    end

    plt.tight_layout()
    plt.savefig("ground_truth.png")

end

forward_example()

function show_gradients()

    observed_depths = depth_renderer(vertices, indices, intrinsics, dims)[:,:,3]

    @gen (grad) function loss((grad)(corrupted_vertices))
        points ~ depth_renderer(corrupted_vertices, corrupted_indices, intrinsics, dims)
        depths = points[:,:,3]
        return -sum((observed_depths .- depths) .^ 2)
    end
    trace = simulate(loss, (corrupted_vertices,))
    ((corrupted_vertices_grad,), _) = choice_gradients(trace, select(), 1.0)
    points = trace[:points]

    plt.figure(figsize=(36, 9))

    # show observed depth image
    plt.subplot(1, 4, 1)
    plt.imshow(observed_depths, cmap="Greys", vmin=0.0, vmax=2.0, origin="lower")
    plt.title("observed depth image")

    # show corrupted depth image
    plt.subplot(1, 4, 2)
    plt.imshow(points[:,:,3], cmap="Greys", vmin=0.0, vmax=2.0, origin="lower")
    plt.title("reconstructed depth image")

    # show differences
    plt.subplot(1, 4, 3)
    plt.imshow(abs.(observed_depths .- points[:,:,3]), cmap="Greys", vmin=0.0, vmax=0.1, origin="lower")
    plt.title("difference depth image")

    # show point cloud, triangles, and gradients..
    plt.subplot(1, 4, 4, projection="3d")
    ax = plt.gca()
    ax.scatter(points[:,:,1], points[:,:,2], points[:,:,3], s=2)
    ax.scatter([0.0], [0.0], [0.0], color="red", s=10, marker="o")
    ax.quiver([0.0], [0.0], [0.0], [1.0], [0.0], [0.0], length=0.5, color="red")
    ax.quiver([0.0], [0.0], [0.0], [0.0], [1.0], [0.0], length=0.5, color="green")
    ax.quiver([0.0], [0.0], [0.0], [0.0], [0.0], [1.0], length=0.5, color="blue")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])
    for t in 1:(size(indices)[1])
        x1, y1, z1 = corrupted_vertices[indices[t,1],:]
        x2, y2, z2 = corrupted_vertices[indices[t,2],:]
        x3, y3, z3 = corrupted_vertices[indices[t,3],:]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", linewidth=2)
        ax.plot([x3, x2], [y3, y2], [z3, z2], color="black")
        ax.plot([x1, x3], [y1, y3], [z1, z3], color="black")
    end

    for v in 1:(size(corrupted_vertices))[1]
        x, y, z = corrupted_vertices[v,:]
        grad = corrupted_vertices_grad[v,:]
     #   grad_dir = 0.2 * (grad / sum(grad))
        grad = grad / 50.0
        xstop, ystop, zstop = [x, y, z] + grad
        ax.plot([x, xstop], [y, ystop], [z, zstop], color="purple")
    end

    plt.tight_layout()
    plt.show()
    plt.savefig("gradients.png")
end

show_gradients()

# use MALA to fix an incorrect initialization for the first 6 vertices

struct NoiseModel <: Gen.Distribution{Matrix{Float64}} end
const noise_model = NoiseModel()

function Gen.logpdf(::NoiseModel,
        x::Matrix{Float64},
        mu::Matrix{Float64})
    diffs = (x .- mu)
    return -sum(diffs .^ 2)
end

Gen.has_argument_grads(::NoiseModel) = (true,)
Gen.has_output_grad(::NoiseModel) = false

function Gen.logpdf_grad(::NoiseModel,
        x::Matrix{Float64},
        mu::Matrix{Float64})
    diffs = x .- mu
    mu_grad = 2 * diffs
    return (nothing, mu_grad)
end

# TODO still debugging the gradients when embeddded in the model..

@gen function model()

    # prior on two foreground triangles
    foreground_vertices_list = []
    for i in 1:6
        v = ({:vertices => i} ~ mvnormal([0.0, 0.0, 1.0], [1000.0 0.0 0.0; 0.0 1000.0 0.0; 0.0 0.0 1000.0]))
        push!(foreground_vertices_list, v)
    end
    foreground_vertices_mat = transpose(hcat(foreground_vertices_list...))
  #  println(typeof(foreground_vertices_mat))
    foreground_indices = vcat(
        [1, 2, 3]', # first triangle
        [4, 5, 6]', # second triangle
    )

    # fixed background
    background_vertices = vcat(
        # two triangles forming a background
        [-2, -2, 2.0]', # 7
        [-2, 2, 2.0]', # 8
        [2, 2, 2.0]', # 9
        [2, -2, 2.0]' # 10
    )
    background_indices = vcat(
        [7, 8, 9]', # background triangle 1
        [9, 10, 7]' # background triangle 2
    )

    points ~ depth_renderer(
        vcat(foreground_vertices_mat, background_vertices),
        vcat(foreground_indices, background_indices),
        intrinsics, dims)

    depths = points[:,:,3]
    obs ~ noise_model(depths)
end


function visualize_trace(fname, trace)
    plt.clf()
    points = trace[:points]
    plt.figure(figsize=(27, 12))

    plt.subplot(1, 5, 1)
    plt.title("observed depth image")
    plt.imshow(trace[:obs], cmap="Greys", vmin=0, vmax=2.0, origin="lower")

    plt.subplot(1, 5, 2)
    plt.title("reconstructed depth image")
    plt.imshow(points[:,:,3], cmap="Greys", vmin=0.0, vmax=2.0, origin="lower")

    plt.subplot(1, 5, 3)
    plt.title("observed .- latent diff")
    plt.imshow(trace[:obs] .- points[:,:,3], cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower")

    plt.subplot(1, 5, 4)
    plt.title("gradient with respect to depth")
    plt.imshow(logpdf_grad(noise_model, trace[:obs], points[:,:,3])[2], cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower")

    plt.subplot(1, 5, 5, projection="3d")
    plt.title("latent point cloud")
    ax = plt.gca()
    ax.scatter(points[:,:,1], points[:,:,2], points[:,:,3], s=2)
    ax.scatter([0.0], [0.0], [0.0], color="red", s=10, marker="o")
    ax.quiver([0.0], [0.0], [0.0], [1.0], [0.0], [0.0], length=0.5, color="red")
    ax.quiver([0.0], [0.0], [0.0], [0.0], [1.0], [0.0], length=0.5, color="green")
    ax.quiver([0.0], [0.0], [0.0], [0.0], [0.0], [1.0], length=0.5, color="blue")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])

    for t in 1:2
        x1, y1, z1 = trace[:vertices => indices[t,1]]
        x2, y2, z2 = trace[:vertices => indices[t,2]]
        x3, y3, z3 = trace[:vertices => indices[t,3]]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", linewidth=2)
        ax.plot([x3, x2], [y3, y2], [z3, z2], color="black")
        ax.plot([x1, x3], [y1, y3], [z1, z3], color="black")
    end

    (_, choice_grads) = choice_gradients(trace, select(:vertices))
    for v in 1:6
        x, y, z = trace[:vertices => v]
        grad = choice_grads[:vertices => v]
     #   grad_dir = 0.2 * (grad / sum(grad))
        grad = grad / 10
        xstop, ystop, zstop = [x, y, z] .+ grad
        ax.plot([x, xstop], [y, ystop], [z, zstop], color="purple")
    end

    plt.tight_layout()
    plt.show()
    #    plt.savefig(fname)
end


@gen function model2((grad)(foreground_vertices_mat))

  #  println(typeof(foreground_vertices_mat))
    foreground_indices = vcat(
        [1, 2, 3]' # first triangle
    )

    # fixed background
    background_vertices = vcat(
        # second triangle
        [-1, -1, 1.5]', # 4
        [1, -1, 1.5]', # 5
        [0.0, 1, 1.5]', # 6

        # two triangles forming a background
        [-2, -2, 2.0]', # 7
        [-2, 2, 2.0]', # 8
        [2, 2, 2.0]', # 9
        [2, -2, 2.0]' # 10
    )
    background_indices = vcat(
        [4, 5, 6]', # second triangle
        [7, 8, 9]', # background triangle 1
        [9, 10, 7]' # background triangle 2
    )

    points ~ depth_renderer(
        vcat(foreground_vertices_mat, background_vertices),
        vcat(foreground_indices, background_indices),
        intrinsics, dims)

    depths = points[:,:,3]
    obs ~ noise_model(depths)
end


function visualize_model_gradients(fname,trace)

    plt.clf()
    points = trace[:points]
    plt.figure(figsize=(27, 12))

    plt.subplot(1, 5, 1)
    plt.title("observed depth image")
    plt.imshow(trace[:obs], cmap="Greys", vmin=0, vmax=2.0, origin="lower")

    plt.subplot(1, 5, 2)
    plt.title("reconstructed depth image")
    plt.imshow(points[:,:,3], cmap="Greys", vmin=0.0, vmax=2.0, origin="lower")

    plt.subplot(1, 5, 3)
    plt.title("observed .- latent diff")
    plt.imshow(trace[:obs] .- points[:,:,3], cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower")

    plt.subplot(1, 5, 4)
    plt.title("gradient with respect to depth")
    plt.imshow(logpdf_grad(noise_model, trace[:obs], points[:,:,3])[2], cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower")

    plt.subplot(1, 5, 5, projection="3d")
    plt.title("latent point cloud")
    ax = plt.gca()
    ax.scatter(points[:,:,1], points[:,:,2], points[:,:,3], s=2)
    ax.scatter([0.0], [0.0], [0.0], color="red", s=10, marker="o")
    ax.quiver([0.0], [0.0], [0.0], [1.0], [0.0], [0.0], length=0.5, color="red")
    ax.quiver([0.0], [0.0], [0.0], [0.0], [1.0], [0.0], length=0.5, color="green")
    ax.quiver([0.0], [0.0], [0.0], [0.0], [0.0], [1.0], length=0.5, color="blue")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])

    foreground_vertices_mat = get_args(trace)[1]

    for t in 1:1
        x1, y1, z1 = foreground_vertices_mat[indices[t,1],:]
        x2, y2, z2 = foreground_vertices_mat[indices[t,2],:]
        x3, y3, z3 = foreground_vertices_mat[indices[t,3],:]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", linewidth=2)
        ax.plot([x3, x2], [y3, y2], [z3, z2], color="black")
        ax.plot([x1, x3], [y1, y3], [z1, z3], color="black")
    end

    ((foreground_vertices_mat_grad,), _) = choice_gradients(trace, select())
    for v in 1:3
        x, y, z = foreground_vertices_mat[v,:]
        grad = foreground_vertices_mat_grad[v,:]
     #   grad_dir = 0.2 * (grad / sum(grad))
        grad = grad / 10
        xstop, ystop, zstop = [x, y, z] .+ grad
        ax.plot([x, xstop], [y, ystop], [z, zstop], color="purple")
    end

    plt.tight_layout()
    plt.savefig(fname)

end

using Printf: @sprintf

function gradient_descent_demo()

    observed = depth_renderer(vertices, indices, intrinsics, dims)[:,:,3]
    constraints = choicemap()
    constraints[:obs] = observed
    foreground_vertices_mat = vertices[1:3,:]
    foreground_vertices_mat[1,1] -= 0.15
    trace, = generate(model2, (foreground_vertices_mat,), constraints)
    visualize_model_gradients(@sprintf("grad_step_%03d.png", 0), trace)
    for iter in 1:200
        println(iter)
        ((foreground_vertices_mat_grad,), _) = choice_gradients(trace, select())
        foreground_vertices_mat += 0.00001 * foreground_vertices_mat_grad
        trace, = update(trace, (foreground_vertices_mat,), (UnknownChange(),), choicemap())
        visualize_model_gradients(@sprintf("grad_step_%03d.png", iter), trace)
    end
end

#gradient_descent_demo()

# TODO implement something that lets you check gradients using finite-differencing...
# but it won't help here, becuasse everything's stochastic..


#  @gen function stochastic_gradient_langevin_move(trace, vertices, eps)
    
#      # unbiased estimate of gradient
#      _, grads = choice_gradients(trace, select((:vertices => v for v in vertices)...))
#      for v in vertices
#          mu = trace[:vertices => v] .+ (eps/2) * grads[:vertices => v]
#          {:vertices => v} ~ mvnormal(mu, [eps^2 0.0 0.0; 0.0 eps^2 0.0; 0.0 0.0 eps^2])
#      end
#  end

using Printf: @sprintf
using Random: seed!

function do_inference()
    seed!(1) # note: this does not affect Redner's random seed

    num_latent_vertices = 6

    observed = depth_renderer(vertices, indices, intrinsics, dims)[:,:,3]
    constraints = choicemap()
    constraints[:obs] = observed

    # initialize to foreground vertices that are slightly wrong
    for i in 1:3
        constraints[:vertices => i] = corrupted_vertices[i,:]#mvnormal(vertices[i,:], 0.01 * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    end
    for i in 4:6
        constraints[:vertices => i] = vertices[i,:]
    end
    trace, = generate(model, (), constraints)
    frame = 0

    fname = @sprintf("iter_%03d.png", frame); frame += 1
    visualize_trace(fname, trace)
    iters = 500
    for iter in 1:iters
        println("iter: $iter")
        #  constraints, _ = propose(stochastic_gradient_langevin_move, (trace, [1, 2, 3], 1e-3))

        # stochastic gradient ascent on MAP
        vertices = [1, 2, 3]
        _, grads = choice_gradients(trace, select((:vertices => v for v in vertices)...))
        constraints = choicemap()
        for v in vertices
            constraints[:vertices => v] = trace[:vertices => v] .+ (0.00001) * grads[:vertices => v]
        end
    
        trace, _ = update(trace, (), (), constraints)
        if iter % 1 == 0
            fname = @sprintf("iter_%03d.png", frame); frame += 1
            visualize_trace(fname, trace)
        end
        plt.clf()

     #   _, choice_grads = choice_gradients(trace, select(:vertices))
      #  constraints = choicemap()
     #   step_size = 1e-3
     #   for i in 1:num_latent_vertices
     #       constraints[:vertices => i] = trace[:vertices => i] .+ (choice_grads[:vertices => i] * step_size)
      #  end
     #   trace, _ = update(trace, (), (), constraints)

    #    trace = map_optimize(trace, select(:vertices => 1, :vertices => 2, :vertices => 3), verbose=true, max_step_size=1e-4, min_step_size=1e-10)
       # trace = map_optimize(trace, select(:vertices => 4, :vertices => 5, :vertices => 6), verbose=true, max_step_size=1e-4, min_step_size=1e-10)

        #  for i in 1:5
      #      trace, = mala(trace, select(:vertices), 0.1)
      #      trace, = mala(trace, select(:vertices), 0.01)
       #     trace, = mala(trace, select(:vertices), 0.001)
        #    trace, acc1 = mala(trace, select(:vertices), 0.0001)
         #   trace, acc2 = mala(trace, select(:vertices), 0.00001)
#         trace, acc3 = mala(trace, select(:vertices), 0.0001)
    #        trace, acc3 = hmc(trace, select(:vertices), L=10, eps=0.001)

       # end
#       println("iter; $iter of $iters, acc1: $acc1, acc2: $acc2, acc3: $acc3")
       println("iter; $iter of $iters")#,  acc3: $acc3")

   #     fname = @sprintf("iter_%03d.png", iter)
     #   visualize_trace(fname, trace)
    end
    fname = @sprintf("iter_%03d.png", frame); frame += 1
    visualize_trace(fname, trace)
end

#do_inference()