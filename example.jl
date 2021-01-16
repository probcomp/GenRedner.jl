using Gen
using PyCall
using GenRedner: get_depth_renderer

plt = pyimport("matplotlib.pyplot")

const depth_renderer = get_depth_renderer(num_samples=2, print_timing=true)

function construct_simple_scene()

    vertices = vcat(

        # first foreground triangle (closest to camera)
        [-0.25, -0.25, 1.0]', # 1
        [0.25, -0.25, 1.0]', # 2
        [0.0, 0.25, 1.0]', # 3

        # second foreground triangle
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

    # offset one vertex of the front triangle by 0.1 units
    corrupted_vertices[1,:] += [-0.1, 0.0, 0.0]
    return corrupted_vertices, indices
end

# get scene and corrupted scene
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

    # run forward three times (to show variability) and plot using matplotlib
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
    plt.savefig("forward.png")

end

forward_example()

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

@gen function model()

    # prior on two foreground triangles
    foreground_vertices_list = []
    for i in 1:6
        v = ({:vertices => i} ~ mvnormal([0.0, 0.0, 1.0], [1000.0 0.0 0.0; 0.0 1000.0 0.0; 0.0 0.0 1000.0]))
        push!(foreground_vertices_list, v)
    end
    foreground_vertices_mat = copy(transpose(hcat(foreground_vertices_list...)))
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
    plt.figure(figsize=(27, 8))

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

    (_, _, choice_grads) = choice_gradients(trace, select(:vertices))
    for v in 1:6
        x, y, z = trace[:vertices => v]
        grad = choice_grads[:vertices => v]
        grad = grad / 10
        xstop, ystop, zstop = [x, y, z] .+ grad
        ax.plot([x, xstop], [y, ystop], [z, zstop], color="purple")
    end

    plt.tight_layout()
    plt.savefig(fname)
end

using Printf: @sprintf
using Random: seed!

function do_inference()
    seed!(1) # note: this does not affect Redner's random seed

    num_latent_vertices = 6

    observed = depth_renderer(vertices, indices, intrinsics, dims)[:,:,3]
    constraints = choicemap()
    constraints[:obs] = observed

    # initialize to foreground vertices that are slightly wrong
    for i in 1:6
        constraints[:vertices => i] = corrupted_vertices[i,:]
    end
    trace, = generate(model, (), constraints)
    frame = 0

    fname = @sprintf("iter_%03d.png", frame); frame += 1
    visualize_trace(fname, trace)
    iters = 500
    for iter in 1:iters
        println("iter: $iter")

        # stochastic gradient ascent for MAP estimation
        (_, _, grads) = choice_gradients(trace, select((:vertices => v for v in 1:num_latent_vertices)...))
        constraints = choicemap()
        for v in 1:num_latent_vertices
            constraints[:vertices => v] = trace[:vertices => v] .+ (0.000001) * grads[:vertices => v]
        end
    
        trace, _ = update(trace, (), (), constraints)
        if iter % 10 == 0
            fname = @sprintf("iter_%03d.png", frame); frame += 1
            visualize_trace(fname, trace)
        end
        plt.clf()

       println("iter; $iter of $iters")#,  acc3: $acc3")

    end
    fname = @sprintf("iter_%03d.png", frame); frame += 1
    visualize_trace(fname, trace)
end

do_inference()
