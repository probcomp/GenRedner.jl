using Gen
using PyCall
using GenRedner: get_depth_renderer

plt = pyimport("matplotlib.pyplot")

const depth_renderer = get_depth_renderer()

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
        [0, 1, 2]', # NOTE needs to start from zero..
        [3, 4, 5]',
        [6, 7, 8]',
        [8, 9, 6]'
    )


    indices = vcat(
        [1, 2, 3]', # first triangle
        [4, 5, 6]', # second triangle
        [7, 8, 9]', # background triangle 1
        [9, 10, 7]' # background triangle 2
    )

    return (vertices, indices)
end

vertices, indices = construct_simple_scene()

function test_simulate()

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

    trace = simulate(depth_renderer, (vertices, indices, intrinsics, dims))
    points = get_retval(trace)

    plt.figure(figsize=(18, 9))

    plt.subplot(1, 2, 1)
    plt.imshow(points[:,:,3], cmap="Greys", vmin=0, vmax=2.0, origin="lower")
    plt.title("depth image")

    plt.subplot(1, 2, 2, projection="3d")
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

    plt.tight_layout()
    plt.savefig("example.png")
end

test_simulate()
