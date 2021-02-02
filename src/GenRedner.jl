module GenRedner

using GenPyTorch; TorchGenerativeFunction, TorchArg
using PyCall

# coordinate frame: https://learnopencv.com/wp-content/uploads/2020/02/world-camera-image-coordinates.png

function get_depth_renderer(;num_samples=4, print_timing=false)

py"""
import pyredner
import torch
import torch.nn as nn

pyredner.set_print_timing($(print_timing))

def construct_intrinsic_mat(intrinsics_vector, width, height):
    # see https://github.com/BachiLi/redner/blob/96d3e27cdd39129b1dc00e0271d3e27c0c5cec8f/pyredner/camera.py#L39-L50
    # for the specification of the intrinsics matrix that this function returns
    fx = intrinsics_vector[0:1]
    fy = intrinsics_vector[1:2]
    cx = intrinsics_vector[2:3]
    cy = intrinsics_vector[3:4]
    a = fx * 2 / width
    b = -fy * 2 / width
    x0 = (2 * cx / width) - 1
    y0 = -(2 * cy / width) + (height / width)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    skew = zero
    row0 = torch.cat([a, skew, x0])
    row1 = torch.cat([ zero,   b, y0])
    row2 = torch.cat([ zero,    zero,  one])
    intrinsic_mat = torch.stack([row0, row1, row2]).contiguous()
    return intrinsic_mat

class RednerDepthRenderer(torch.nn.Module):
    
    def __init__(self, num_samples):
        super(RednerDepthRenderer, self).__init__()
        self.num_samples = num_samples

    def forward(self, vertices, indices, intrinsics, dims):

        # vertices is num_vertices x 3 tensor of floats
        # indices is num_triangles x 3 tensor of ints, and starts from one
        
        # shift indices down by one, so they start from zero
        indices = indices - 1

        (width, height) = dims
        intrinsic_mat = construct_intrinsic_mat(intrinsics, width, height)

        # the mesh vertices must already be in the camera's 3D coordinate frame
        cam_to_world = torch.eye(4)

        camera = pyredner.Camera(
            resolution=(height, width),
            cam_to_world=cam_to_world,
            clip_near=0.0001,
            intrinsic_mat=intrinsic_mat)

        objects = [pyredner.Object(vertices.contiguous(), indices.contiguous(), pyredner.Material())]

        scene = pyredner.Scene(camera=camera, objects=objects)

        # return a height x width x 3 tensor of floats
        # the channels give x, y, and z coordinates in the 3D camera frame, respectively
        # so the third channel points[:,:,2] is the depth
        points = pyredner.render_g_buffer(scene, [pyredner.channels.position], self.num_samples)
        assert tuple(points.size()) == (height, width, 3)
        return points
"""

    return TorchGenerativeFunction(
    py"RednerDepthRenderer($(num_samples))", [
        TorchArg(true, py"torch.float"), # vertices
        TorchArg(false, py"torch.int"), # indices
        TorchArg(true, py"torch.float"), # intrinsics
        TorchArg(false, py"torch.int"), # dims
    ], 1)

end

export get_depth_renderer

end # module
