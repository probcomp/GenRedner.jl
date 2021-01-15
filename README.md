# GenRedner.jl
Gen.jl wrapper for the [Redner differentiable renderer](https://github.com/BachiLi/redner)


## Depth rendering generative function

The GenRedner module currently exports a one function:
```julia
depth_renderer = GenRedner.get_depth_renderer(num_samples)
```
where `num_samples` is the number of samples that the renderer will user internally for rendering and for its internal gradient estimation.
The returned value is a [generative function](https://www.gen.dev/dev/ref/gfi/#Generative-Functions-1) of concrete type `GenPyTorch.TorchGenerativeFunction`.

The returned generative function has the following type signature:

Arguments:

- `vertices` is a `n` x 3 matrix of floating point values, where `n` is the number of vertices in the mesh. Each row gives the x, y, and z coordinates of a mesh vertex in the camera's 3D coordinate frame (see below for this frame).

- `indices` is a `n` x 3 matrix of integers, where `n` is the number of triangle faces in the mesh. Each row gives the indices of the three vertices that comprise the face. The indices of vertices face start from 1.

- `instrinsics` is a length-4 vector of floating point values of the form `[fx, fy, cx, cy]` where `fx` is the horizontal focal length of the camera model, `fy` is the vertical focal length, and `cx` and `cy` give the principle points.

- `dims` is a length-2 vector of integers of the form `[width, height]` where `width` is the number of rows in the resulting depth image and `height` is the number of columns.

The return value is a 3D array of floating point values `points` with dimension `width` x `height` x 3, where `points[:,:,1]` contains the x-coordinates in the camera's 3D coordinate frame, `points[:,:,2]` contains the y-coordinates, and `points[:,:,3]` contains the z-coordinates (i.e. the depth image).

## Forward rendering and gradients

Like all generative functions, the returned depth rendering generative function supports both forward execution (via `Gen.simulate`) and computing gradients with respect to `vertices` and `intrinsics` (via `Gen.choice_gradients` or `Gen.accumulate_param_gradients!`). Note that this generative function has no trainable parameters of its own. The generative function is intended to be called from within a generative model of depth images. The generative funciton internally calls Redner for both forward rendering and for gradient computation.

## Coordinate frame

The input vertices and the output point cloud both are in the camera's 3D coordinate frame.
This coordinate frame is the same as the [3D camera coordinate frame used by OpenCV](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#details) (the black coorinate frame in the image below):

![3 D camera model schematic from OpenCV](https://docs.opencv.org/master/pinhole_camera_model.png)

## Other outputs

The Redner differentiable renderer produces many more output channels, including various types of RGB rendering (including photorealistic rendering), and object masks.
This wrapper will be extended with additional generative functions that wrap these other rendering capabilities in the future.
