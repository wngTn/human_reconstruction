import numpy as np
from pathlib import Path
import taichi as ti
import taichi_render_gpu.taichi_three.transform as transform
import taichi_three as t3

prefix = Path("meetings/week_3/")

parameters = np.load(prefix.joinpath("0536.npz"), allow_pickle=True)
intri = parameters["intrinsic_mat"]
extri = parameters["extrinsic_mat"].item(0)



# to make tina actually display things, we need at least three things:
#
# 1. Scene - the top structure that manages all resources in the scene
scene = t3.Scene()

# 2. Model - the model to be displayed
#
# here we use `tina.MeshModel` which can load models from OBJ format files
model = t3.readobj('meetings/week_3/smplx_0.obj')
# and, don't forget to add the model into the scene so that it gets displayed
scene.add_model(model)
camera = transform.Camera(res=(512, 512))

# render scene to image
scene.render()
ti.imwrite((camera.img.to_numpy() + 1)/2, "test.jpg")