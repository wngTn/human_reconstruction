import open3d as o3d
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

prefix = Path("meetings/week_3/")

img_files = sorted(prefix.glob("*.png"))

parameters = np.load(prefix.joinpath("0536.npz"), allow_pickle=True)
intri = parameters["intrinsic_mat"]
extri = parameters["extrinsic_mat"].item(0)

# loading obj file
mesh = o3d.io.read_triangle_mesh(str(prefix.joinpath("smplx_0.obj")))
points_3d = np.asarray(mesh.vertices)

# rotate over x-axis by d degrees
d = 90
R_obj = np.array([
    [1, 0, 0],
    [0, np.cos(np.deg2rad(d)), -np.sin(np.deg2rad(d))],
    [0, np.sin(np.deg2rad(d)), np.cos(np.deg2rad(d))]
])
points_3d = np.matmul(R_obj, points_3d.T).T

# add origin and axis points
origin_and_axis_points = np.array([
    [0, 0, 0],  # origin
    [1, 0, 0],  # x-axis
    [0, 1, 0],  # y-axis
    [0, 0, 1],  # z-axis
])
origin_and_axis_points = origin_and_axis_points * 0.1  # scale down
points_3d = np.concatenate([points_3d, origin_and_axis_points])

K = intri
for i in tqdm(range(4)):
    # 4 x 4 word -> camera matrix
    world_2_camera = extri[f"cam_T_{i}"]
    world_2_camera = np.concatenate((world_2_camera, np.array([[0, 0, 0, 1]])), axis=0)
    R = world_2_camera[:3, :3]
    T = world_2_camera[:3, 3]

    # 3d points -> 2d points
    points_2d = np.matmul(K, np.matmul(R, points_3d.T) + T[:, np.newaxis])
    points_2d = points_2d[:2, :] / points_2d[2, :]
    points_2d = points_2d.T
    # clamp points2d to 0 and 512
    points_2d = np.clip(points_2d, 0, 511)

    # rendering the points onto the image and save it
    img = Image.open(img_files[i])
    img = np.array(img)
    for j, point in enumerate(points_2d):
        if j >= len(points_2d) - 4:  # the last 4 points are the origin and axes
            if j == len(points_2d) - 4:  # origin
                img[int(point[1]), int(point[0])] = [255, 255, 255, 255]  # white
            elif j == len(points_2d) - 3:  # x-axis
                img[int(point[1]), int(point[0])] = [255, 0, 0, 255]  # red
            elif j == len(points_2d) - 2:  # y-axis
                img[int(point[1]), int(point[0])] = [0, 255, 0, 255]  # green
            elif j == len(points_2d) - 1:  # z-axis
                img[int(point[1]), int(point[0])] = [0, 0, 255, 255]  # blue
        else:
            img[int(point[1]), int(point[0])] = [255, 0, 0, 255]  # red for all other points

    img = Image.fromarray(img)
    img.save(prefix.joinpath(f"rendered_origin_{i}.png"))
