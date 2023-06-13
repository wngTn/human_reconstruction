import open3d as o3d
import numpy as np
from PIL import Image
import open3d as o3d
import open3d.visualization.gui as gui
import os


a = ["data/Synthetic/first_trial/Depth/r_0_0_depth_0000.png",
"data/Synthetic/first_trial/Depth/r_0_1_depth_0000.png",
"data/Synthetic/first_trial/Depth/r_0_2_depth_0000.png",
"data/Synthetic/first_trial/Depth/r_0_3_depth_0000.png",]

parameters = np.load("data/Synthetic/first_trial/0536.npz", allow_pickle=True)
intri = parameters["intrinsic_mat"]
extri = parameters["extrinsic_mat"].item(0)

pcd = o3d.geometry.PointCloud()

for i in range(4):
    # 3x3 intrinsics matrix
    K = intri
    # intrinsics = o3d.camera.PinholeCameraIntrinsic(512, 512, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    # 4 x 4 word -> camera matrix
    world_2_camera = extri[f"cam_T_{i}"]
    world_2_camera = np.concatenate((world_2_camera, np.array([[0, 0, 0, 1]])), axis=0)
    d_tmp = Image.open(a[i])
    # d = o3d.geometry.Image(np.asarray(d_tmp)[:, :, 0].astype(np.uint16))
    # create 3d point cloud
    depth_data = np.asarray(d_tmp)[:, :, 0]
    # get camera to world
    camera_2_world = np.linalg.inv(world_2_camera)
    R = camera_2_world[:3, :3]
    T = camera_2_world[:3, 3]
    pcd_data = []
    import itertools
    for x, y in itertools.product(range(512), range(512)):
        if depth_data[x, y] == 0:
            continue
        px_to_depth_cam = np.dot(np.linalg.inv(K), np.array([x, y, 1])) * depth_data[x, y] / 100
        world_point = (R.T @ px_to_depth_cam,)[:3] + (-R.T @ T).T
        pcd_data.append(world_point[0])

    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(np.array(pcd_data))
    pcd += _pcd

        


smpl = o3d.io.read_triangle_mesh("meetings/week_3/0.obj")
smpl.compute_vertex_normals()

app = gui.Application.instance
app.initialize()
vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
vis.show_settings = True
vis.add_geometry("pcd", pcd)
vis.add_geometry("smpl", smpl)
vis.add_geometry("origin", o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0]))
app.add_window(vis)
app.run()
