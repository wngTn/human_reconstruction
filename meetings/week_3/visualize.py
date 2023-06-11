import open3d as o3d
import numpy as np
from PIL import Image
import open3d as o3d
import open3d.visualization.gui as gui
import os
import sys


a = ["data/Synthetic/second_trial/Depth/r_0_0_depth_0000.png",
"data/Synthetic/second_trial/Depth/r_0_1_depth_0000.png",
"data/Synthetic/second_trial/Depth/r_0_2_depth_0000.png",
"data/Synthetic/second_trial/Depth/r_0_3_depth_0000.png",]

parameters = np.load("data/Synthetic/second_trial/output_data.npz", allow_pickle=True)
intri = parameters["intrinsic_mat"]
extri = parameters["extrinsic_mat"].item(0)

pcd = o3d.geometry.PointCloud()

for i in range(4):
    K = intri
    intrinsics = o3d.camera.PinholeCameraIntrinsic(512, 512, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    #extrinsics = np.concatenate((extri[f"cam_T_{i}"], [[0, 0, 0, 1]]), axis=0)
    #extrinsics[:, -1] *= 1000 # Blender is in m, Open3D in mm 
    extrinsics = np.array(extri[f"cam_T_{i}"])

    extrinsics[:, -1] *= 0.001 # Blender is in m, Open3D in mm 

    rgb = o3d.io.read_image(f"data/Synthetic/second_trial/r_0_{i}_rgb.png")
    d_tmp = Image.open(a[i])
    d = o3d.geometry.Image(np.asarray(d_tmp).astype(np.uint8)) # Scale depth?

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d, convert_rgb_to_intensity=False) #change bit depth, no need for trunc

    _pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    _pcd.transform(extrinsics)
    
    pcd += _pcd


smpl = o3d.io.read_triangle_mesh("meetings/week_3/0.obj") # correct scale?
smpl.compute_vertex_normals()

app = gui.Application.instance
app.initialize()
vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
vis.show_settings = True
vis.add_geometry("pcd", pcd)
vis.add_geometry("smpl", smpl)
vis.add_geometry("origin", o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0]))
app.add_window(vis)
app.run()
