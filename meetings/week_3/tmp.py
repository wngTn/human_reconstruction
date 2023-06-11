import open3d as o3d
import numpy as np
from PIL import Image
import open3d as o3d
import open3d.visualization.gui as gui
import os
import sys
import json


a = ["data/Synthetic/frist_trial/Depth/r_0_0_depth_0000.png",
"data/Synthetic/first_trial/Depth/r_0_1_depth_0000.png",
"data/Synthetic/first_trial/Depth/r_0_2_depth_0000.png",
"data/Synthetic/first_trial/Depth/r_0_3_depth_0000.png",]
b = ["data/Synthetic/test/Depth/000010.png"]
c = ["data/Synthetic/first_trial/Depth/r_0_0_depth_0000.png"]
d = ["data/Synthetic/test/Depth/r_depth_0001.png"]

parameters = np.load("data/Synthetic/test/0001.npz", allow_pickle=True)
intri = parameters["intrinsic_mat"]
extri = parameters["extrinsic_mat"]

# Assuming the camera pose information is for the "10"th frame
extrin_file_path = "data/Synthetic/test/scene_gt.json"
intrin_file_path = "data/Synthetic/test/scene_camera.json"
ex_file_path = "data/Synthetic/test/camera_info.json"

# Open the JSON file
with open(extrin_file_path, "r") as json_file:
    json_data = json.load(json_file)
pose_data = json_data["10"][0]

with open(intrin_file_path, "r") as json_file:
    cam_data = json.load(json_file)

with open(ex_file_path, "r") as json_file:
    cam_data_ex = json.load(json_file)

# Extract rotation matrix and translation vector from the pose data
rotation = np.array(pose_data["cam_R_m2c"]).reshape(3, 3)
translation = np.array(pose_data["cam_t_m2c"])

# Create a 4x4 transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation
transformation_matrix[:3, 3] = translation

#print(transformation_matrix)

# Extract the "cam_K" values from the JSON data
cam_K_values = cam_data["10"]["cam_K"]

#ape data
cam_T_1_ape = cam_data_ex["cam_mat_extr"]["cam_T"]

# Create extrinsic matrix
extrinsic_matrix_ape = np.array(cam_T_1_ape)
print(extri)

#print(cam_K_values)

# Reshape the cam_K_values into a 3x3 intrinsic matrix
intrinsic_matr = np.array(cam_K_values).reshape(3, 3)

pcd = o3d.geometry.PointCloud()

#for i in range(1):
#K = intrinsic_matr
K = intri
intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(512, 512, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
extri = np.concatenate((extri, [[0, 0, 0, 1]]), axis=0)
extri[:, -1] = extri[:, -1]  # Blender is in m, Open3D in mm 

rgb = o3d.io.read_image(f"data/Synthetic/test/r_rgb.png")
#rgb = o3d.io.read_image(f"data/Synthetic/first_trial/r_0_{0}.png")
d_tmp = Image.open(d[0])
d = o3d.geometry.Image(np.asarray(d_tmp).astype(np.uint8)) # Scale depth?

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d, convert_rgb_to_intensity=False) #change bit depth, no need for trunc

_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_matrix)
# _pcd.transform(extrinsics)
#_pcd.transform(extri)
_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #pcd += _pcd
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0., 0., 0.])
test = o3d.visualization.draw_geometries([_pcd, origin])
#test = o3d.visualization.draw_geometries([_pcd])

o3d.visualization.draw_geometries([test])

smpl = o3d.io.read_triangle_mesh("meetings/week_3/0.obj") # correct scale?
smpl.compute_vertex_normals()

app = gui.Application.instance
app.initialize()
vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
vis.show_settings = True
vis.add_geometry("pcd", _pcd)
#vis.add_geometry("smpl", smpl)
vis.add_geometry("origin", o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0]))
app.add_window(vis)
app.run()
