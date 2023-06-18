# # from PIL import Image
# # import numpy as np
# # import math

# # # Load the OBJ file
# # with open('dataset/MultiHuman/single/obj/401/401.obj', 'r') as f:
# #     lines = f.readlines()

# # # Collect the colors and update the OBJ lines
# # colors = []
# # new_lines = []
# # for line in lines:
# #     if line.startswith('v '):
# #         parts = line.split()
# #         x, y, z = map(float, parts[1:4])
# #         r, g, b = map(float, parts[4:7])
# #         colors.append((r, g, b))
# #         new_lines.append(f'v {x} {y} {z}\n')
# #     else:
# #         new_lines.append(line)

# # # Create the texture image
# # colors = np.array(colors)
# # colors = np.clip(colors, 0, 1)  # ensure colors are in the range [0, 1]
# # colors = (colors * 255).astype(np.uint8)  # convert from float to 8-bit color

# # # Calculate the size of the image
# # num_colors = len(colors)
# # side_length = math.ceil(math.sqrt(num_colors))
# # colors = np.concatenate([colors, np.zeros((side_length * side_length - num_colors, 3), dtype=np.uint8)])  # pad with black
# # colors = colors.reshape(side_length, side_length, 3)

# # image = Image.fromarray(colors, 'RGB')

# # # Save the texture image
# # image.save('texture.jpg')

# # # Add texture coordinates to the OBJ file
# # grid_step = 1 / (side_length - 1)
# # for i in range(num_colors):
# #     x = (i % side_length) * grid_step
# #     y = (i // side_length) * grid_step
# #     new_lines.insert(i + 1, f'vt {x} {y}\n')

# # # Add texture coordinate indices to the faces
# # for i, line in enumerate(new_lines):
# #     if line.startswith('f'):
# #         parts = line.split()
# #         for j in range(1, len(parts)):
# #             parts[j] += f'/{j-1}'  # assuming one face per vertex, so index is just j-1
# #         new_lines[i] = ' '.join(parts) + '\n'

# # # Save the new OBJ file
# # with open('output.obj', 'w') as f:
# #     f.writelines(new_lines)











## depth.npz file to point cloud

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import open3d as o3d
import itertools
from tqdm import tqdm
from PIL import Image
from PIL.ImageFilter import MinFilter
import os

def find_border(img):
    img = img.filter(MinFilter(11))
    img = np.array(img)
    img_1 = np.sum(img, axis=2)
    img_x = np.sum(img_1, axis=0)
    img_y = np.sum(img_1, axis=1)
    x_min = img_x.shape[0]
    x_max = 0
    y_min = img_y.shape[0]
    y_max = 0
    for x in range(img_x.shape[0]):
        if img_x[x] > 0:
            x_min = x
            break
    for x in range(img_x.shape[0] - 1, 0, -1):
        if img_x[x] > 0:
            x_max = x
            break
    for y in range(img_y.shape[0]):
        if img_y[y] > 0:
            y_min = y
            break
    for y in range(img_y.shape[0] - 1, 0, -1):
        if img_y[y] > 0:
            y_max = y
            break
    return x_min, x_max, y_min, y_max



    # intrinsics[1, :] *= -1.0
    # intrinsics[1, 2] += 512
    # fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    # mask_img = Image.open(os.path.join('F:/SS23/AT3DCV/at3dcv_project/data/multihuman_single_inputs/mask/401','{}.png'.format(yaw))).convert('RGB')
    # x_min, x_max, y_min, y_max = find_border(mask_img)
    # y_min -= 50
    # y_max += 50
    # y_len = y_max - y_min
    # x_min = (x_max + x_min) // 2 - y_len // 2
    # x_max = x_min + y_len
    # scale = 512.0 / y_len
    # fx = fx * scale
    # fy = fy * scale
    # cx = scale * (cx - x_min)
    # cy = scale * (cy - y_min)
    # intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2] = fx, fy, cx, cy

def create_extrinsic_matrix(rotation_angle):
    # Convert rotation angle to radians
    rotation_angle_rad = np.radians(rotation_angle)

    # Create rotation matrix around the z-axis
    rotation_matrix = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad), 0],
                                [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad), 0],
                                [0, 0, 1]])

    # Create 3x4 extrinsic matrix with rotation and translation
    extrinsic_matrix = np.zeros((4, 4))
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[3, 3] = 1  # Assuming translation along z-axis
    return extrinsic_matrix

point_cloud_points = []
for yaw in ["0", "90", "180", "270"]:
    data = np.load(f'F:/SS23/AT3DCV/at3dcv_project/data/multihuman_single_inputs/depth/401/{yaw}.npz')
    extrinsics = np.load(f"F:/SS23/AT3DCV/at3dcv_project/data/multihuman_single_inputs/parameter/401/{yaw}_extrinsic.npy")
    intrinsics = np.load(f"F:/SS23/AT3DCV/at3dcv_project/data/multihuman_single_inputs/parameter/401/{yaw}_intrinsic.npy")
    depth_map = data['arr_0']





    height, width = depth_map.shape[:2]
    fx, fy, cx, cy = intrinsics[0, 0], -intrinsics[1, 1], intrinsics[0, 2], -intrinsics[1, 2] + 512
    intri = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # extri = create_extrinsic_matrix(int(yaw))

    # extrinsics = np.array(extrinsics, dtype = np.float64)
    extri = np.vstack((extrinsics, [0, 0, 0, 1]))
    
    # extri = np.zeros_like(extrinsics, dtype=np.float64)
    # R = extrinsics[:3, :3]
    # translation = extrinsics[:3, 3]
    # extri[:3, :3]= R.T
    # extri[:3, 3] = -R.T @ translation
    # extri = np.vstack((extri, [0, 0, 0, 1]))

    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_map.astype(np.float32)), intri)       #, depth_scale = 10
  
    pcd.transform(extri)
    point_cloud_points.append(pcd)

o3d.visualization.draw_geometries([point_cloud_points[0], point_cloud_points[1], point_cloud_points[2], point_cloud_points[3]])
# # o3d.io.write_point_cloud("test.ply", ([point_cloud_points[0], point_cloud_points[1], point_cloud_points[2], point_cloud_points[3]]))




    # This section is written and property of Tony Wang, please contact tony.wang@tum.de if questions arise. No spam please <3 xoxo

#     for x, y in tqdm(itertools.product(range(depth_map.shape[0]), range(depth_map.shape[1])), total=depth_map.shape[0] * depth_map.shape[1]):
#         depth = depth_map[x, y]
#         if depth == 0:
#             continue
#         depth *= 10
#         # import ipdb; ipdb.set_trace()



#         px_to_depth_cam = np.linalg.inv(intrinsics) @ np.array([x, y, 1]) * depth
#         # depth_cam_to_world = (extrinsics[:3, :3].T @ px_to_depth_cam) + extrinsics[:3, 3]
#         R = extrinsics[:3, :3]
#         translation = extrinsics[:3, 3]
#         depth_cam_to_world = R.T @ px_to_depth_cam - R.T @ translation

#         point_cloud_points.append(depth_cam_to_world)

# point_cloud_open3d = o3d.geometry.PointCloud()
# point_cloud_open3d.points = o3d.utility.Vector3dVector(np.array(point_cloud_points))
# o3d.io.write_point_cloud("tony.ply", point_cloud_open3d)




# "F:\SS23\AT3DCV\at3dcv_project\data\Synthetic\first_trial\smplx_z\0\smplx.obj"

# # np.save('F:/SS23/AT3DCV/pc_trial.npy', pc)







# ## count vertices of .obj files
# def count_vertices(obj_file_path):
#     vertex_count = 0

#     with open(obj_file_path, 'r') as obj_file:
#         for line in obj_file:
#             line = line.strip()  # Remove leading/trailing whitespace

#             if line.startswith('v '):
#                 vertex_count += 1

#     return vertex_count

# # obj_file_path = "F:\SS23\AT3DCV\at3dcv_project/lib/data/smplx_fine.obj"
# # F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/smplx/0/smplx.obj






### TRANSFORM coordinate system z axis --> y axis

def axis_transform(inputfp, direction, outputfile = False):
    fp = inputfp
    # fp = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/smplx_z/0/smplx.obj"
    with open(fp, 'r') as file:
        lines = file.readlines()

    if direction == 'z2y':
        transformation_matrix = [[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]]
    elif direction == 'y2z':
        transformation_matrix = [[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]]

    # Apply the transformation to the vertex coordinates
    transformed_lines = []
    for line in lines:
        if line.startswith("v "):  # Line with vertex coordinates
            vertex = [float(coord) for coord in line.split()[1:]]
            transformed_vertex = [sum([transformation_matrix[i][j] * vertex[j] for j in range(3)]) for i in range(3)]
            transformed_lines.append("v " + " ".join(str(coord) for coord in transformed_vertex) + "\n")
        else:
            transformed_lines.append(line)

    # Save the transformed .obj file
    if outputfile:
        output_file_path = "F:/test.obj"
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(transformed_lines)

        print(f"Transformed file saved: {output_file_path}")

