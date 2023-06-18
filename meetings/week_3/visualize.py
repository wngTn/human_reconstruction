import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import itertools
import cv2

def load_parameters(prefix_path):
    parameters = np.load(prefix_path.joinpath("output_data.npz"), allow_pickle=True)
    return parameters

def load_cam_parameters(parameters, cam_num):
    intrinsics = np.array(parameters["scene_camera"].item()[f"cam_T_{cam_num}"]["cam_K"]).reshape(3, 3)
    world_to_cam_R = np.array(parameters["scene_camera"].item()[f"cam_T_{cam_num}"]["cam_R_w2c"]).reshape(3, 3)
    world_to_cam_T = np.array(parameters["scene_camera"].item()[f"cam_T_{cam_num}"]["cam_t_w2c"]).reshape(3, 1)
    world_to_cam_matrix = np.concatenate([world_to_cam_R, world_to_cam_T], axis=1)
    # reshaping to 4z4
    world_to_cam_matrix = np.concatenate([world_to_cam_matrix, np.array([[0, 0, 0, 1]])], axis=0)

    camera_to_world_matrix = np.array(parameters["camera_world"].item()[f"cam_T_{cam_num}"]).reshape(4, 4)

    return intrinsics, world_to_cam_matrix, camera_to_world_matrix




def project_to_2D(prefix_path, output_path):
    prefix = Path(prefix_path)
    output_path = Path(output_path)

    img_output_path = (output_path / "rendered_3D_2D_images")
    img_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving rendered images to {output_path / 'rendered_3D_2D_images'}")

    img_files = sorted(prefix.joinpath("images").glob("*.png"))

    parameters = load_parameters(prefix)

    # loading obj file
    mesh = o3d.io.read_triangle_mesh(str(prefix / "smplx" / "smplx_0.obj"))
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

    for i in tqdm(range(4)):
        K, world_2_camera, camera_2_world = load_cam_parameters(parameters, i)
        # 4 x 4 word -> camera matrix
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
                img[int(point[1]), int(point[0])] = [255, 0, 0, 255]  # red for all other points.

        img = Image.fromarray(img)
        img.save(img_output_path.joinpath(f"rendered_origin_{i}.png"))
        print(f"Saved rendered image to {img_output_path.joinpath(f'rendered_origin_{i}.png')}")


def show_depth_projection(prefix_path):
    prefix_path = Path(prefix_path)

    depth_files = sorted(prefix_path.joinpath("depth").glob("*.png"))

    parameters = load_parameters(prefix_path)
    pcd = o3d.geometry.PointCloud()

    d = 90
    R_obj = np.array([
        [1, 0, 0],
        [0, np.cos(np.deg2rad(d)), -np.sin(np.deg2rad(d))],
        [0, np.sin(np.deg2rad(d)), np.cos(np.deg2rad(d))]
    ])

    for i, depth_file in enumerate(depth_files):
        K, world_2_camera, camera_2_world = load_cam_parameters(parameters, i)
        d_tmp = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        depth_data = np.asarray(d_tmp)[:, :, 0]
        # depth_data = np.max(depth_data) - depth_data
        # get camera to world
        # camera_2_world = world_2_camera
        # p_m = camera_2_world
        p_m = world_2_camera
        p_m[:3, 3] = p_m[:3, 3] / 1000
        R = p_m[:3, :3]
        T = p_m[:3, 3]
        pcd_points = []
        for x, y in tqdm(itertools.product(range(512), range(512)), total=512*512, desc=f"Processing depth image {i}"):
            if depth_data[x, y] == 0:
                continue
            # projection from 2D depth to 3D
            depth_value = depth_data[x, y] / 100
            depth_cam_point = np.dot(np.linalg.inv(K), np.array([x, y, 1])) * depth_value
            
            world_point = world_2_camera @ np.array([*depth_cam_point, 1])
            # world_point = (R.T @ depth_cam_point[:3]) + (-R.T @ T).T
            pcd_points.append(world_point[:3])


        # pcd_points = np.matmul(R_obj, np.array(pcd_points).T).T
        pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.array(pcd.points), pcd_points), axis=0))

    smpl = o3d.io.read_triangle_mesh(str(prefix_path / "smplx" / "smplx_0.obj"))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix_path", default="meetings/week_3", help="Path prefix for the files")
    parser.add_argument("-o", "--output_path", type=str, default="meetings/week_3/output", help="Output of the rendered images")
    parser.add_argument("--twod", action="store_true", help="Project the 3D points to 2D")
    parser.add_argument("--depth", action="store_true", help="Show Depth Projection")
    args = parser.parse_args()
    if args.twod:
        project_to_2D(args.prefix_path, output_path=args.output_path)
    if args.depth:
        show_depth_projection(args.prefix_path)
