import json
import yaml
import os
import argparse
import cv2
import numpy as np
import glob
from tqdm import tqdm
import re


def get_rotation_matrix(T):
    return T[:3, :3].flatten().tolist()


def get_translation_vector(T):
    return T[:3, 3].tolist()


def write_yaml(d, filename):
    with open(filename, 'w') as f:
        f.write("%YAML:1.0\n")
        f.write("---\n")
        for k, v in d.items():
            if isinstance(v, dict):
                f.write(k + ": !!opencv-matrix\n")
                for k2, v2 in v.items():
                    if isinstance(v2, list):
                        f.write("    " + k2 + ": [" + ", ".join(map(str, v2)) + "]\n")
                    else:
                        f.write("    " + k2 + ": " + str(v2) + "\n")
            elif isinstance(v, list):
                f.write(k + ":\n")
                for item in v:
                    f.write("    - '" + str(item) + "'\n")
            elif isinstance(v, tuple):
                for item in v:
                    if isinstance(item, dict):
                        f.write(k + ": !!opencv-matrix\n")
                        for k2, v2 in item.items():
                            if isinstance(v2, list):
                                f.write("    " + k2 + ": [" + ", ".join(map(str, v2)) + "]\n")
                            else:
                                f.write("    " + k2 + ": " + str(v2) + "\n")
            else:
                f.write(k + ": " + str(v) + "\n")



def conversion(input_json_file, num_cameras, output_dir):
    # Load the JSON data
    with open(input_json_file, 'r') as json_file:
        data = json.load(json_file)

    # Write the intrinsic camera parameters to YAML
    intrinsics = {
        "names": [f"{i}" for i in range(num_cameras)],
    }

    for i in range(num_cameras):
        intrinsics[f"K_{i}"] = {
            "rows":
                3,
            "cols":
                3,
            "dt":
                'd',
            "data": [
                data['cam_mat_intr']['f_x'], 0., data['cam_mat_intr']['c_x'], 0., data['cam_mat_intr']['f_y'],
                data['cam_mat_intr']['c_y'], 0., 0., 1.
            ]
        }
        intrinsics[f"dist_{i}"] = {"rows": 5, "cols": 1, "dt": 'd', "data": [0] * 5},  # Assuming zero distortion

    intr_output_path = os.path.join(output_dir, 'intri.yml')
    write_yaml(intrinsics, intr_output_path)
    # Write the extrinsic camera parameters to YAML
    extrinsics = {
        "names": [f"{i}" for i in range(num_cameras)],
    }

    for i in range(num_cameras):

        rotation = np.array(data["scene_camera"][f"cam_T_{i}"]["cam_R_w2c"]).reshape(3, 3)
        translation = (np.array(data["scene_camera"][f"cam_T_{i}"]["cam_t_w2c"]) / 1000.0).reshape(3, 1)

        extrinsics['Rot_' + str(i)] = {"rows": 3, "cols": 3, "dt": 'd', "data": rotation.flatten().tolist()}
        extrinsics['T_' + str(i)] = {"rows": 3, "cols": 1, "dt": 'd', "data": translation.flatten().tolist()}
        extrinsics['R_' + str(i)] = {
            "rows": 3,
            "cols": 1,
            "dt": 'd',
            "data": cv2.Rodrigues(rotation)[0].flatten().tolist()
        }

    extr_output_path = os.path.join(output_dir, 'extri.yml')
    write_yaml(extrinsics, extr_output_path)


def convert(input_json_file, output_dir, image_dir, fps, num_cameras):
    os.makedirs(output_dir, exist_ok=True)
    # Load JSON file
    # convert json to YAMLs
    conversion(input_json_file, num_cameras, output_dir)

    video_dir = os.path.join(output_dir, 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    for cam in range(num_cameras):
        # Image to video conversion
        images = glob.glob(os.path.join(image_dir, f"cam_{cam}", '*.png'))
        # Sorting function to extract the frame number and sort by it
        images.sort(key=lambda f: int(os.path.basename(f).split('_')[1]))
        video_path = os.path.join(video_dir, f'{cam}.mp4')
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        # For each camera, create a video from images
        for img_path in tqdm(images, desc="Creating videos"):

            video.write(cv2.imread(img_path))

        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to YAML files and images to videos")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory")
    parser.add_argument("-d", "--imagedir", required=True, help="Path to the image directory")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second for the video")
    parser.add_argument("-n", "--num_cameras", type=int, default=4, help="Number of cameras")
    args = parser.parse_args()
    convert(args.input, args.output, args.imagedir, args.fps, args.num_cameras)