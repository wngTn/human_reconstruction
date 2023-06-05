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

def conversion(input_json_file):
    # Load the JSON data
    with open(input_json_file, 'r') as json_file:
        data = json.load(json_file)

    # Write the intrinsic camera parameters to YAML
    with open('intri.yml', 'w') as intri_file:
        intrinsics = {
            "names": ["0", "1", "2", "3"],
            "K_0": {"rows": 3, "cols": 3, "dt": 'd', "data": [data['cam_mat_intr']['f_x'], 0., data['cam_mat_intr']['c_x'], 0., data['cam_mat_intr']['f_y'], data['cam_mat_intr']['c_y'], 0., 0., 1. ]},
            "dist_0": {"rows": 5, "cols": 1, "dt": 'd', "data": [0]*5}, # Assuming zero distortion
        }

        yaml.dump(intrinsics, intri_file, default_flow_style=False)

    # Write the extrinsic camera parameters to YAML
    with open('extri.yml', 'w') as extri_file:
        extrinsics = {
            "names": ["0", "1", "2", "3"],
        }

        for i in range(4):
            T = np.array(data['cam_mat_extr']['cam_T_' + str(i)])
            R = get_rotation_matrix(T)
            t = get_translation_vector(T)

            extrinsics['R_' + str(i)] = {"rows": 3, "cols": 3, "dt": 'd', "data": R}
            extrinsics['T_' + str(i)] = {"rows": 3, "cols": 1, "dt": 'd', "data": t}

        yaml.dump(extrinsics, extri_file, default_flow_style=False)


def convert(input_json_file, output_dir, image_dir, fps):

    # Load JSON file
    # convert json to YAMLs
    conversion(input_json_file)

    video_dir = os.path.join(output_dir, 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Image to video conversion
    images = glob.glob(os.path.join(image_dir, '*.png'))

    # Sorting function to extract the frame number and sort by it
    images.sort(key=lambda f: int(re.search(r'(\d+)_\d+.png$', os.path.basename(f)).group(1)))

    # Group images by camera name
    images_by_cam = {}
    for img_path in images:
        cam_name = img_path.split('_')[-1].split('.')[0]
        if cam_name not in images_by_cam:
            images_by_cam[cam_name] = []
        images_by_cam[cam_name].append(img_path)

    # For each camera, create a video from images
    for cam_name, img_paths in tqdm(images_by_cam.items(), desc="Creating videos"):
        video_path = os.path.join(video_dir, f'{cam_name}.mp4')
        frame = cv2.imread(img_paths[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for img_path in img_paths:
            video.write(cv2.imread(img_path))

        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to YAML files and images to videos")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory")
    parser.add_argument("-d", "--imagedir", required=True, help="Path to the image directory")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second for the video")
    args = parser.parse_args()
    convert(args.input, args.output, args.imagedir, args.fps)