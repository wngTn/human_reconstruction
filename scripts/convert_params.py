import json
import yaml
import os
import argparse
import cv2
import numpy as np
import glob
from tqdm import tqdm
import re


def convert(input_json_file, output_dir, image_dir, fps):
    # load json data
    with open(input_json_file, "r") as read_file:
        data = json.load(read_file)

    intr = {}
    intr["%YAML:1.0"] = None
    intr["names"] = [str(i) for i in range(4)]
    for i in range(4):
        intr[f"K_{i}"] = {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": [data["cam_mat_intr"]["f_x"], 0, data["cam_mat_intr"]["c_x"], 0, data["cam_mat_intr"]["f_y"], data["cam_mat_intr"]["c_y"], 0, 0, 1]
        }
        intr[f"dist_{i}"] = {
            "rows": 5,
            "cols": 1,
            "dt": "d",
            "data": [0, 0, 0, 0, 0]
        }

    extr = {}
    extr["%YAML:1.0"] = None
    extr["names"] = [str(i) for i in range(4)]
    
    for i in range(4):
        extr[f"R_{i}"] = {
            "rows": 3,
            "cols": 1,
            "dt": "d",
            "data": np.eye(3).ravel().tolist()
        }
        extr[f"Rot_{i}"] = {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": np.array(data["cam_mat_extr"][f"cam_T_{i}"])[:3, :3].ravel().tolist()
        }
        extr[f"T_{i}"] = {
            "rows": 3,
            "cols": 1,
            "dt": "d",
            "data": np.array(data["cam_mat_extr"][f"cam_T_{i}"])[:3, 3].ravel().tolist()
        }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "intri.yml"), 'w') as yaml_file:
        yaml.dump(intr, yaml_file, default_flow_style=False)

    with open(os.path.join(output_dir, "extri.yml"), 'w') as yaml_file:
        yaml.dump(extr, yaml_file, default_flow_style=False)

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