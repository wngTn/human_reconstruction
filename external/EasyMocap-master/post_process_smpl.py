import os
import torch
import numpy as np
import json
import open3d as o3d
import json
import sys
from tqdm import tqdm
from pathlib import Path
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'easymocap')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from smplmodel.body_param import load_model

# reads a json file
def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data

# reads a smpl file
def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ["Rh", "Th", "poses", "shapes", "expression"]:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        outputs.append(data)
    return outputs


# creates mesh out of vertices and faces
def create_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def load_mesh(smpl_json_file):
    """
    Loads the meshes from the data_dir

    :param data_dir: The path to the smpl files
    :param frame_ids: The frame id
    :return: Returns a list of the meshes of the frame id
    """
    # loads the smpl model
    body_model = load_model(gender="neutral", model_path="external/EasyMocap-master/data/smplx")

    data = read_smpl(smpl_json_file)
    # all the meshes in a frame
    frame_meshes = []
    frame_ids = []
    for i in range(len(data)):
        frame = data[i]
        Rh = frame["Rh"]
        Th = frame["Th"]
        poses = frame["poses"]
        shapes = frame["shapes"]

        # gets the vertices
        vertices = body_model(poses, shapes, Rh, Th, return_verts=True, return_tensor=False)[0]
        rotation_90 = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        vertices = vertices @ rotation_90.T

        # the mesh

        model = create_mesh(vertices=vertices, faces=body_model.faces)

        frame_meshes.append(model)
        frame_ids.append(frame["id"])

    return frame_meshes, frame_ids

def main(input_dir, output_dir):
    smpl_json_file = Path(input_dir).glob("*.json")

    for j, smpl_file in (process_bar:=tqdm(enumerate(smpl_json_file), desc="Processing SMPL files")): 
        if j >= 5:
            break
        process_bar.set_postfix(file=smpl_file)
        frame_meshes, frame_ids = load_mesh(smpl_file)
        for i in range(len(frame_meshes)):
            o3d.io.write_triangle_mesh(os.path.join(output_dir, "smpl_" + str(j).zfill(6) + ".obj"), frame_meshes[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL post processing")
    parser.add_argument("--input_dir", type=str, default="data/Synthetic/first_trial_easymocap/output-track/smpl", help="The input directory of the smpl files")
    parser.add_argument("--output_dir", type=str, default="data/Synthetic/first_trial/Obj_Pred", help="The output directory of the smpl files")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
