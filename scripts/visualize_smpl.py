import os
import torch
import numpy as np
import json
import open3d as o3d
import json
import sys
sys.path.insert(0, "F:\\SS23\\AT3DCV\\at3dcv_project\\external\\EasyMocap-master\\easymocap")
from smplmodel.body_param import load_model
import cv2

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


def load_joints(data_dir: str, frame_id):
    """
    Loads the body joints from the data_dir

    :param data_dir: The path to the smpl files
    :param frame_ids: The frame id
    :return: Returns a list of the body_joints of the frame id
    """
    # loads the smpl model
    body_model = load_model(gender="neutral", model_path="data/smpl_models")

    data = read_smpl(os.path.join(data_dir, str(frame_id).zfill(6) + ".json"))
    # all the meshes in a frame
    frame_pcds = []
    for i in range(len(data)):
        frame = data[i]
        Rh = frame["Rh"]
        Th = frame["Th"]
        poses = frame["poses"]
        shapes = frame["shapes"]

        # gets the vertices
        vertices = body_model(
            poses,
            shapes,
            Rh,
            Th,
            return_verts=False,
            return_tensor=False,
            return_smpl_joints=True,
        )[0]

        frame_pcds.append(vertices)

    return frame_pcds


def load_mesh(data_dir: str, frame_id):
    """
    Loads the meshes from the data_dir

    :param data_dir: The path to the smpl files
    :param frame_ids: The frame id
    :return: Returns a list of the meshes of the frame id
    """
    # loads the smpl model
    body_model = load_model(gender="neutral", model_path="external/EasyMocap-master/data/smplx")

    data = read_smpl(os.path.join(data_dir, str(frame_id).zfill(6) + ".json"))
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

        # the mesh

        model = create_mesh(vertices=vertices, faces=body_model.faces)

        frame_meshes.append(model)
        frame_ids.append(frame["id"])

    return frame_meshes, frame_ids

import yaml

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.3f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras

def read_extri(extri_name):
    extri = FileStorage(extri_name)
    camnames = extri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['R'] = extri.read('R_{}'.format(key))
        cam['Rot'] = extri.read('Rot_{}'.format(key))
        cam['T'] = extri.read('T_{}'.format(key))
        cameras[key] = cam
    return cameras

def load_intrinsics(file_path):
    intrinsics = read_intri(file_path)
    return intrinsics

def load_extrinsics(file_path):
    extrinsics = read_extri(file_path)
    return extrinsics

def project_to_2d(vertices, intrinsics, extrinsics):
    vertices_4d_hom = np.hstack((vertices, np.ones((vertices.shape[0],1)))) # Convert to homogeneous coordinates
    points_2d = intrinsics @ (extrinsics[:3, ...] @ vertices_4d_hom.T)
    points_2d /= points_2d[2, :] # divide by the third row to perform the actual perspective projection
    return points_2d[:2, :].T # we only need the first two rows, return those transposed

def render_mesh_to_image(mesh, image, intrinsics, extrinsics):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    projected_vertices = project_to_2d(vertices, intrinsics, extrinsics)

    for triangle in triangles:
        pts = projected_vertices[triangle, :].astype(np.int32)
        cv2.fillConvexPoly(image, pts, (0, 255, 0)) # fill the triangle with green color

    return image

# load camera parameters
intri = load_intrinsics('data/Synthetic/first_trial_easymocap/intri.yml')
extri = load_extrinsics('data/Synthetic/first_trial_easymocap/extri.yml')

frame_dirs = ['data/Synthetic/first_trial_easymocap/images/0',
              'data/Synthetic/first_trial_easymocap/images/1',
              'data/Synthetic/first_trial_easymocap/images/2',
              'data/Synthetic/first_trial_easymocap/images/3']

from tqdm import tqdm
for cam_num, frame_dir in enumerate(frame_dirs):
    intrinsics = np.array(intri[str(cam_num)]['K'])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = np.array(extri[str(cam_num)]['R'])
    extrinsics[:3, 3] = np.array(extri[str(cam_num)]['T']).reshape(3,)
    frame_images = os.listdir(frame_dir)
    for i, frame_image_name in tqdm(enumerate(frame_images)):
        # read image
        image = cv2.imread(os.path.join(frame_dir, frame_image_name))
        # get frame id
        frame_id = int(frame_image_name.split('.')[0])
        # load the mesh
        frame_meshes, _ = load_mesh('data/Synthetic/first_trial_easymocap/output-track/smpl', frame_id)
        for mesh in frame_meshes:
            # render mesh to image
            # image = render_mesh_to_image(mesh, image, intrinsics, extrinsics)
            # save the image
            # cv2.imwrite(os.path.join(frame_dir, f"rendered_{frame_image_name}"), image)
            o3d.io.write_triangle_mesh(f"data/Synthetic/first_trial_easymocap/smpl/{i}.obj", mesh)

