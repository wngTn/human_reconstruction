import os
import cv2
import numpy as np
import argparse
import open3d as o3d
import pyrender
import trimesh
from PIL import Image
import sys
# sys.path.insert(0, "F:\\SS23\\AT3DCV\\at3dcv_project\\external\\EasyMocap-master\\easymocap")
# from smplmodel.body_param import load_model

class FileStorage:
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__.split('.')
        self.major_version = int(version[0])
        self.second_version = int(version[1])
        self.isWrite = isWrite

        if self.isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

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
            return self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            return results
        else:
            raise NotImplementedError

    def close(self):
        self.__del__()


def load_intrinsics(file_path):
    assert os.path.exists(file_path), file_path
    intri = FileStorage(file_path)
    cameras = {key: {'K': intri.read('K_{}'.format(key)), 'invK': np.linalg.inv(intri.read('K_{}'.format(key))), 'dist': intri.read('dist_{}'.format(key))} for key in intri.read('names', dt='list')}
    return cameras


def load_extrinsics(file_path):
    extri = FileStorage(file_path)
    cameras = {key: {'R': extri.read('R_{}'.format(key)), 'Rot': extri.read('Rot_{}'.format(key)), 'T': extri.read('T_{}'.format(key))} for key in extri.read('names', dt='list')}
    return cameras

def render(meshes, cv2_image, R, T, K, output_img_path):
    H, W, _ = cv2_image.shape

    pred_pose = np.eye(4)
    pred_pose[:3, :3] = R
    pred_pose[:3, 3] = T.flatten()

    camera_pose = np.eye(4)
    camera_pose[1, 1] *= -1.0

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1., 1., 1.))
    camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    scene.add(camera, pose=camera_pose)

    for mesh in meshes:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        out_mesh = trimesh.Trimesh(vertices, faces, validate=False, process=False)
        human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3 ,0.5])
        mesh = pyrender.Mesh.from_trimesh(out_mesh, material=human_mat, smooth=True, wireframe=False)
        scene.add(mesh, 'mesh', pose=pred_pose)

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    output_img = 255 - ((255 - color[:, :, :-1]) * valid_mask + (1 - valid_mask) * cv2_image)
    output_img = (output_img * 255).astype(np.uint8)
    cv2.imwrite(output_img_path, output_img)


def main(root_dir, intri_file_path, extri_file_path):
    intrinsics = load_intrinsics(intri_file_path)
    extrinsics = load_extrinsics(extri_file_path)

    output_dir = os.path.join(root_dir, 'rendered_visualization')
    image_dirs = [os.path.join(root_dir, 'images', str(i)) for i in range(4)]
    smpl_dir = os.path.join(root_dir, 'smpl')

    for idx, image_dir in enumerate(image_dirs):
        cam_output_dir = os.path.join(output_dir, f"camera_{idx}")

        for frame_num in os.listdir(image_dir):
            img_path = os.path.join(image_dir, f"{frame_num.zfill(6)}")
            cv2_image = cv2.imread(img_path)

            smpl_path = os.path.join(smpl_dir, f"{int(frame_num[:-4])}.obj")
            smpl_mesh = o3d.io.read_triangle_mesh(smpl_path)

            K = intrinsics[str(idx)]['K']
            R = extrinsics[str(idx)]['R']
            T = extrinsics[str(idx)]['T'].reshape(3,)
            render(smpl_mesh, cv2_image, R, T, K, os.path.join(cam_output_dir, f"{frame_num.zfill(6)}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for the data.")
    parser.add_argument("--intri_file_path", type=str, required=True, help="Intrinsic parameters file path.")
    parser.add_argument("--extri_file_path", type=str, required=True, help="Extrinsic parameters file path.")
    args = parser.parse_args()

    main(args.root_dir, args.intri_file_path, args.extri_file_path)