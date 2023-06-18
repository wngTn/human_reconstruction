import trimesh
import numpy as np
import os

def modify_smpl_objects(folder_path):
    for subfolder in os.listdir(folder_path):
        file_path = os.path.join(folder_path, subfolder, 'smplx.obj')
        obj = trimesh.load_mesh(file_path)

        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
        rotated_vertices = np.dot(obj.vertices, rotation_matrix.T)
        rotated_normals = np.dot(obj.vertex_normals, rotation_matrix.T)
        obj.vertices = rotated_vertices
        obj.vertex_normals = rotated_normals
        # output_path = os.path.join('F:/', subfolder)
        # os.makedirs(output_path, exist_ok=True)
        obj.export(file_path)
        

folder_path = 'F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/smplx'
modify_smpl_objects(folder_path)