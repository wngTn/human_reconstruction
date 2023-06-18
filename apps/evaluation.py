# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
from PIL import Image
from tqdm import tqdm

import argparse

def euler_to_rot_mat(r_x, r_y, r_z):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r_x), -math.sin(r_x)],
                    [0, math.sin(r_x), math.cos(r_x)]
                    ])

    R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                    [0, 1, 0],
                    [-math.sin(r_y), 0, math.cos(r_y)]
                    ])

    R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                    [math.sin(r_z), math.cos(r_z), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


class MeshEvaluator:
    _normal_render = None

    @staticmethod
    def init_gl():
        # from .render.gl.normal_render import NormalRender
        # MeshEvaluator._normal_render = NormalRender(width=512, height=512)
        MeshEvaluator._normal_render = None

    def __init__(self):
        pass

    def set_mesh(self, src_path, tgt_path, scale_factor=1.0, offset=0):
        self.src_mesh = trimesh.load(src_path)
        self.tgt_mesh = trimesh.load(tgt_path)

        self.scale_factor = scale_factor
        self.offset = offset


    def get_chamfer_dist(self, num_samples=10000):
        # Chamfer
        # import ipdb; ipdb.set_trace()
        src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)
        tgt_surf_pts, _ = trimesh.sample.sample_surface(self.tgt_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)
        _, tgt_src_dist, _ = trimesh.proximity.closest_point(self.src_mesh, tgt_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        ## isfinite
        src_tgt_dist[~np.isfinite(src_tgt_dist)] = 0
        tgt_src_dist[~np.isfinite(tgt_src_dist)] = 0 

        src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

        return chamfer_dist

    def get_surface_dist(self, num_samples=10000):
        # P2S
        src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        src_tgt_dist[~np.isfinite(src_tgt_dist)] = 0
        
        src_tgt_dist = src_tgt_dist.mean()

        return src_tgt_dist
    

def run(obj_root, obj_list, target_root):
    
    total_vals = []
    items = []
    for obj in tqdm(obj_list):
    # for i, obj in tqdm(enumerate(obj_list)):
        obj_split_name = obj.split('_')
        obj_frame_id = obj_split_name[2]
        obj_path = os.path.join(obj_root, obj)

        ## adjust accordingly
        target_path = os.path.join(target_root, obj_frame_id, f'{obj_frame_id}.obj')
        # target_path = os.path.join(target_root, f'{obj_frame_id}_{i}', 'smplx.obj')

        evaluator.set_mesh(obj_path, target_path)

        vals = []
        vals.append(evaluator.get_chamfer_dist())
        vals.append(evaluator.get_surface_dist())

        item = {
            'name': f'{obj_frame_id}.obj',
            'vals': vals
        }

        total_vals.append(vals)
        items.append(item)

    # np.save(os.path.join("../results", 'eval_results_items.npy'), np.array(items))
    # np.save(os.path.join("../results", 'eval_results_vals.npy'), total_vals)

    vals = np.array(total_vals).mean(0)
    print('MH dataset chamfer: %.6f  p2s: %.6f ' % (vals[0], vals[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='r')
    parser.add_argument('-t', '--tar_path', type=str, default='t')
    parser.add_argument('--config', type=str)
    parser.add_argument('--dataroot', type=str)
    args = parser.parse_args()

    evaluator = MeshEvaluator()

    # evaluator.init_gl()


    obj_root = "results/dmc_demo/demo"
    target_root = "data/demo"
    # obj_root = "results/multihuman_single_trial"
    # target_root = "data/MultiHuman/single/obj"
    _obj_list = os.listdir(obj_root)
    obj_list = [obj for obj in _obj_list if obj.endswith('.obj')]
    # import ipdb; ipdb.set_trace()
    run(obj_root, obj_list, target_root)



## RECORDS of eval experiments:
# EXAMPLE-SKIRT) - chamfer: 0.0059  p2s: 0.0058  
# DEMO-DOUBLE - chamfer: 0.0159  p2s: 0.0166 
# MH-SINGLE-30-6VIEWS - chamfer: 0.0114  p2s: 0.0103    --> in meters --> *100 comparable with metric stated in papers (in centimeter)
# MH-SINGLE-30-6VIEWS - chamfer: 0.0102  p2s: 0.0091  