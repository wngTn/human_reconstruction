import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os, sys
from PIL import Image
from tqdm import tqdm
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def write_pcd(pcd1pts, pcd2pts, file_name):

    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1pts)
    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2pts)

    pcd1_o3d.paint_uniform_color([1, 0, 0])
    pcd2_o3d.paint_uniform_color([0, 0, 1])
    combined_pcd = pcd1_o3d + pcd2_o3d

    # save ply files
    o3d.io.write_point_cloud(file_name, combined_pcd)
    print(f"Saved point cloud as PLY file: {file_name}")


def euler_to_rot_mat(r_x, r_y, r_z):
    R_x = np.array([[1, 0, 0], [0, math.cos(r_x), -math.sin(r_x)], [0, math.sin(r_x), math.cos(r_x)]])

    R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0], [-math.sin(r_y), 0, math.cos(r_y)]])

    R_z = np.array([[math.cos(r_z), -math.sin(r_z), 1], [math.sin(r_z), math.cos(r_z), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class Evaluator:

    def __init__(self,
                 pred_paths,
                 gt_paths,
                 num_samples,
                 save_pcd,
                 folder,
                 human_paths=None,
                 cloth_paths=None,
                 dataset_name=None,
                 transformation=None):
        """
        _summary_

        Args:
            pred_paths (list[strings]): 
            gt_paths (list[strings]): TODO 
            evaluate_separately (bool): TODO
            num_samples (int): TODO
            save_pcd (bool): TODO 
            folder (str): TODO
        """

        self.pred_paths = pred_paths
        self.gt_paths = gt_paths
        self.human_paths = human_paths
        self.cloth_paths = cloth_paths

        self.transformation = transformation

        self.pred_objects = list(self.get_pred_objects)
        self.gt_objects = list(self.get_gt_objects)
        self.human_objects = list(self.get_human_objects) if human_paths is not None else None
        self.cloth_objects = list(self.get_cloth_objects) if cloth_paths is not None else None

        self.num_samples = num_samples
        self.save_pcd = save_pcd
        self.folder = folder
        os.makedirs(os.path.join(self.folder), exist_ok=True)

        self.log_path = os.path.join(self.folder, "metrics.txt")
        self.log_fout = open(self.log_path, "w")

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str, flush=True)

    @property
    def get_pred_objects(self):
        return self._get_pred_objects()

    def _get_pred_objects(self):
        for path in self.pred_paths:
            pred_obj = trimesh.load_mesh(path)
            if self.transformation is not None:
                pred_obj.apply_transform(self.transformation)
            yield pred_obj

    @property
    def get_gt_objects(self):
        return self._get_gt_objects()

    def _get_gt_objects(self):
        for path in self.gt_paths:
            yield trimesh.load_mesh(path)

    @property
    def get_human_objects(self):
        return self._get_human_objects()

    def _get_human_objects(self):
        for path in self.human_paths:
            yield trimesh.load_mesh(path)

    @property
    def get_cloth_objects(self):
        return self._get_cloth_objects()

    def _get_cloth_objects(self):
        for path in self.cloth_paths:
            yield trimesh.load_mesh(path)

    def get_chamfer_distance(self):
        self._log(f"{'-' * 6}Evaluating Chamfer Distance{'-' * 6}")
        dist_list = []
        for file_index, (pred_obj, gt_obj) in tqdm(enumerate(zip(self.pred_objects, self.gt_objects)),
                                                   total=len(self.pred_paths)):
            point_pairs_dict = self.compute_chamfer_dist_with_trimesh(pred_obj, gt_obj)
            dist_list.append(point_pairs_dict["chamfer_distance"])
            self._log(f"{file_index}th pair of meshes: chamfer dist = {point_pairs_dict['chamfer_distance']}")

            if self.save_pcd:
                output_dir = os.path.join(self.folder, "pcd_visualization", "chamfer_distance")
                os.makedirs(output_dir, exist_ok=True)
                file_name = os.path.join(
                    output_dir,
                    f"pred_{os.path.basename(self.pred_paths[file_index])[:-4]}_gt_{os.path.basename(self.gt_paths[file_index])[:-4]}.ply"
                )
                write_pcd(point_pairs_dict["gt_pts"], point_pairs_dict["closest_pts_to_gt"], file_name)
        self._log(f"{'-' * 6}Result of Chamfer Distance{'-' * 6}")
        self._log(f"Evaluated {len(dist_list)} meshes, the mean chamfer distance is {np.mean(dist_list)}")
        self._log(f"{'-' * 6}End Result of Chamfer Distance{'-' * 6}")

        if self.human_objects is not None and self.cloth_objects is not None:
            dist_list_human = []
            dist_list_cloth = []
            for file_index, (pred_obj, human_obj,
                             cloth_obj) in enumerate(zip(self.pred_objects, self.human_objects, self.cloth_objects)):
                point_pairs_dict_human = self.compute_chamfer_dist_with_trimesh(pred_obj, human_obj)
                point_pairs_dict_cloth = self.compute_chamfer_dist_with_trimesh(pred_obj, cloth_obj)
                dist_list_human.append(point_pairs_dict_human["chamfer_distance"])
                dist_list_cloth.append(point_pairs_dict_cloth["chamfer_distance"])

            self._log(f"{'-' * 6}Result of Clothed Chamfer Distance{'-' * 6}")
            self._log(
                f"Evaluated {min(len(self.pred_paths), len(self.human_paths), len(self.cloth_paths))} meshes, the mean chamfer distance is for human {np.mean(dist_list_human)} and for cloth {np.mean(dist_list_cloth)}"
            )
            self._log(f"{'-' * 6}End Result of Clothed Chamfer Distance{'-' * 6}")

    def compute_chamfer_dist_with_trimesh(self, pred_obj, gt_obj):
        pred_surf_pts, _ = trimesh.sample.sample_surface(pred_obj, self.num_samples)
        gt_surf_pts, _ = trimesh.sample.sample_surface(gt_obj, self.num_samples)
        closest_pts_to_pred, pred_gt_dist, _ = trimesh.proximity.closest_point(gt_obj, pred_surf_pts)
        closest_pts_to_gt, gt_pred_dist, _ = trimesh.proximity.closest_point(pred_obj, gt_surf_pts)

        pred_gt_dist[np.isnan(pred_gt_dist)] = 0
        gt_pred_dist[np.isnan(gt_pred_dist)] = 0
        pred_gt_dist[~np.isfinite(pred_gt_dist)] = 0
        gt_pred_dist[~np.isfinite(gt_pred_dist)] = 0

        pred_gt_dist = pred_gt_dist.mean()
        gt_pred_dist = gt_pred_dist.mean()

        chamfer_dist = (pred_gt_dist + gt_pred_dist) / 2

        point_pairs_dict = {
            "chamfer_distance": chamfer_dist,
            "pred_pts": pred_surf_pts,
            "gt_pts": gt_surf_pts,
            "closest_pts_to_gt": closest_pts_to_gt,
            "closest_pts_to_pred": closest_pts_to_pred
        }
        return point_pairs_dict

    def init_gl(self):
        from lib.render.gl.normal_render import NormalRender
        self.normal_render = NormalRender(width=512, height=512)

    def get_P2S_distance(self):
        self._log(f"{'-' * 6}Evaluating P2S Distance{'-' * 6}")
        p2s_dist_list = []
        for idx, (pred_obj, gt_obj) in enumerate(zip(self.pred_objects, self.gt_objects)):
            # for file_index, (pred_obj_file, gt_obj_file) in enumerate(zip(self.pred_paths, self.gt_paths)):
            # pred_obj = trimesh.load_mesh(pred_obj_file)
            # gt_obj = trimesh.load_mesh(gt_obj_file)
            pred_surf_pts, _ = trimesh.sample.sample_surface(pred_obj, self.num_samples)
            p2s_dist = self.compute_P2S_distance(pred_surf_pts, gt_obj)
            p2s_dist_list.append(p2s_dist)
            self._log(f"{idx}th pair of meshes: p2s dist (pts:pred / mesh:gt) = {p2s_dist}")

        self._log(f"{'-' * 6}Result of P2S Distance{'-' * 6}")
        self._log(
            f"Evaluated {min(len(self.pred_paths), len(self.gt_paths))} meshes, the mean P2S distance is {np.mean(p2s_dist_list)}"
        )
        self._log(f"{'-' * 6}End Result of P2S Distance{'-' * 6}")

        if self.human_objects is not None and self.get_cloth_objects is not None:
            p2s_list_human = []
            p2s_list_cloth = []
            # for (human_obj, cloth_obj) in zip(self.human_objects, self.get_cloth_objects):
            for file_index, (pred_obj, human_obj,
                             cloth_obj) in enumerate(zip(self.pred_objects, self.human_objects,
                                                         self.get_cloth_objects)):
                human_gt_pts, _ = trimesh.sample.sample_surface(human_obj, self.num_samples)
                cloth_gt_pts, _ = trimesh.sample.sample_surface(cloth_obj, self.num_samples)
                p2s_dist_human = self.compute_P2S_distance(human_gt_pts, pred_obj)
                p2s_dist_cloth = self.compute_P2S_distance(cloth_gt_pts, pred_obj)
                p2s_list_human.append(p2s_dist_human)
                p2s_list_cloth.append(p2s_dist_cloth)
            self._log(f"{'-' * 6}Result of Clothed P2S Distance{'-' * 6}")
            self._log(
                f"Evaluated {min(len(self.pred_paths), len(self.human_paths), len(self.cloth_paths))} meshes, the P2S dist to pred is for human {np.mean(p2s_list_human)} and for cloth {np.mean(p2s_list_cloth)}"
            )
            self._log(f"{'-' * 6}End Result of Clothed P2S Distance{'-' * 6}")
        # return np.mean(p2s_dist_list)

    def compute_P2S_distance(self, pts, mesh):
        _, pred_gt_dist, _ = trimesh.proximity.closest_point(mesh, pts)
        pred_gt_dist[np.isnan(pred_gt_dist)] = 0
        pred_gt_dist = pred_gt_dist.mean()
        # pred_gt_dist[~np.isfinite(pred_gt_dist)] = 0
        return pred_gt_dist

    def _render_normal(self, mesh, deg, scale_factor=120.0, offset=-90):
        view_mat = np.identity(4)
        view_mat[:3, :3] *= 2 / 256
        rz = deg / 180. * np.pi
        model_mat = np.identity(4)
        model_mat[:3, :3] = euler_to_rot_mat(0, rz, 0)
        model_mat[1, 3] = offset
        view_mat[2, 2] *= -1

        self.normal_render.set_matrices(view_mat, model_mat)
        self.normal_render.set_normal_mesh(scale_factor * mesh.vertices, mesh.faces, mesh.vertex_normals, mesh.faces)
        self.normal_render.draw()
        normal_img = self.normal_render.get_color()
        return normal_img

    def _get_reproj_normal_error(self, pred_obj, gt_obj, deg):
        gt_normal = self._render_normal(gt_obj, deg)
        pred_normal = self._render_normal(pred_obj, deg)
        error = ((pred_normal[:, :, :3] - gt_normal[:, :, :3])**2).mean() * 3
        return error, pred_normal, gt_normal

    def get_reproj_normal_error(self, frontal=True, back=True, left=True, right=True):
        # reproj error
        # if save_demo_img is not None, save a visualization at the given path (etc, "./test.png")
        self._log(f"{'-' * 6}Evaluating Reprojection Error{'-' * 6}")
        os.makedirs(os.path.join(self.folder, "repr_error"), exist_ok=True)

        if self.normal_render is None:
            print("In order to use normal render, "
                  "you have to create a valid normal_render before running any normal evaluation.")
            return -1

        error_list = []
        # Define translation (0.15 units in Y direction)
        translation_matrix = trimesh.transformations.translation_matrix([0, -0.15, 0])
        for file_index, (pred_obj, gt_obj) in enumerate(zip(self.pred_objects, self.gt_objects)):
            pred_obj.apply_transform(translation_matrix)
            gt_obj.apply_transform(translation_matrix)

            side_cnt = 0
            total_error = 0
            demo_list = []
            if frontal:
                side_cnt += 1
                error, pred_normal, gt_normal = self._get_reproj_normal_error(pred_obj, gt_obj, 0)
                total_error += error
                demo_list.append(np.concatenate([pred_normal, gt_normal], axis=0))
            if back:
                side_cnt += 1
                error, pred_normal, gt_normal = self._get_reproj_normal_error(pred_obj, gt_obj, 180)
                total_error += error
                demo_list.append(np.concatenate([pred_normal, gt_normal], axis=0))
            if left:
                side_cnt += 1
                error, pred_normal, gt_normal = self._get_reproj_normal_error(pred_obj, gt_obj, 90)
                total_error += error
                demo_list.append(np.concatenate([pred_normal, gt_normal], axis=0))
            if right:
                side_cnt += 1
                error, pred_normal, gt_normal = self._get_reproj_normal_error(pred_obj, gt_obj, 270)
                total_error += error
                demo_list.append(np.concatenate([pred_normal, gt_normal], axis=0))
            res_array = np.concatenate(demo_list, axis=1)
            res_img = Image.fromarray((res_array * 255).astype(np.uint8))
            save_name = os.path.splitext(self.pred_paths[file_index])[0].split('/')[-1]
            os.makedirs(os.path.join(self.folder, "repr_error"), exist_ok=True)
            save_path = os.path.join(self.folder, "repr_error", 'normal_vis_%s.png' % save_name)
            res_img.save(save_path)
            error_list.append(total_error / side_cnt)
            self._log(f"Reprojection normal error for {file_index}th mesh = {total_error / side_cnt}")

        self._log(f"{'-' * 6}Result of Reprojection Error{'-' * 6}")
        self._log(
            f"Evaluated {min(len(self.pred_paths), len(self.gt_paths))} meshes, the mean reprojection normal error is {np.mean([x for x in error_list if not math.isnan(x)])}"
        )
        self._log(f"{'-' * 6}End Result of Reprojection Error{'-' * 6}")


def get_pifu_hd_configs(dataset, output_dir):
    val_dataroot = os.path.join("data", "Synthetic", dataset)
    pred_paths = []
    gt_paths = []
    human_paths = []
    cloth_paths = []
    # Define rotation (90 degrees about Z axis)
    if output_dir.split('/')[-2] == "EASY_PIFU_HD_CAM_6":
        rotation_matrix = trimesh.transformations.rotation_matrix(-0.3 * np.radians(90), [0, 1, 0])
    else:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])

    # Define translation (0.9 units in Y direction)
    translation_matrix = trimesh.transformations.translation_matrix([0, 0.9, 0])

    # Combine rotation and translation into one matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    if dataset == "squat":
        val_frames = [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
            57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107,
            109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
            151, 153, 155, 157, 159, 161, 163, 165
        ]
        for val_frame in val_frames:
            pred_paths.append(os.path.join(output_dir, "meshes", f"pred_{val_frame:04}.obj"))
            gt_paths.append(os.path.join(val_dataroot, "Obj", "person_0", "combined", f"smplx_{val_frame:06}.obj"))
            human_paths.append(
                os.path.join(val_dataroot, "Obj", "person_0", "smplx_no_cloth", f"smplx_{val_frame:04}.obj"))
            cloth_paths.append(os.path.join(val_dataroot, "Obj", "person_0", "cloth", f"cloth_{val_frame:04}.obj"))

        config = {
            "pred_paths": pred_paths,
            "gt_paths": gt_paths,
            "num_samples": 5000,
            "save_pcd": True,
            "folder": output_dir,
            "human_paths": human_paths,
            "cloth_paths": cloth_paths,
            "transformation": transformation_matrix
        }
        return config
    else:
        raise NotImplementedError


if __name__ == '__main__':
    kwargs = get_pifu_hd_configs("squat", os.path.join("outputs", "EASY_PIFU_HD_CAM_6", "squat"))
    evaluator = Evaluator(**kwargs)

    evaluator.get_chamfer_distance()
    evaluator.get_P2S_distance()
    evaluator.init_gl()
    evaluator.get_reproj_normal_error()
