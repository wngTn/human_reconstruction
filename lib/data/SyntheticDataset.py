from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw
import cv2
import torch
from PIL.ImageFilter import GaussianBlur, MinFilter
import trimesh
from tqdm import tqdm
import math
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.options import parse_config
from lib.geometry import *
from lib.sample_util import *
from lib.mesh_util import *
from lib.train_util import find_border
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"



class SyntheticDataset(Dataset):
    def __init__(self, opt, cache_data, cache_data_lock, phase='train'):
        self.opt = opt
        self.projection_mode = 'perspective'
        self.phase = phase
        self.voxel_shape = (128, 128, 128)
        self.is_train = (phase == 'train')

        # Path setup
        self.root = opt.train_dataroot
        self.RENDER = os.path.join(self.root, 'RGB')
        self.PARAM = os.path.join(self.root, 'output_data.npz')
        self.OBJ = os.path.join(self.root, 'Obj')
        self.SMPL = os.path.join(self.root, 'Obj')
        self.DEPTH = os.path.join(self.root, 'Depth')
        self.NORMAL = os.path.join(self.root, 'Normal')
        self.SMPL_NORMAL = os.path.join(self.root, 'smpl_pos')
        self.MASK = os.path.join(self.root, 'Segmentation')
        self.VAL_ROOT = opt.val_dataroot

        if opt.obj_path is not None:
           self.OBJ = os.path.join(self.root,opt.obj_path)
        if opt.smpl_path is not None:
           self.SMPL = os.path.join(self.root, opt.smpl_path)

        self.smpl_faces = readobj(opt.smpl_faces)['f']

        self.load_size = self.opt.loadSize

        self.cameras = self.opt.cameras            
        self.num_views = len(self.cameras)
        self.subjects = self.opt.persons
        self.frames = self.opt.frames

        self.cache_data = cache_data
        self.cache_data_lock = cache_data_lock

        self.num_sample_inout = self.opt.num_sample_inout

        if phase == 'val':
            self.cameras = [i + 1 for i in self.opt.cameras]
            self.frames = [i + 1 for i in self.opt.frames]
            self.num_sample_inout = 100

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

    def __len__(self):
        return len(self.frames) * len(self.subjects)
    
    def clear_cache(self):
        self.cache_data.clear()

    def visibility_sample(self, data, depth, calib, mask=None):
        surface_points = data['surface_points']
        sample_points = data['sample_points']
        inside = data['inside']
        # plt.subplot(121)
        # plt.imshow(depth[0])
        # plt.subplot(122)
        # plt.imshow(depth[1])
        # plt.savefig('show/depth.jpg')
        depth = depth.clone().unsqueeze(1)
        if self.opt.visibility_sample:
            surface_points = torch.FloatTensor(surface_points.T)
            xyz = []
            for view in range(self.num_views):
                xyz.append(perspective(surface_points.unsqueeze(0), calib[view, :, :].unsqueeze(0)).unsqueeze(1))
            xyz = torch.cat(xyz, dim=1)
            pts_xy = xyz[:, :, :2, :]
            pts_z = xyz[0, :, 2:, :]

            pts_depth = []
            for view in range(self.num_views):
                pts_depth.append(
                    index(depth[view, :, :, :].unsqueeze(0), pts_xy[:, view, :, :], size=self.opt.loadSize))
            pts_depth = torch.cat(pts_depth, dim=0)
            pts_depth[pts_depth < 1e-6] = 1e6
            pts_depth = 1 / pts_depth

            pts_visibility = (pts_depth - pts_z) > -0.005
            pts_visibility = (torch.sum(pts_visibility, dim=0) > 0).squeeze(0)

            inside_points = []
            outside_points = []
            # vin = torch.FloatTensor(surface_points.T)[pts_visibility, :]
            # save_samples_truncted_prob('show/vis.ply', vin, np.ones((vin.shape[0], 1)))

            n = self.opt.num_sample_inout
            vis_pts = torch.FloatTensor(sample_points)[:4 * n][pts_visibility[:4 * n], :]
            vis_inside = torch.BoolTensor(inside)[:4 * n][pts_visibility[:4 * n]]
            vin = vis_pts[vis_inside, :]
            vout = vis_pts[torch.logical_not(vis_inside), :]
            if len(vin.shape) > 1:
                vin = vin[torch.randperm(vin.shape[0]), :]
                inside_points.append(vin[:min(self.num_sample_inout // 2 * self.num_views, vin.shape[0])])
            if len(vout.shape) > 1:
                vout = vout[torch.randperm(vout.shape[0]), :]
                outside_points.append(vout[:min(self.num_sample_inout // 2 * self.num_views, vout.shape[0])])

            # save_samples_truncted_prob('show/vis_in.ply', vin, np.ones((vin.shape[0], 1)))
            # save_samples_truncted_prob('show/vis_out.ply', vout, np.zeros((vout.shape[0], 1)))

            n_vis_pts = torch.FloatTensor(sample_points)[2 * n:6 * n][torch.logical_not(pts_visibility[2 * n:6 * n]), :]
            n_vis_inside = torch.BoolTensor(inside)[2 * n:6 * n][torch.logical_not(pts_visibility[2 * n:6 * n])]
            vin = n_vis_pts[n_vis_inside, :]
            vout = n_vis_pts[torch.logical_not(n_vis_inside), :]
            if len(vin.shape) > 1:
                vin = vin[torch.randperm(vin.shape[0]), :]
                inside_points.append(vin[:min(self.num_sample_inout // 2, vin.shape[0])])
            if len(vout.shape) > 1:
                vout = vout[torch.randperm(vout.shape[0]), :]
                outside_points.append(vout[:min(self.num_sample_inout // 2, vout.shape[0])])

            # save_samples_truncted_prob('show/n_vis_in.ply', vin, np.ones((vin.shape[0], 1)))
            # save_samples_truncted_prob('show/n_vis_out.ply', vout, np.zeros((vout.shape[0], 1)))
            # exit(0)

            ran_pts = torch.FloatTensor(sample_points)[6 * n:]
            ran_inside = torch.BoolTensor(inside)[6 * n:]
            vin = ran_pts[ran_inside, :]
            vout = ran_pts[torch.logical_not(ran_inside), :]
            if len(vin.shape) > 1:
                vin = vin[torch.randperm(vin.shape[0]), :]
                inside_points.append(vin[:min(self.num_sample_inout // 2, vin.shape[0])])
            if len(vout.shape) > 1:
                vout = vout[torch.randperm(vout.shape[0]), :]
                outside_points.append(vout[:min(self.num_sample_inout // 2, vout.shape[0])])

            inside_points = torch.cat(inside_points, dim=0)
            outside_points = torch.cat(outside_points, dim=0)
            # samples = inside_points.transpose(0, 1)
            # labels = torch.ones((1, inside_points.shape[0]))
            samples = torch.cat([inside_points, outside_points], dim=0).transpose(0, 1)
            labels = torch.cat([torch.ones((1, inside_points.shape[0])), torch.zeros((1, outside_points.shape[0]))], 1)
            ran_idx = torch.randperm(samples.shape[1])[:n]
            samples = samples[:, ran_idx]
            labels = labels[:, ran_idx]
            # save_samples_truncted_prob('show/samples.ply', samples.numpy().T, labels.numpy().T)
            # exit(0)
        else:
            inside_points = sample_points[inside]
            np.random.shuffle(inside_points)
            outside_points = sample_points[np.logical_not(inside)]
            np.random.shuffle(outside_points)

            nin = inside_points.shape[0]
            inside_points = inside_points[
                            :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
            outside_points = outside_points[
                                :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                                    :(
                                                                                                            self.num_sample_inout - nin)]

            samples = np.concatenate([inside_points, outside_points], 0).T
            labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))],
                                    1)

            samples = torch.Tensor(samples).float()
            labels = torch.Tensor(labels).float()
            if self.opt.debug_data:
                save_samples_truncted_prob('show/samples.ply', samples.numpy().T, labels.numpy().T)
        return {
            'samples': samples,
            'labels': labels,
            'feat_points': data['feat_points']
        }

    def select_sampling_method(self, frame_id, person_id, b_min, b_max):
        person_key = f"{person_id}_{frame_id}"
        if self.cache_data.__contains__(person_key):
            return self.cache_data[person_key]
        # print(person_id, self.cache_data.__len__())
        root_dir = self.OBJ
        # if self.is_train:
        if self.phase != 'inference':
            mesh = trimesh.load(os.path.join(root_dir, f"person_{person_id}", "combined", f'smplx_{str(frame_id).zfill(6)}.obj'))
            if self.opt.coarse_part:
                radius_list = [self.opt.sigma, self.opt.sigma * 2, self.opt.sigma * 4]
            else:
                radius_list = [self.opt.sigma / 3, self.opt.sigma, self.opt.sigma * 2]
            surface_points = np.zeros((6 * self.num_sample_inout, 3))
            sample_points = np.zeros((6 * self.num_sample_inout, 3))
            for i in range(3):
                d = 2 * self.num_sample_inout
                surface_points[i * d:(i + 1) * d, :], _ = trimesh.sample.sample_surface(mesh,
                                                                                        2 * self.num_sample_inout)
                sample_points[i * d:(i + 1) * d, :] = surface_points[i * d:(i + 1) * d, :] + np.random.normal(
                    scale=radius_list[i], size=(2 * self.num_sample_inout, 3))

            # add random points within image space
            length = b_max - b_min
            random_points = np.random.rand(self.num_sample_inout, 3) * length + b_min
            sample_points = np.concatenate([sample_points, random_points], 0)
            inside = mesh.contains(sample_points)

            del mesh
        else:
            sample_points = torch.zeros(1)
            surface_points = torch.zeros(1)
            inside = torch.zeros(1)

        feat_points = torch.zeros(1)

        self.cache_data_lock.acquire()
        self.cache_data[person_key] = {
            'sample_points': sample_points,
            'surface_points': surface_points,
            'inside': inside,
            'feat_points': feat_points
        }
        self.cache_data_lock.release()

        return self.cache_data[person_key]

    def get_norm(self, frame_id, person_id):
        b_min = torch.zeros(3).float()
        b_max = torch.zeros(3).float()
        scale = torch.zeros(1).float()
        center = torch.zeros(3).float()

        t3_mesh = readobj(os.path.join(self.OBJ, f"person_{person_id}", "combined", f'smplx_{str(frame_id).zfill(6)}.obj'))['vi'][:, :3]
        b0 = np.min(t3_mesh, axis=0)
        b1 = np.max(t3_mesh, axis=0)
        center = torch.FloatTensor((b0 + b1) / 2)
        scale = torch.FloatTensor([np.min(1.0 / (b1 - b0)) * 0.9])
        b_min = center - 0.5 / scale
        b_max = center + 0.5 / scale

        normal = np.zeros((3))
        for f in self.smpl_faces:
            a, b, c = t3_mesh[f[0]][0], t3_mesh[f[1]][0], t3_mesh[f[2]][0]
            normal += cross_3d(c - a, b - a)
        del t3_mesh
        if self.opt.flip_normal:
            normal = -normal

        return {
            'b_min': b_min,
            'b_max': b_max,
            'scale': scale,
            'center': center,
            'direction': normal
        }
    
    def pad_to_shape(self, array):
        pad_shape = [(0, s - a) for a, s in zip(array.shape, self.voxel_shape)]
        return np.pad(array, pad_shape, 'constant')
    
    def load_parameters(self, prefix_path):
        parameters = np.load(os.path.join(prefix_path, "output_data.npz"), allow_pickle=True)
        return parameters

    def load_cam_parameters(self, parameters, cam_num):
        intrinsics = np.array(parameters["scene_camera"].item()[f"cam_T_{cam_num}"]["cam_K"]).reshape(3, 3)
        world_to_cam_R = np.array(parameters["scene_camera"].item()[f"cam_T_{cam_num}"]["cam_R_w2c"]).reshape(3, 3)
        d = 90
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.deg2rad(d)), -np.sin(np.deg2rad(d)), 0],
            [0, np.sin(np.deg2rad(d)), np.cos(np.deg2rad(d)), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        world_to_cam_T = np.array(parameters["scene_camera"].item()[f"cam_T_{cam_num}"]["cam_t_w2c"]).reshape(3, 1)
        world_to_cam_matrix = np.concatenate([world_to_cam_R, world_to_cam_T], axis=1)
        # reshaping to 4z4
        world_to_cam_matrix = np.concatenate([world_to_cam_matrix, np.array([[0, 0, 0, 1]])], axis=0)
        world_to_cam_matrix[:3, 3] = world_to_cam_matrix[:3, 3] / 1000
        world_to_cam_matrix = world_to_cam_matrix @ rotation_matrix

        camera_to_world_matrix = np.array(parameters["camera_world"].item()[f"cam_T_{cam_num}"]).reshape(4, 4)

        return intrinsics, world_to_cam_matrix, camera_to_world_matrix
    
    def load_render_data(self, frame_id, person_id):

        calib_list = []
        render_list = []
        extrinsic_list = []
        depth_list = []
        smpl_norm_list = []
        normal_list = []
        mask_list = []
        ero_mask_list = []

        all_cam_parameters = self.load_parameters(self.root)

        for cam_id in self.cameras:
            render_path = os.path.join(self.RENDER, f"cam_{cam_id}", f"r_{frame_id}_{cam_id}_rgb.png")
            depth_path = os.path.join(self.DEPTH, f"cam_{cam_id}", f"r_{frame_id}_{cam_id}_depth_{str(frame_id).zfill(4)}.exr")
            normal_path = os.path.join(self.NORMAL, f"cam_{cam_id}", f"r_{frame_id}_{cam_id}_normal_{str(frame_id).zfill(4)}.exr")
            smpl_norm_path = os.path.join(self.SMPL_NORMAL, f"person_{person_id}", f"cam_{cam_id}", f'r_{frame_id}_{cam_id}_global_smpl.png')
            mask_path = os.path.join(self.MASK,  f"person_{person_id}", f"cam_{cam_id}", f"r_{frame_id}_{cam_id}_segmentation_{str(frame_id).zfill(4)}.png")

            intrinsic, extrinsic, _ = self.load_cam_parameters(all_cam_parameters, cam_id)
            extrinsic = extrinsic[:3, :]

            mask = Image.open(mask_path).convert('RGB')
            render = Image.open(render_path).convert('RGB')
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
            normal = (normal + 1) / 2.0
            normal = (255 * normal).astype(np.uint8)
            normal = Image.fromarray(normal, 'RGB')
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0
            # depth = Image.fromarray(depth)
            depth = Image.fromarray(depth.astype(np.uint8))
            smpl_norm = Image.open(smpl_norm_path)

            imgs_list = [render, depth, normal, mask, smpl_norm]
            if self.opt.flip_x:
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = transforms.RandomHorizontalFlip(p=1.0)(img)
                intrinsic[0, :] *= -1.0
                intrinsic[0, 2] += self.load_size

            if not self.is_train:
                if (not self.opt.no_correct):
                    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
                    x_min, x_max, y_min, y_max = find_border(imgs_list[3])
                    y_min -= 50
                    y_max += 50
                    y_len = y_max - y_min
                    x_min = (x_max + x_min) // 2 - y_len // 2
                    x_max = x_min + y_len
                    scale = 512.0 / y_len

                    fx = fx * scale
                    fy = fy * scale
                    cx = scale * (cx - x_min)
                    cy = scale * (cy - y_min)
                    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = fx, fy, cx, cy
                    depth = transforms.RandomVerticalFlip(p=1.0)(depth)

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = ImageOps.expand(img, pad_size, fill=0)

                w, h = imgs_list[0].size
                th, tw = self.load_size, self.load_size

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    for i, img in enumerate(imgs_list):
                        imgs_list[i] = img.resize((w, h), Image.BILINEAR)
                    intrinsic[0, 0] *= rand_scale
                    intrinsic[1, 1] *= rand_scale

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10)),
                                        int(round((w - tw) / 10)))
                    dy = random.randint(-int(round((h - th) / 10)),
                                        int(round((h - th) / 10)))
                else:
                    dx = 0
                    dy = 0

                intrinsic[0, 2] += -dx
                intrinsic[1, 2] += -dy

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                for i, img in enumerate(imgs_list):
                    imgs_list[i] = img.crop((x1, y1, x1 + tw, y1 + th))

                render, depth, normal, mask, smpl_norm = imgs_list
                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)
            else:
                if not self.is_train and (not self.opt.no_correct):
                    for i, img in enumerate(imgs_list):
                        if i == 4:
                            fill_n = (128, 128, 128)
                        else:
                            fill_n = (0, 0, 0)
                        img = ImageOps.expand(img, 256, fill=fill_n)
                        imgs_list[i] = (img.crop((x_min + 256, y_min + 256, x_max + 256, y_max + 256))).resize(
                            (512, 512), Image.BILINEAR)
                render, depth, normal, mask, smpl_norm = imgs_list

            if self.opt.mask_part:
                mask_draw = ImageDraw.Draw(mask)
                rand_num = np.random.rand()
                if rand_num > 0.75:
                    mask_num = 8
                elif rand_num > 0.25:
                    mask_num = 4
                else:
                    mask_num = 0
                for i in range(mask_num):
                    x, y = np.random.rand() * 512, np.random.rand() * 512
                    w, h = np.random.rand() * 75 + 25, np.random.rand() * 75 + 25
                    mask_draw.rectangle([x, y, x + w, h + y], fill=(0, 0, 0), outline=(0, 0, 0))
                for i in range(mask_num):
                    x, y = np.random.rand() * 512, np.random.rand() * 512
                    w, h = np.random.rand() * 75 + 25, np.random.rand() * 75 + 25
                    mask_draw.ellipse([x, y, x + w, h + y], fill=(0, 0, 0), outline=(0, 0, 0))
                # print(subject)
                # mask.save('metric/attention_mask.png')
                # exit(0)
            ero_mask = mask.filter(MinFilter(3))

            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic[:, :3]).float()

            mask_new = mask.filter(MinFilter(3))
            mask = torch.sum(torch.FloatTensor((np.array(mask).reshape((512, 512, -1)))), dim=2) / 255
            mask[mask > 0] = 1.0
            ero_mask = torch.FloatTensor(np.array(mask).reshape((512, 512, -1)))[:, :, 0] / 255
            render = self.to_tensor(render) * mask.reshape(1, 512, 512)
            normal = self.to_tensor(normal) * mask.reshape(1, 512, 512)
            smpl_norm = self.to_tensor(smpl_norm)

            mask = torch.sum(torch.FloatTensor((np.array(mask_new).reshape((512, 512, -1)))), dim=2) / 255
            mask[mask > 0] = 1.0

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            depth = np.array(depth)
            if len(depth.shape) >= 3:
                depth_list.append(torch.FloatTensor(depth[:, :, 0]) * mask)
            else:
                depth_list.append(torch.FloatTensor(depth) * mask)
            normal_list.append(normal)
            smpl_norm_list.append(smpl_norm)
            mask_list.append(mask.reshape(1, 512, 512))
            ero_mask_list.append(ero_mask.reshape(1, 512, 512))

        return {
            'image': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'depth': torch.stack(depth_list, dim=0),
            'smpl_normal': torch.stack(smpl_norm_list, dim=0),
            'normal': torch.stack(normal_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'ero_mask': torch.stack(ero_mask_list, dim=0)
        }

    def get_item(self, index):
        
        subject_id = index % len(self.subjects)
        frame_id = index // len(self.subjects)

        subject_id = self.subjects[subject_id]
        res = {
            'name': f"Person_{subject_id}",
            'mesh_path': os.path.join(self.OBJ, f"person_{subject_id}", "combined", f'smplx_{str(index).zfill(6)}.obj'),
            'sid': subject_id,
            'phase': self.phase,
        }

        render_data = self.load_render_data(frame_id, subject_id)
        res.update(render_data) 
        norm_parameter = self.get_norm(frame_id, subject_id)
        res.update(norm_parameter)

        start = time.time()
        sample_data = self.select_sampling_method(frame_id, subject_id, res['b_min'].numpy(), res['b_max'].numpy())
        if self.phase != 'inference':
            sample_data = self.visibility_sample(sample_data, res['depth'], res['calib'], res['mask'])
        res.update(sample_data)
        print(f"Time for sampling: {time.time() - start}")

        # Warning dirty fix
        if self.SMPL.endswith('_pred'):
            mesh = trimesh.load(os.path.join(self.SMPL, f'smpl_{str(index).zfill(6)}.obj'))
        else:
            mesh = trimesh.load(os.path.join(self.SMPL, f"person_{subject_id}", "smplx", f'smplx_{str(index).zfill(6)}.obj'))
        res['extrinsic'][0, :, :] = 0
        for i in range(3):
            res['extrinsic'][0, i, i] = 1

        translation = np.zeros((4, 4))
        translation[:3, 3] = -np.array(res['center']) * res['scale'].numpy()
        translation[1, 3] += 0.5
        for i in range(3):
            translation[i, i] = res['scale'].numpy()
        translation[3, 3] = 1
        mesh.apply_transform(translation)

        # center
        transform = np.zeros((4, 4))
        for i in range(4):
            transform[i, i] = 1
        transform[1, 3] = -0.5
        mesh.apply_transform(transform)

        # rotation
        direction = res['direction']
        x, z = direction[0], direction[2]
        theta = math.acos(z / math.sqrt(z * z + x * x))
        if x < 0:
            theta = 2 * math.acos(-1) - theta
        res['extrinsic'][0] = torch.FloatTensor(rotationY(-theta))
        if self.opt.flip_smpl:
            res['extrinsic'][0] = res['extrinsic'][0] @ torch.FloatTensor(rotationX(math.acos(-1)))

        if self.opt.random_rotation:
            pi = math.acos(-1)
            beta = 40 * pi / 180
            rand_rot = np.array(rotationX((np.random.rand() - 0.5) * beta)) @ np.array(
                rotationY((np.random.rand() - 0.5) * beta)) @ np.array(rotationZ((np.random.rand() - 0.5) * beta))
            res['extrinsic'][0] = torch.FloatTensor(rand_rot) @ res['extrinsic'][0]

        rotation = np.zeros((4, 4))
        rotation[3, 3] = 1
        rotation[:3, :3] = res['extrinsic'][0]
        mesh.apply_transform(rotation)

        transform[1, 3] = 0.5
        mesh.apply_transform(transform)
        bbox = mesh.bounds
        bbox_size = np.max(bbox[1] - bbox[0])  # size of the bounding box
        pitch = bbox_size / (128 - 1)  # size of each voxel
        vox = mesh.voxelized(pitch=pitch)
        vox.fill()

        voxel_grid = self.pad_to_shape(vox.matrix)
        res['vox'] = torch.FloatTensor(voxel_grid).unsqueeze(0)

        if self.opt.debug_data:
            for num_view_i in range(self.num_views):
                img = np.uint8(
                    (np.transpose(render_data['image'][num_view_i][0:3, :, :].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
                img = np.array(img, dtype=np.uint8).copy()
                calib = render_data['calib'][num_view_i]
                pts = torch.FloatTensor(res['samples'][:, res['labels'][0] > 0.5]) # [3, N]
                # pts = res['samples']
                # pts = torch.FloatTensor(res['feat_points'])
                print(pts)
                pts = perspective(pts.unsqueeze(0), calib.unsqueeze(0)).squeeze(0).transpose(0, 1)
                for p in pts:
                    img = cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
                cv2.imwrite(f'show/data_set_2d_rendered_{index}_{num_view_i}.jpg', img)

        return res

    def __getitem__(self, index):
        return self.get_item(index)


# get options
opt = parse_config()
if __name__=='__main__':
    data = SyntheticDataset(opt, phase='train', num_views=4)
    print(data[0]['name'])

