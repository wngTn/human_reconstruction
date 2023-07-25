from skimage import measure
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import trimesh
import os
import open3d as o3d

from .sdf import create_grid, eval_grid_octree, eval_grid
from .net_util import reshape_sample_tensor
from .geometry import index

def gen_validation(opt, net, device, data, epoch, iteration, use_octree=True, threshold=0.5):
    image_tensor = data['image'].squeeze(0)
    calib_tensor = data['calib'].squeeze(0)
    extrinsic = data['extrinsic']
    vox_tensor = data['vox']
    smpl_normal = data['smpl_normal']
    save_smpl_normal = smpl_normal.clone()
    normal_tensor = data['normal'].squeeze(0)
    scale, center = data['scale'], data['center']
    mask, ero_mask = data['mask'], data['ero_mask']
    # labels = data['labels']
    # pts = data['samples']

    net.mask_init(mask, ero_mask)   # --> unsqueeze(0)
    net.norm_init(scale, center)
    net.smpl_init(smpl_normal)      # --> unsqueeze(0)
    
    net.filter2d(torch.cat([image_tensor.unsqueeze(0), smpl_normal], dim=2))
    if opt.fine_part:
        if normal_tensor.shape[2] == 1024:
            print('1024')
            smpl_normal = torch.nn.Upsample(size=[1024, 1024], mode='bilinear')(smpl_normal.squeeze(0)).unsqueeze(0)
        net.filter_normal(torch.cat([normal_tensor.unsqueeze(0), smpl_normal], dim=2))
    
    net.filter3d(vox_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = os.path.join(opt.val_results_path, f"{epoch}_{iteration}.png") # save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_smpl = (np.transpose(save_smpl_normal[0][v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img / 2 + save_smpl / 2)
    for v in range(normal_tensor.shape[0]):
        save_nm = normal_tensor[v]
        save_nm = F.interpolate(save_nm.unsqueeze(0), size=[512, 512], mode='bilinear')[0]
        save_nm = (np.transpose(save_nm.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_nm)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    try:
        verts, faces, _, _ = reconstruction_3d(
            net, device, calib_tensor.unsqueeze(0), extrinsic, opt.resolution, np.array(b_min.squeeze(0).cpu()), np.array(b_max.squeeze(0).cpu()), use_octree=use_octree, threshold=threshold)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=device).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor.squeeze(0)[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(os.path.join(opt.val_results_path, f"{epoch}_{iteration+1}.obj"), verts, faces, color)
        print('Saved to ' + os.path.join(opt.val_results_path, f"{epoch}_{iteration+1}.obj"))
    except Exception as e:
        print("Yo, something went wrong", e)


def gen_mesh_dmc(opt, net, cuda, data, save_path, use_octree=True, threshold=0.5):
    image_tensor = data['image'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    extrinsic = data['extrinsic'].to(device=cuda).unsqueeze(0)
    vox_tensor = data['vox'].to(device=cuda).unsqueeze(0)
    smpl_normal = data['smpl_normal'].to(device=cuda).unsqueeze(0)
    save_smpl_normal = smpl_normal.clone()
    normal_tensor = data['normal'].to(device=cuda)
    scale, center = data['scale'].to(device=cuda).unsqueeze(0), data['center'].to(device=cuda).unsqueeze(0)
    mask, ero_mask = data['mask'].to(device=cuda).unsqueeze(0), data['ero_mask'].to(device=cuda).unsqueeze(0)

    net.mask_init(mask, ero_mask)
    net.norm_init(scale, center)
    net.smpl_init(smpl_normal)
    
    net.filter2d(torch.cat([image_tensor.unsqueeze(0), smpl_normal], dim=2))
    if opt.fine_part:
        if normal_tensor.shape[2] == 1024:
            print('1024')
            smpl_normal = torch.nn.Upsample(size=[1024, 1024], mode='bilinear')(smpl_normal.squeeze(0)).unsqueeze(0)
        net.filter_normal(torch.cat([normal_tensor.unsqueeze(0), smpl_normal], dim=2))
    
    net.filter3d(vox_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_smpl = (np.transpose(save_smpl_normal[0][v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img / 2 + save_smpl / 2)
    for v in range(normal_tensor.shape[0]):
        save_nm = normal_tensor[v]
        save_nm = F.interpolate(save_nm.unsqueeze(0), size=[512, 512], mode='bilinear')[0]
        save_nm = (np.transpose(save_nm.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_nm)
        # save_nm = smpl_normal[0:, v]
        # save_nm = F.interpolate(save_nm, size=[512, 512], mode='bilinear')[0]
        # save_nm = (np.transpose(save_nm.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        # save_img_list.append(save_nm)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    try:
        point_cloud_save_path = save_path[:-4] + '.ply'
        verts, faces, _, _ = reconstruction_3d(
            net, cuda, calib_tensor.unsqueeze(0), extrinsic, opt.resolution, b_min, b_max, point_cloud_save_path,use_octree=use_octree, threshold=threshold)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print("Yo, something went wrong", e)

def grid_to_point_cloud(sdf):
    # Asserting the shape of the sdf
    assert sdf.shape == (512, 512, 512)

    # Generate a grid of coordinates
    x, y, z = np.meshgrid(np.arange(sdf.shape[0]), 
                          np.arange(sdf.shape[1]), 
                          np.arange(sdf.shape[2]), 
                          indexing='ij')

    # Flatten the coordinate grids and the sdf to 1D arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    sdf_flat = sdf.flatten()

    # Each grid cell corresponds to a point in the point cloud
    points = np.column_stack((x, y, z))

    # The color is a linear blend of green and red based on the sdf value
    red = sdf_flat
    green = 1.0 - sdf_flat

    # Colors in open3d are in the range [0, 1], so we need to scale them
    colors = np.column_stack((red, green, np.zeros_like(red)))  # No blue component

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the points and colors to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def reconstruction_3d(net, cuda, calib_tensor, extrinsic, 
                   resolution, b_min, b_max, pcd_save_path,
                   net_3d=False, use_octree=False, num_samples=30000, threshold=0.5, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    #print(b_min, b_max)
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    # print(coords.shape, mat.shape)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        # print(points.shape)
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor, extrinsic)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    #try:
    # print("The highest probability for SDF was,", sdf.max())
    # point_cloud = grid_to_point_cloud(sdf)
    # o3d.io.write_point_cloud(pcd_save_path, point_cloud)
    verts, faces, normals, values = measure._marching_cubes_lewiner.marching_cubes(sdf, threshold)
    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    return verts, faces, normals, values

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors, reverse=False):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        if reverse:
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
        else:
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()



def _append(faces, indices):
    if len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) == 3:
        faces.append(indices)
    else:
        assert False, len(indices)


def readobj(path, scale=1):
    vi = []
    vt = []
    vn = []
    faces = []

    with open(path, 'r') as myfile:
        lines = myfile.readlines()

    # cache vertices
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == 'v':
            vi.append(fields)
        elif type == 'vt':
            vt.append(fields)
        elif type == 'vn':
            vn.append(fields)

    # cache faces
    for line in lines:
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        # line looks like 'f 5/1/1 1/2/1 4/3/1'
        # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
        if type != 'f':
            continue

        # a field should look like '5/1/1'
        # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
        # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
        indices = [[int(_) - 1 if _ != '' else 0 for _ in field.split('/')] for field in fields]

        if len(indices) == 4:
            faces.append([indices[0], indices[1], indices[2]])
            faces.append([indices[2], indices[3], indices[0]])
        elif len(indices) == 3:
            faces.append(indices)
        else:
            assert False, len(indices)

    ret = {}
    ret['vi'] = None if len(vi) == 0 else np.array(vi).astype(np.float32) * scale
    ret['vt'] = None if len(vt) == 0 else np.array(vt).astype(np.float32)
    ret['vn'] = None if len(vn) == 0 else np.array(vn).astype(np.float32)
    ret['f'] = None if len(faces) == 0 else np.array(faces).astype(np.int32)
    return ret

