import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from pathlib import Path
import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from lib.options import parse_config
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import SyntheticDataset
from lib.model import *

import trimesh
import trimesh.proximity
import trimesh.sample

# get options
opt = parse_config()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(opt):
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    log = SummaryWriter(opt.log_path)
    total_iteration = 0
    netG = DMCNet(opt, projection_mode='perspective').to(device)
    netN = NormalNet().to(device)
    print('Using Network: ', netG.name, netN.name)
    if device.type == 'cuda':
        gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
        netG = DataParallel(netG, device_ids=gpu_ids)
        netN = DataParallel(netN, device_ids=gpu_ids)
    else:
        netG = DataParallel(netG)
        netN = DataParallel(netN)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate)
    lr = opt.learning_rate

    def set_train():
        netG.train()
    
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=device), strict=False)
    
    if opt.load_netN_checkpoint_path is not None:
        print('loading for net N ...', opt.load_netN_checkpoint_path)
        netN.load_state_dict(torch.load(opt.load_netN_checkpoint_path, map_location=device), strict=False)
    
    print("loaded finished!")
    
    train_dataset = SyntheticDataset(opt, phase='train', num_views=4)
    val_dataset = SyntheticDataset(opt, phase='validation', num_views=4)
    # test_dataset = SyntheticDataset(opt, phase='test', num_views=4)
        
    projection_mode = train_dataset.projection_mode
    print('projection_mode:', projection_mode)
    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_data_loader))

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=1, shuffle=False,
                                 num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('val data size: ', len(val_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    # test_data_loader = DataLoader(test_dataset,
    #                               batch_size=1, shuffle=False,
    #                               num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    # print('test data size: ', len(test_data_loader))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.json')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # training
    start_epoch = 0
    print("start training......")

    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()
        set_train()
        iter_data_time = time.time()
        # np.random.seed(int(time.time()))
        # random.seed(int(time.time()))
        # torch.manual_seed(int(time.time()))
        train_bar = tqdm(enumerate(train_data_loader))
        save_path = Path(opt.results_path) / opt.name / str(epoch)
        save_path.mkdir(parents=True, exist_ok=True)
        for train_idx, train_data in train_bar:
            if train_idx >= 1:
                break
            total_iteration += 1
            iter_start_time = time.time()
            # retrieve the data
            for key in train_data:
                if torch.is_tensor(train_data[key]):
                    train_data[key] = train_data[key].to(device=device)

            # predict normal
            with torch.no_grad():
                net_normal = netN.forward(train_data['image'])
                net_normal = net_normal * train_data['mask']
               
            train_data['normal'] = net_normal.detach()
            res, error = netG.forward(train_data)
            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            log.add_scalar('loss', error.item(), total_iteration)

            if train_idx % opt.val_step == 0:
                print("Performing Validation Now")
                val_loss = validate(netG, netN, val_data_loader, train_idx + (epoch * len(train_dataset)))
                # log.add_scalar('val_loss', val_loss, epoch)
                print('Current val loss: ', val_loss)

            if train_idx % opt.freq_plot == 0:
                descrip = 'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                    opt.name, epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma,
                    iter_start_time - iter_data_time,
                    iter_net_time - iter_start_time, int(eta // 60),
                    int(eta - 60 * (eta // 60)))
                train_bar.set_description(descrip)

            if train_idx % opt.freq_save == 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
                torch.save(optimizerG.state_dict(), '%s/%s/optim_latest' % (opt.checkpoints_path, opt.name))
                torch.save(optimizerG.state_dict(), '%s/%s/optim_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            # if train_idx % opt.freq_save_ply == 0:
            #     ply_save_path = Path(save_path) / f"{train_idx}.ply"
            #     r = res[0].cpu()
            #     points = train_data['samples'][0].transpose(0, 1).cpu()
            #     save_samples_truncted_prob(ply_save_path, points.detach().numpy(), r.detach().numpy())

            iter_data_time = time.time()
        
        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, [5, 10, 25], 0.1)
        # train_dataset.clear_cache()

    log.close()

def validate(netG, netN, val_data_loader, train_idx):
        total_loss = 0.0
        num_samples = 20
        total_err = 0

        with torch.no_grad():
            netG.eval()
            netN.eval()
            
            for i, val_data in tqdm(enumerate(val_data_loader), total=20):
                if i >= 1:
                    break
                for key in val_data:
                    if torch.is_tensor(val_data[key]):
                        val_data[key] = val_data[key].to(device=device)

                # predict normal
                net_normal = netN.forward(val_data['image'])
                net_normal = net_normal * val_data['mask']
                
                val_data['normal'] = net_normal.detach()
                res, error = netG.forward(val_data)

                ply_save_path = Path("val_vis") / f"pred_{train_idx + i}.ply"
                r = res[0].cpu()
                points = val_data['samples'][0].transpose(0, 1).cpu()
                save_samples_truncted_prob(ply_save_path, points.detach().numpy(), r.detach().numpy())

                ply_gt_path = Path("val_vis") / f"gt_{train_idx + i}.ply"
                points = val_data['samples'][0].transpose(0, 1)
                save_samples_truncted_prob(ply_gt_path, points.cpu().detach().numpy(), val_data['labels'][0].cpu().detach().numpy())

                # print("Saved thign for,", train_idx + i)    

                total_err += error               
                    
                    # resolution = val_data['vox'].shape[-1]
                    # coords, _ = create_grid(resolution, resolution, resolution, val_data['b_min'], val_data['b_max'])
                    # def eval_func(points):
                    #     points = np.expand_dims(points, axis=0)
                    #     samples = torch.from_numpy(points).to(device).float()
                    #     netG.query(samples, val_data['calib'], val_data['extrinsic'])
                    #     pred = netG.get_preds()[0][0]
                    #     return pred.detach().cpu().numpy()
                    # if oct.use_octree:
                    #     sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
                    # else:
                    #     sdf = eval_grid(coords, eval_func, num_samples=num_samples)

                    # pts, _ = trimesh.sample.sample_surface(, num_samples)
                    # _, src_tgt_dist, _ = trimesh.proximity.closest_point(val_data['mesh_path'][0], pts)
                    # src_tgt_dist[np.isnan(src_tgt_dist)] = 0
                    # src_tgt_dist[~np.isfinite(src_tgt_dist)] = 0        
                    # p2s_dist = src_tgt_dist.mean()

                    # total_loss += p2s_dist

                # # save_path = '%s/%s/val_inference_%s_%d.obj' % (opt.results_path, opt.name, val_data['name'], i)
                # save_root = os.path.join(opt.results_path, opt.name)        #, 'epoch_%s' % epoch
                # save_path = os.path.joint(save_root, 'val_inference_%d.obj' % (train_idx + i))
                # image_tensor = val_data['image'].to(device).unsqueeze(0)
                # mask_tensor = val_data['mask'].to(device).unsqueeze(0)
                # res = netN.forward(image_tensor)
                # res = res * mask_tensor
                # val_data['normal'] = res[0]
                # print('Saving val mesh to ' + save_path)
                # gen_mesh_dmc(opt, netG, device, val_data, save_path, use_octree=True)

                # target_root = os.path.join(opt.dataroot, 'Obj')
                # chamfer = evaluation(save_root, target_root)
                # total_loss += chamfer

        return total_err 
# / 20


if __name__ == '__main__':
    train(opt)