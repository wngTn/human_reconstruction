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
from apps import evaluation

# get options
opt = parse_config()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(opt):
    # np.random.seed(int(time.time()))
    # random.seed(int(time.time()))
    # torch.manual_seed(int(time.time()))
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
    val_dataset = SyntheticDataset(opt, phase='val', num_views=4)
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
                                    num_workers=1, pin_memory=opt.pin_memory)
    print('val data size: ', len(val_data_loader))


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

            if train_idx % opt.freq_save_ply == 0:
                ply_save_path = Path(save_path) / f"{train_idx}.ply"
                r = res[0].cpu()
                points = train_data['samples'][0].transpose(0, 1).cpu()
                save_samples_truncted_prob(ply_save_path, points.detach().numpy(), r.detach().numpy())

            iter_data_time = time.time()

        if epoch!=0 and epoch%opt.freq_val == 0:
            print('Val now:')
            val_loss = validate(opt, netG, netN, val_data_loader, epoch)
            log.add_scalar('val_loss', val_loss, epoch)
            print('Current val loss: ', val_loss)
        # log.add_scalar('val_loss'. val_loss, epoch)
        
        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, [5, 10, 25], 0.1)
        torch.cuda.empty_cache()

    log.close()

def validate(opt, netG, netN, val_data_loader, epoch):
    test_netG = netG.module
    test_netN = netN.module
    test_netG.eval()
    test_netN.eval()

    mean_error = 0

    for i, val_data in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
        # val_save_path = os.path.join(opt.val_results_path, f"{epoch}_{i+1}.obj")
        val_save_dir = os.path.join("F:/SS23/AT3DCV/at3dcv_project/results/synthetic_first_trial_overfitting_coarse", "val_results")
        os.makedirs(val_save_dir, exist_ok=True)
        val_save_path = os.path.join(val_save_dir, f"{epoch}_{i}.obj")    
        if i >= 4:
            break

        for key in val_data:
            if torch.is_tensor(val_data[key]):
                val_data[key] = val_data[key].to(device=device)

        with torch.no_grad():
            net_normal = netN.forward(val_data['image'])
            net_normal = net_normal * val_data['mask']
            
        val_data['normal'] = net_normal.detach()

        with torch.no_grad():

            if opt.val_type == 'mse':
                res, error = test_netG.forward(val_data)
                error = error.item()
                # print(type(error), error)
                mean_error = (mean_error * i + error) / (i + 1)
                error = mean_error
            else:
                print('Generating mesh (inference) ... ')
                test_netG.training = False
                gen_validation(opt, test_netG, device, val_data, epoch, i, val_save_dir, threshold=0.5, use_octree=True)

                # val p2s error:
                num_samples = 10000
                src_mesh = trimesh.load(val_save_path)
                # tgt_mesh = trimesh.load(os.path.join(val_data['OBJ'], f"person_{person_id}", "combined", f'smplx_{str(frame_id).zfill(6)}.obj'))
                # tgt_mesh = trimesh.load(os.path.join(val_data['OBJ'], f"person_0", "combined", f'smplx_{str(i+1).zfill(6)}.obj'))
                tgt_mesh = trimesh.load(val_data['mesh_path'][0])
                src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)
                _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
                src_tgt_dist[np.isnan(src_tgt_dist)] = 0
                src_tgt_dist[~np.isfinite(src_tgt_dist)] = 0
                p2s_error = src_tgt_dist.mean()
                error = p2s_error
                print('p2s error = ', p2s_error)

        # if i % opt.freq_save_ply == 0:
        #     ply_save_path = os.path.join(opt.val_results_path, f"{epoch}_{i}.ply")
        #     r = res[0].cpu()
        #     points = val_data['samples'][0].transpose(0, 1).cpu()
        #     del val_data
        #     save_samples_truncted_prob(ply_save_path, points.detach().numpy(), r.detach().numpy())
        #     print(f"Saving val ply in {ply_save_path}")
        #     del r
        #     del res
        #     del points
    return error

if __name__ == '__main__':
    train(opt)