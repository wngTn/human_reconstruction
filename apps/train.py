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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from lib.options import parse_config
from lib.mesh_util import gen_validation
from lib.sample_util import *
from lib.train_util import *
from lib.data import SyntheticDataset
from lib.model import *
from multiprocessing import Process, Manager, Lock

from torch.cuda.amp import autocast as autocast

def init_paths(opt):
    opt.exp_name = f"{datetime.now().strftime('%m_%d-%H_%M_%S')}_{str(opt.name).capitalize()}"
    opt.checkpoints_path = f"outputs/{opt.exp_name}/checkpoints"
    opt.train_results_path = f"outputs/{opt.exp_name}/train_results"
    opt.val_results_path = f"outputs/{opt.exp_name}/val_results"
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.train_results_path, exist_ok=True)
    os.makedirs(opt.val_results_path, exist_ok=True)
    with open(f"outputs/{opt.exp_name}/configs.json", "w") as f:
        args_dict = vars(opt)
        json.dump(args_dict, f, indent=4)


def train(opt):
    # np.random.seed(int(time.time()))
    # random.seed(int(time.time()))
    # torch.manual_seed(int(time.time()))
    log = SummaryWriter(os.path.join("outputs", opt.exp_name, "logs"))
    netG = DMCNet(opt, projection_mode='perspective').to(device)
    netN = NormalNet().to(device)
    
    if device.type == 'cuda':
        gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
        netG = DataParallel(netG, device_ids=gpu_ids)
        netN = DataParallel(netN, device_ids=gpu_ids)
    else:
        netG = DataParallel(netG)
        netN = DataParallel(netN)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate)
    lr = opt.learning_rate
    
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=device), strict=False)
    
    if opt.load_netN_checkpoint_path is not None:
        print('loading for net N ...', opt.load_netN_checkpoint_path)
        netN.load_state_dict(torch.load(opt.load_netN_checkpoint_path, map_location=device), strict=False)
    
    
    train_dataset = SyntheticDataset(opt, cache_data=Manager().dict(), cache_data_lock=Lock(), phase='train')
    val_dataset = SyntheticDataset(opt, cache_data=Manager().dict(), cache_data_lock=Lock(), phase='val')
        
    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_data_loader))

    val_data_loader = DataLoader(val_dataset,
                                    batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                    num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('val data size: ', len(val_data_loader))

    # training
    print("start training......")

    current_iteration = 0

    # Define a list to store the last 10 error values
    errors = []
    total_iteration = int(opt.num_epoch * (len(train_data_loader) // opt.batch_size))

    for epoch in range(opt.num_epoch):
        netG.train()
        iter_data_time = time.time()
        
        for train_idx, train_data in tqdm(enumerate(train_data_loader), total = len(train_data_loader)):
            current_iteration += 1
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

            del net_normal
            
            # with autocast():
            netG.training = True
            res, error = netG.forward(train_data)
            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            iter_end_time = time.time()

            log.add_scalar('loss', error.item(), current_iteration)

            errors.append(error.item())
            # If the list has more than 10 items, remove the first one
            if len(errors) > 10:
                errors.pop(0)
            mean_error = sum(errors) / len(errors)

            elapsed_time_per_iteration = iter_end_time - iter_start_time
            remaining_iterations = total_iteration - current_iteration
            eta_seconds = elapsed_time_per_iteration * remaining_iterations

            # del train_data
            torch.cuda.empty_cache()

            if train_idx % opt.freq_plot == 0:
                descrip = (
                    f'Name: {opt.name}\n'
                    f'Epoch: {epoch}\n'
                    f'Iteration: {current_iteration}/{total_iteration}\n'
                    f'Err: {mean_error:.06f} (Over last 10 iterations)\n'
                    f'LR: {lr:.06f}\n'
                    f'Data Time: {iter_start_time - iter_data_time:.05f}\n'
                    f'Network Time: {iter_end_time - iter_start_time:.05f}\n'
                    f'ETA: {int(eta_seconds // 3600)} hours, {int((eta_seconds % 3600) // 60)} minutes'
                )
    
                print(descrip)

            if train_idx % opt.freq_save == 0:
                print(f"Saving checkpoints at iteration: {current_iteration}")
                torch.save(netG.state_dict(), f'{opt.checkpoints_path}/netG_latest')
                torch.save(netG.state_dict(), f'{opt.checkpoints_path}/netG_it_{current_iteration}')
                torch.save(optimizerG.state_dict(), f'{opt.checkpoints_path}/optim_latest')
                torch.save(optimizerG.state_dict(), f'{opt.checkpoints_path}/optim_it_{current_iteration}')

            # if train_idx % opt.freq_save_ply == 0:
            #     ply_save_path = os.path.join(opt.train_results_path, f"{epoch}_{current_iteration}.ply")
            #     r = res[0].cpu()
            #     points = train_data['samples'][0].transpose(0, 1).cpu()
            #     save_samples_truncted_prob(ply_save_path, points.detach().numpy(), r.detach().numpy())
            #     print(f"Saving train ply in {ply_save_path}")
            #     del r
            #     del res
            #     del points

            iter_data_time = time.time()

        if epoch != 0 and epoch % opt.freq_val == 0:
            print("Performing Validation Now")
            val_loss = validate(opt, netG, netN, val_data_loader, epoch)
            log.add_scalar('val_loss', val_loss, epoch)
            print('Current val loss: ', val_loss)

        
        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, [5, 10, 25], 0.1)
        train_dataset.clear_cache()

    del train_data

    log.close()

def validate(opt, netG, netN, val_data_loader, epoch):
    test_netG = netG.module
    test_netN = netN.module
    test_netG.eval()
    test_netN.eval()
    mean_error = 0

    for i, val_data in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
        val_save_path = os.path.join(opt.val_results_path, f"{epoch}_{i+1}.obj")

        if i >= 4:
            break

        for key in val_data:
            if torch.is_tensor(val_data[key]):
                val_data[key] = val_data[key].to(device=device)

        with torch.no_grad():
            net_normal = netN.forward(val_data['image'])
            net_normal = net_normal * val_data['mask']
            
        val_data['normal'] = net_normal.detach()

        del net_normal

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
                gen_validation(opt, test_netG, device, val_data, epoch, i, threshold=0.5, use_octree=True)

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
    



    

def main():
    opt = parse_config()
    init_paths(opt)
    train(opt) 

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)
    torch.cuda.empty_cache()
    main()