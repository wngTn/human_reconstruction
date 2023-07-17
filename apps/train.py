import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from datetime import datetime

from lib.options import parse_config
from lib.sample_util import *
from lib.train_util import *
from lib.data import SyntheticDataset
from lib.model import DMCNet, NormalNet
from lib.solver import Solver
from multiprocessing import Process, Manager, Lock

def init_paths(opt):
    opt.exp_name = f"{datetime.now().strftime('%m_%d-%H_%M_%S')}_{str(opt.name).upper()}"
    opt.checkpoints_path = f"outputs/{opt.exp_name}/checkpoints"
    opt.train_results_path = f"outputs/{opt.exp_name}/train_results"
    opt.val_results_path = f"outputs/{opt.exp_name}/val_results"
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.train_results_path, exist_ok=True)
    os.makedirs(opt.val_results_path, exist_ok=True)
    with open(f"outputs/{opt.exp_name}/configs.json", "w") as f:
        args_dict = vars(opt)
        json.dump(args_dict, f, indent=4)

def train():
    opt = parse_config()
    init_paths(opt)

    netG = DMCNet(opt, projection_mode='perspective').to(device)
    netN = NormalNet().to(device)

    if device.type == 'cuda':
        gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
        netG = DataParallel(netG, device_ids=gpu_ids)
        netN = DataParallel(netN, device_ids=gpu_ids)
    else:
        netG = DataParallel(netG)
        netN = DataParallel(netN)

    train_dataset = SyntheticDataset(opt, phase='train')
    val_dataset = SyntheticDataset(opt, phase='val')

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size,
                                   shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads,
                                   pin_memory=opt.pin_memory)
    print('train data size: ', len(train_data_loader))

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=not opt.serial_batches,
                                 num_workers=opt.num_threads,
                                 pin_memory=opt.pin_memory)
    print('val data size: ', len(val_data_loader))

    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate)

    solver = Solver(netN, netG, opt, train_data_loader, val_data_loader, optimizerG, device)
    print("Start training...")
    solver(opt.num_epoch)




if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)
    train()