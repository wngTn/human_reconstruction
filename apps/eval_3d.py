import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import time
from datetime import datetime
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
from multiprocessing import Process, Manager, Lock

from lib.options import parse_config
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

from apps.evaluator import Evaluator

def compute_metrics(opt):
    
    output_dir = os.path.join("outputs", opt.folder)

    val_frames = opt.val_frames
    pred_paths = []
    gt_paths = []
    human_paths = []
    cloth_paths = []

    for val_frame in val_frames:
        pred_paths.append(os.path.join(output_dir, "meshes", f"pred_{val_frame:04}.obj"))
        gt_paths.append(os.path.join(opt.val_dataroot, "Obj", "person_0", "combined", f"smplx_{val_frame:06}.obj"))
        human_paths.append(os.path.join(opt.val_dataroot, "Obj", "person_0", "smplx_no_cloth", f"smplx_{val_frame:04}.obj"))
        cloth_paths.append(os.path.join(opt.val_dataroot, "Obj", "person_0", "cloth", f"cloth_{val_frame:04}.obj"))

    evaluator = Evaluator(pred_paths=pred_paths,
                            gt_paths=opt.gt_paths,
                            num_samples=5000,
                            save_pcd=True,
                            folder=output_dir,
                            human_paths=human_paths,
                            cloth_paths=cloth_paths)

    evaluator.get_chamfer_distance()

    

def reconstruct(opt):

    netG = DMCNet(opt, projection_mode='perspective').to(device=device)
    netN = NormalNet().to(device)
    print('Using Network: ', netG.name)

    netG = DataParallel(netG)
    netN = DataParallel(netN)

    output_dir = os.path.join("outputs", opt.folder)
    save_dir = os.path.join(output_dir, 'meshes')
    
    # load checkpoints
    netG_checkpoint_path = os.path.join(output_dir, 'checkpoints', 'netG_latest.pth')
    print('loading for net G ...', netG_checkpoint_path)
    netG.load_state_dict(torch.load(netG_checkpoint_path, map_location=device), strict=True)

    print('loading for net N ...', opt.load_netN_checkpoint_path)
    netN.load_state_dict(torch.load(opt.load_netN_checkpoint_path, map_location=device), strict=True)

    print("loaded finished!")

    test_netG = netG.module
    netN = netN.module
    dataset = SyntheticDataset(opt, phase='test')
    print(f"Test Set Length: dataset.__len__()")

    with torch.no_grad():
        test_netG.eval()
        print('Generating meshes...')

        for i in tqdm(range(len(dataset))):
            test_data = dataset[i]

            save_path = os.path.join(save_dir, f"pred_{str(test_data['frame_id']).zfill(4)}.obj")

            image_tensor = test_data['image'].to(device=device).unsqueeze(0)
            mask_tensor = test_data['mask'].to(device=device).unsqueeze(0)
            res = netN.forward(image_tensor)
            res = res * mask_tensor
            # test_data['normal'] = torch.nn.Upsample(size=[1024, 1024], mode='bilinear')(res[0])
            test_data['normal'] = res[0]
            gen_mesh_dmc(opt, test_netG, device, test_data, save_path, threshold=opt.mc_threshold, use_octree=True)


def evaluate(opt):
    if len(os.listdir(os.path.join("outputs", opt.folder, "meshes"))) < len(opt.val_frames):
        print("No meshes found. Reconstructing meshes...")
        reconstruct(opt)
    compute_metrics(opt)





if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    # get options
    opt = parse_config()
    evaluate(opt)
