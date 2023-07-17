import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from datetime import datetime
from lib.sample_util import save_samples_truncted_prob
from skimage import measure
from lib.mesh_util import save_obj_mesh_with_color

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[info] learning_rate: {lr}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
[info] data_time: {data_time}s
"""

VAL_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[val]   val_loss: {val_loss}
[val]   chamfer_dist: {chamfer_dist}
"""

class Solver():
    def __init__(self, netN, netG, opt, train_dataloader, val_dataloader, optimizer, device):

        self.epoch = 0                    # set in __call__
        self.opt = opt
        self.netN = netN 
        self.netG = netG
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer

        self.output_dir = os.path.join("outputs", opt.exp_name)
        self.device = device

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "chamfer_dist": float("inf")
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }

        # tensorboard
        os.makedirs(train_tb_out:=os.path.join(self.output_dir, "tensorboard", "train"), exist_ok=True)
        os.makedirs(val_tb_out:=os.path.join(self.output_dir, "tensorboard","val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(train_tb_out),
            "val": SummaryWriter(val_tb_out)
        }

        # training log
        log_path = os.path.join(self.output_dir, "log.txt")
        self.log_fout = open(log_path, "a")

        eval_path = os.path.join(self.output_dir, "eval.txt")
        self.eval_fout = open(eval_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE 
        self.__val_report_template = VAL_REPORT_TEMPLATE

        self.lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_epoch, eta_min=0.000001)


    def __call__(self, epoch):
        # setting
        self.epoch = epoch
        self._total_iter["train"] = len(self.train_dataloader) * epoch
        self._total_iter["val"] = len(self.val_dataloader) * (epoch // self.opt.freq_val)

        for epoch_id in range(epoch):
            self._log("epoch {} starting...".format(epoch_id + 1))

            if (epoch_id + 1) % self.opt.freq_val == 0 and epoch_id > 0:
                self._eval_feed(self.val_dataloader, epoch_id)
            
            # feed 
            self._train_feed(self.train_dataloader, epoch_id)

            # update lr scheduler
            if self.lr_scheduler:
                print("update learning rate --> {}\n".format(set(self.lr_scheduler.get_lr())))
                self.lr_scheduler.step()
                

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str, flush=True)

    def _log_eval(self, info_str):
        self.eval_fout.write(info_str + "\n")
        self.eval_fout.flush()
        print(info_str, flush=True)

    def _forward(self, data_dict):
        with torch.no_grad():
            net_normal = self.netN(data_dict['image'])
            net_normal = net_normal * data_dict['mask']
        data_dict['normal'] = net_normal.detach()
        data_dict = self.netG(data_dict)
        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def create_voxelgrid(self, points, probabilities, grid_size):
        # compute the bounds of the points
        min_bound = points.min(0)
        max_bound = points.max(0)

        # compute the size of the grid
        grid_shape = np.ceil((max_bound - min_bound) / grid_size).astype(int)

        # initialize the voxel grid with zeros
        voxelgrid = np.zeros(grid_shape)

        # for each point, compute the corresponding cell and add the probability to it
        for point, prob in zip(points, probabilities):
            index = ((point - min_bound) / grid_size).astype(int)
            voxelgrid[tuple(index)] += prob

        return voxelgrid

    def _eval_feed(self, dataloader, epoch_id):
        self.netG.eval()
        self.netN.eval()

        self._reset_log("val")
        dataloader = tqdm(dataloader, desc="eval", file=sys.stdout)
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].to(device=self.device)

            # initialize the running loss
            self._running_log = {
                "loss": 0,
                "chamfer_dist": 0,
            }
            # forward
            with torch.no_grad():
                data_dict = self._forward(data_dict)
            frame_id = data_dict['frame_id'][0]
            ply_save_path = os.path.join(self.opt.val_results_path, f"{epoch_id}__frame_{frame_id}.ply")
            r = data_dict["preds"].cpu()
            points = data_dict['samples'][0].transpose(0, 1).cpu()
            save_samples_truncted_prob(ply_save_path, points.detach().numpy(), r.detach().numpy())

            voxelgrid = self.create_voxelgrid(points.detach().numpy(), r[0, 0].detach().numpy(), 0.005)
            try:
                verts, faces, normals, values = measure._marching_cubes_lewiner.marching_cubes(voxelgrid, 0.5)
                color = np.ones_like(verts) * 0.5
                save_path = os.path.join(self.opt.val_results_path, f"{epoch_id}__frame_{frame_id}.obj")
                save_obj_mesh_with_color(save_path, verts, faces, color)
            except:
                pass

            self._running_log["loss"] = data_dict["loss"] 
            self._running_log["chamfer_dist"] = 0 
            
            # record log
            self.log["val"]["loss"].append(self._running_log["loss"].item())
            self.log["val"]["chamfer_dist"].append(self._running_log["chamfer_dist"])

        # report
        # dump log
        self._dump_log("val")
        self._val_report(epoch_id)

    def _reset_log(self, phase):
        if phase == "train":
            self.log[phase] = {
                # info
                "forward": [],
                "backward": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
            }
        else:
            self.log[phase] = {
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                # metrics (float, not torch.cuda.FloatTensor)
                "chamfer_dist": [],
            }

    def _train_feed(self, dataloader, epoch_id):
        self.netN.train()
        self.netG.train()

        self._reset_log("train")
        for train_data in dataloader:
            # move to cuda
            for key in train_data:
                if torch.is_tensor(train_data[key]):
                    train_data[key] = train_data[key].to(device=self.device)

            # initialize the running loss
            self._running_log = {
                "loss": 0,
            }
            self.log["train"]["fetch"].append(train_data["load_time"].sum().item())

            # forward
            start = time.time()
            train_data = self._forward(train_data)
            self._running_log["loss"] = train_data["loss"]
            self.log["train"]["forward"].append(time.time() - start)
            # backward
            start = time.time()
            self._backward()
            self.log["train"]["backward"].append(time.time() - start)
            
            # record log
            self.log["train"]["loss"].append(self._running_log["loss"].item())

            # report
            iter_time = self.log["train"]["fetch"][-1]
            iter_time += self.log["train"]["forward"][-1]
            iter_time += self.log["train"]["backward"][-1]
            self.log["train"]["iter_time"].append(iter_time)
            if (self._global_iter_id + 1) % self.opt.freq_plot == 0:
                self._train_report(epoch_id)

                # dump log
                self._dump_log("train")

            self._global_iter_id += 1

            # save model
            if (self._global_iter_id + 1) % self.opt.freq_save == 0:
                self._log(f"Saving checkpoints at iteration: {self._global_iter_id + 1}")
                torch.save(self.netG.state_dict(), f'{self.opt.checkpoints_path}/netG_latest')
                torch.save(self.netG.state_dict(), f'{self.opt.checkpoints_path}/netG_it_{self._global_iter_id}')

    def _val_report(self, epoch_id):
        # print report
        val_report = self.__val_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["val"],
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            chamfer_dist=round(np.mean([v for v in self.log["val"]["chamfer_dist"]]), 5),
        )
        self._log_eval(val_report)


    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.val_dataloader) * self.opt.freq_val * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            lr=round(self.optimizer.param_groups[0]["lr"], 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"],
            data_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self._log(iter_report)

    def _dump_log(self, phase):
        if phase == "train":
            log = {
                "loss": ["loss"]
            }
        elif phase == "val":
            log = {
                "loss": ["loss"],
                "metrics": ["chamfer_dist"]
            }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

def decode_eta(eta_sec):
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_sec < 60:
        eta['s'] = int(eta_sec)
    elif eta_sec >= 60 and eta_sec < 3600:
        eta['m'] = int(eta_sec / 60)
        eta['s'] = int(eta_sec % 60)
    else:
        eta['h'] = int(eta_sec / (60 * 60))
        eta['m'] = int(eta_sec % (60 * 60) / 60)
        eta['s'] = int(eta_sec % (60 * 60) % 60)

    return eta