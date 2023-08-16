from time import time
from types import SimpleNamespace
from utils.helpers import *
import torch.optim
import numpy as np
import torch.nn.functional as F

# import data
torch.manual_seed(42)
from torch.utils import data
from utils.h36m_dataset_class import H36M_Data, MPI_INF_3DHP_Dataset
import torch.optim as optim
import torch.nn as nn
from utils.metrics import Metrics
import math
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Train 2D INN')
parser.add_argument("-l", "--left_right_side_keypoints", help="number of key-points in each split",
                    type=int, default=22)

args = parser.parse_args()
num_bases = args.left_right_side_keypoints

wandb.init(project="LInKs")
wandb.run.name = "INN2D Left, Right, Torso and Legs " + wandb.run.name

config = wandb.config
config.learning_rate = 0.0002  # 0.0001
config.BATCH_SIZE = 256
config.N_epochs = 100

config.num_left_right_keypoints = num_bases
config.leg_num_keypoints = 14
config.torso_num_keypoints = 20

# config.datafile = '../EVAL_DATA/correct_interesting_frames_h36m.pkl'
#
# my_dataset = H36M_Data(config.datafile, train=True, normalize_func=normalize_head, get_2dgt=True,
#                        subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
# train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))


inn_2d_left_split = Ff.SequenceINN(config.num_left_right_keypoints)
inn_2d_right_split = Ff.SequenceINN(config.num_left_right_keypoints)
inn_2d_torso_split = Ff.SequenceINN(config.torso_num_keypoints)
inn_2d_legs_split = Ff.SequenceINN(config.leg_num_keypoints)
full_pose_inn2d = Ff.SequenceINN(34)
for k in range(8):
    inn_2d_right_split.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    inn_2d_left_split.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    inn_2d_torso_split.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    inn_2d_legs_split.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    full_pose_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
inn_2d_right_split.cuda()
inn_2d_left_split.cuda()
inn_2d_torso_split.cuda()
inn_2d_legs_split.cuda()
full_pose_inn2d.cuda()

full_pose_inn2d.load_state_dict(torch.load('models/mpi_norm_flow_sampling.pt'))
# freeze all weights in INN
for param in full_pose_inn2d.parameters():
    param.requires_grad = False


left_split_opt = optim.Adam(inn_2d_left_split.parameters(), lr=config.learning_rate, weight_decay=1e-5)
right_split_opt = optim.Adam(inn_2d_right_split.parameters(), lr=config.learning_rate, weight_decay=1e-5)
leg_optimizer = optim.Adam(inn_2d_legs_split.parameters(), lr=config.learning_rate, weight_decay=1e-5)
torso_optimizer = optim.Adam(inn_2d_torso_split.parameters(), lr=config.learning_rate, weight_decay=1e-5)
leg_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=leg_optimizer, gamma=0.95)
torso_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=torso_optimizer, gamma=0.95)
left_split_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=left_split_opt, gamma=0.95)
right_split_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=right_split_opt, gamma=0.95)

torch.autograd.set_detect_anomaly(True)

# print('start training with ' + str(torso_num_bases) + ' torso PCA bases and ' + str(leg_num_bases) + ' leg PCA bases')
print('start training one left right split inn2d')

metrics = Metrics()

losses = SimpleNamespace()
losses_mean = SimpleNamespace()

for epoch in range(config.N_epochs):
    tic = time()
    for i, sample in enumerate(train_loader):

        #noise_factor = 0.002 * epoch

        # poses_2d = {key:sample[key] for key in all_cams}
        poses_2d = sample['p2d_gt']

        left_inp_pose, right_inp_pose = split_data_left_right(poses_2d)

        leg_poses_2d = torch.Tensor(poses_2d.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)).cuda()
        torso_poses_2d = torch.Tensor(poses_2d.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)).cuda()

        z_2d_leg, log_jac_det_2d_leg = inn_2d_legs_split(leg_poses_2d)
        z_2d_torso, log_jac_det_2d_torso = inn_2d_torso_split(torso_poses_2d)

        leg_likeli_true = (0.5 * torch.sum(z_2d_leg ** 2, 1) - log_jac_det_2d_leg)
        torso_likeli_true = (0.5 * torch.sum(z_2d_torso ** 2, 1) - log_jac_det_2d_torso)

        z_2d_left, log_jac_det_2d_left = inn_2d_left_split(left_inp_pose.cuda())
        z_2d_right, log_jac_det_2d_right = inn_2d_right_split(right_inp_pose.cuda())

        left_likeli_true = (0.5 * torch.sum(z_2d_left ** 2, 1) - log_jac_det_2d_left)
        right_likeli_true = (0.5 * torch.sum(z_2d_right ** 2, 1) - log_jac_det_2d_right)

        losses.dist_2d_left = left_likeli_true.mean()
        losses.dist_2d_right = right_likeli_true.mean()
        losses.dist_2d_legs = leg_likeli_true.mean()
        losses.dist_2d_torso = torso_likeli_true.mean()

        # generate samples from the normalising flow
        with torch.no_grad():
            full_pose_gaussian, _ = full_pose_inn2d(poses_2d.cuda())
            noisy_gaussian = add_noise(full_pose_gaussian, noise_factor=0.2)
            drawn_samples, _ = full_pose_inn2d(noisy_gaussian, rev=True)
            drawn_samples = drawn_samples.reshape(-1, 2, 17)
            drawn_samples[:, :, [0]] = 0.0
            drawn_samples = drawn_samples.reshape(-1, poses_2d.shape[1])
            inp_samples = drawn_samples.data

            left_inp_pose, right_inp_pose = split_data_left_right(inp_samples)

            leg_poses_2d = torch.Tensor(inp_samples.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)).cuda()
            torso_poses_2d = torch.Tensor(inp_samples.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)).cuda()


        z_2d_leg, log_jac_det_2d_leg = inn_2d_legs_split(leg_poses_2d)
        z_2d_torso, log_jac_det_2d_torso = inn_2d_torso_split(torso_poses_2d)

        leg_likeli_sample = (0.5 * torch.sum(z_2d_leg ** 2, 1) - log_jac_det_2d_leg)
        torso_likeli_sample = (0.5 * torch.sum(z_2d_torso ** 2, 1) - log_jac_det_2d_torso)

        z_2d_left, log_jac_det_2d_left = inn_2d_left_split(left_inp_pose.cuda())
        z_2d_right, log_jac_det_2d_right = inn_2d_right_split(right_inp_pose.cuda())

        left_likeli_sample = (0.5 * torch.sum(z_2d_left ** 2, 1) - log_jac_det_2d_left)
        right_likeli_sample = (0.5 * torch.sum(z_2d_right ** 2, 1) - log_jac_det_2d_right)

        losses.dist_2d_legs_sample = leg_likeli_sample.mean()
        losses.dist_2d_torso_sample = torso_likeli_sample.mean()

        losses.dist_2d_left_sample = left_likeli_sample.mean()
        losses.dist_2d_right_sample = right_likeli_sample.mean()

        losses.loss = losses.dist_2d_right_sample + losses.dist_2d_right + losses.dist_2d_left + losses.dist_2d_left_sample \
                      + losses.dist_2d_torso_sample + losses.dist_2d_torso + losses.dist_2d_legs_sample + losses.dist_2d_legs

        left_split_opt.zero_grad()
        right_split_opt.zero_grad()
        leg_optimizer.zero_grad()
        torso_optimizer.zero_grad()
        losses.loss.backward()
        left_split_opt.step()
        right_split_opt.step()
        leg_optimizer.step()
        torso_optimizer.step()

        for key, value in losses.__dict__.items():
            if key not in losses_mean.__dict__.keys():
                losses_mean.__dict__[key] = []

            losses_mean.__dict__[key].append(value.item())

        if not (epoch == 0 and i == 0):
            for key, value in losses_mean.__dict__.items():
                wandb.log({key: np.mean(value)})

        losses_mean = SimpleNamespace()

    wandb.log({'epoch': epoch})
    left_split_sched.step()
    right_split_sched.step()
    leg_sched.step()
    torso_sched.step()
    torch.save(inn_2d_left_split.state_dict(), 'mpi_norm_flow_left_2' + '.pt')
    torch.save(inn_2d_right_split.state_dict(), 'mpi_norm_flow_right_2' + '.pt')
    torch.save(inn_2d_legs_split.state_dict(), 'mpi_norm_flow_legs_2' + '.pt')
    torch.save(inn_2d_torso_split.state_dict(), 'mpi_norm_flow_torso_2' + '.pt')
