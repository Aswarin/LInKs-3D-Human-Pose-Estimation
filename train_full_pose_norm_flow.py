"""This code creates the full pose norm flow which is then used for generative sampling when training the lifting networks"""

from time import time
from types import SimpleNamespace
from torch.utils import data
import torch.optim as optim
from utils.helpers import *
from utils.h36m_dataset_class import H36M_Data, MPI_INF_3DHP_Dataset
from utils.metrics import Metrics
import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Train 2D INN')
parser.add_argument("-n", "--num_keypoints", help="number of keypoints",
                    type=int, default=34)

args = parser.parse_args()
num_keypoints = args.num_keypoints

wandb.init(project="LInKs")
wandb.run.name = "INN2D Full Pose With Sampling " + wandb.run.name

config = wandb.config
config.learning_rate = 0.0002  #0.0001
config.BATCH_SIZE = 4*64
config.N_epochs = 100

config.num_keypoints = num_keypoints

# config.datafile = '../EVAL_DATA/h36m_data.pkl'
#
# my_dataset = H36M_Data(config.datafile, train=True, get_pca=True, normalize_func=normalize_head, get_2dgt=True, subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
# train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024,  dims_out))

inn_2d = Ff.SequenceINN(num_keypoints)
for k in range(8):
    inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
inn_2d.cuda()

params = list(inn_2d.parameters())
optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)



torch.autograd.set_detect_anomaly(True)

print('start training inn2d')

metrics = Metrics()

losses = SimpleNamespace()
losses_mean = SimpleNamespace()

for epoch in range(config.N_epochs):
    tic = time()
    for i, sample in enumerate(train_loader):

        #poses_2d = {key:sample[key] for key in all_cams}
        poses_2d = sample['p2d_gt']
        inp_poses = torch.Tensor(poses_2d).cuda()

        z_2d, log_jac_det_2d = inn_2d(inp_poses)

        likeli = (0.5 * torch.sum(z_2d ** 2, 1) - log_jac_det_2d)
        losses.dist_2d = likeli.mean()


        with torch.no_grad():
            gaussian_noisy = add_noise(z_2d, noise_factor=0.2)
            drawn_samples, _ = inn_2d(gaussian_noisy.cuda(), rev=True)
            drawn_samples = drawn_samples.reshape(-1, 2, 17)
            drawn_samples[:, :, [0]] = 0.0
            drawn_samples = drawn_samples.reshape(-1, inp_poses.shape[1])
            inp_samples = drawn_samples.data

        z_2d_sample, log_jac_det_2d_sample = inn_2d(inp_samples)
        likeli_sample = (0.5 * torch.sum(z_2d_sample ** 2, 1) - log_jac_det_2d_sample)
        losses.dist_2d_sample = likeli_sample.mean()

        losses.loss = losses.dist_2d + losses.dist_2d_sample


        optimizer.zero_grad()
        losses.loss.backward()
        optimizer.step()

        for key, value in losses.__dict__.items():
            if key not in losses_mean.__dict__.keys():
                losses_mean.__dict__[key] = []

            losses_mean.__dict__[key].append(value.item())

        if not (epoch == 0 and i == 0):
            for key, value in losses_mean.__dict__.items():
                wandb.log({key: np.mean(value)})


        losses_mean = SimpleNamespace()

    scheduler.step()
    wandb.log({'epoch': epoch})
    torch.save(inn_2d.state_dict(), 'models/norm_flow_sampling.pt')
