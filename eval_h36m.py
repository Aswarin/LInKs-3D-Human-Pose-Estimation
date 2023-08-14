"""We used this file the evaluate our models on the human3.6m data"""


import torch.nn
import torch.optim
from torch.utils import data
import pytorch_lightning as pl

from utils.h36m_dataset_class import H36M_Data
from utils.models_def import Left_Right_Lifter, Leg_Lifter, Torso_Lifter
from types import SimpleNamespace
from utils.rotation_conversions import euler_angles_to_matrix
from utils.metrics import Metrics
from utils.metrics_batch import Metrics as mb
from utils.helpers import *

# https://github.com/VLL-HD/FrEIA
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse

datafile = '../EVAL_DATA/correct_interesting_frames_h36m.pkl'

left_lifter = Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.25,)
right_lifter = Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.25,)

legs_lifter = Leg_Lifter(use_batchnorm=False, num_joints=7, use_dropout=False, d_rate=0.25)
torso_lifter = Torso_Lifter(use_batchnorm=False, num_joints=10, use_dropout=False, d_rate=0.25)

left_lifter.load_state_dict(torch.load('models/left_lifter.pt'))
right_lifter.load_state_dict(torch.load('models/right_lifter.pt'))

# legs_lifter.load_state_dict(torch.load('models/leg_lifter.pt'))
# torso_lifter.load_state_dict(torch.load('models/torso_lifter.pt'))


test_data = H36M_Data(datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True, subjects=['S9', 'S11'])


poses_2d = torch.tensor(test_data.data['poses_2d'])
poses_3d = torch.tensor(test_data.data['poses_3d'])


metrics = Metrics()

"""Left and Right evaluation uncomment"""
inp_left, inp_right = split_data_left_right(poses_2d)

pred_left, _ = left_lifter(inp_left)
pred_right, _ = right_lifter(inp_right)

pred_left[:, 0] = 0.0
pred_right[:, 0] = 0.0

pred_test = combine_left_right_pred_1d(pred_left, pred_right, choice='right').reshape(-1, 17)


pred_test_depth = pred_test + 10

"""Leg and Torso evaluation uncomment"""
# poses_2d = poses_2d.reshape(-1, 2, 17)
# inp_legs = poses_2d[:, :, :7].reshape(-1, 14)
# inp_torso = poses_2d[:, :, 7:].reshape(-1, 20)
#
# legs_pred_test, _ = legs_lifter(inp_legs)
# torso_pred_test, _ = torso_lifter(inp_torso)
#
# pred_test = torch.cat((legs_pred_test, torso_pred_test), dim=1)
# pred_test[:, 0] = 0.0
#
# pred_test_depth = pred_test + 10

pred_test_poses = torch.cat(
    ((poses_2d.reshape(-1, 2, 17) * pred_test_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
     pred_test_depth), dim=1).detach().cpu().numpy()

# rotate to camera coordinate system
test_poses_cam_frame = pred_test_poses.reshape(-1, 3, 17)

pa = 0

err_list = list()
for eval_cnt in range(int(poses_3d.shape[0])):
    err = metrics.pmpjpe(poses_3d[eval_cnt].reshape(-1, 51).cpu().numpy(),
                         pred_test_poses[eval_cnt].reshape(-1, 51),
                         reflection='best')
    pa += err
    err_list.append(err)

pa /= poses_3d.shape[0]

mpjpe_scaled = mb().mpjpe(poses_3d,
                                 torch.tensor(test_poses_cam_frame), num_joints=17,
                                 root_joint=0).mean().cpu().numpy()

print('The PA-MPJPE error was ' + str(pa))
print('The N-MPJPE error was ' + str(mpjpe_scaled))