"""The below code trains the occlusion models
During training the occlusion models learn from the predictions of the leg and torso network.
During validation the occlusion models predict from the partial 3D poses produced from various configurations of our lifting networks
This helps simulate more realisitc occlusion scenarios
"""
import torch.nn
import torch.optim
from torch.utils import data
import pytorch_lightning as pl
from utils.models_def import Occluded_Torso_Predictor, Occluded_Legs_Predictor, Occluded_Limb_Predictor, \
    Occluded_Left_Right_Predictor, Left_Right_Lifter, Leg_Lifter, Torso_Lifter
from types import SimpleNamespace
from utils.rotation_conversions import euler_angles_to_matrix
from utils.metrics import Metrics
from utils.metrics_batch import Metrics as mb
from utils.helpers import *
from utils.h36m_dataset_class import H36M_Data

# https://github.com/VLL-HD/FrEIA
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse

parser = argparse.ArgumentParser(description='Train 2D INN with PCA')
parser.add_argument("-n", "--num_bases", help="number of PCA bases",
                    type=int, default=26)
parser.add_argument("-b", "--bl", help="bone lengths",
                    type=float, default=50.0)  # 50.0
parser.add_argument("-t", "--translation", help="camera translation",
                    type=float, default=10.0)
parser.add_argument("-r", "--rep2d", help="2d reprojection",
                    type=float, default=1.0)
parser.add_argument("-o", "--rot3d", help="3d reconstruction",
                    type=float, default=1.0)
parser.add_argument("-v", "--velocity", help="velocity",
                    type=float, default=1.0)
parser.add_argument("-l", "--likelihood", help="likelihood",
                    type=float, default=1.0)

args = parser.parse_args()
num_bases = args.num_bases

wandb.init(project="3DV Paper")
wandb.run.name = "3D Occlusion Part Prediction Model Training " + str(num_bases) + "_" + wandb.run.name
project_folder = ''
data_folder = ''
config = wandb.config
config.learning_rate = 0.0002
config.BATCH_SIZE = 256
config.N_epochs = 10

config.use_elevation = True
config.sample_data = True

config.weight_bl = float(args.bl)
config.weight_likeli = float(args.likelihood)
config.depth = float(args.translation)
config.use_gt = True

config.num_joints = 17
config.num_bases = num_bases


def combine_pose_and_limb(pose, limb, which_limb):
    limb = limb.reshape(-1, 3, 3)
    pose = pose.reshape(-1, 3, 14)
    if which_limb == 'll':
        full_pose = torch.cat((pose[:, :, :4], limb, pose[:, :, 4:]), dim=2)
    elif which_limb == 'rl':
        full_pose = torch.cat((pose[:, :, :1], limb, pose[:, :, 1:]), dim=2)
    elif which_limb == 'la':
        full_pose = torch.cat((pose[:, :, :11], limb, pose[:, :, 11:]), dim=2)
    elif which_limb == 'ra':
        full_pose = torch.cat((pose, limb), dim=2)
    return full_pose.reshape(-1, 51)


class Limb_Predictor(pl.LightningModule):
    def __init__(self, torso_lifter, leg_lifter, left_lifter, right_lifter):
        super(Limb_Predictor, self).__init__()

        self.left_lifter = left_lifter
        self.torso_lifter = torso_lifter
        self.leg_lifter = leg_lifter
        self.right_lifter = right_lifter
        self.model_list = {}
        self.left_leg_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
        self.right_leg_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
        self.left_arm_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
        self.right_arm_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
        self.both_legs_predictor = Occluded_Legs_Predictor(use_batchnorm=False, num_joints=11).cuda()
        self.torso_predictor = Occluded_Torso_Predictor(use_batchnorm=False, num_joints=7).cuda()
        self.left_predictor = Occluded_Left_Right_Predictor(use_batchnorm=False, num_joints=11).cuda()
        self.right_predictor = Occluded_Left_Right_Predictor(use_batchnorm=False, num_joints=11).cuda()


        self.automatic_optimization = False

        self.metrics = Metrics()

        self.losses = SimpleNamespace()
        self.losses_mean = SimpleNamespace()

    def forward(self, x):
        predict = self.depth_estimator(x)
        return predict

    def configure_optimizers(self):

        ra_opt = torch.optim.Adam(self.right_arm_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        ra_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=ra_opt, gamma=0.95)

        la_opt = torch.optim.Adam(self.left_arm_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        la_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=la_opt, gamma=0.95)

        rl_opt = torch.optim.Adam(self.right_leg_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        rl_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=rl_opt, gamma=0.95)

        ll_opt = torch.optim.Adam(self.left_leg_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        ll_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=ll_opt, gamma=0.95)

        left_opt = torch.optim.Adam(self.left_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        left_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=left_opt, gamma=0.95)

        right_opt = torch.optim.Adam(self.right_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        right_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=right_opt, gamma=0.95)

        legs_opt = torch.optim.Adam(self.both_legs_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        legs_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=legs_opt, gamma=0.95)

        torso_opt = torch.optim.Adam(self.torso_predictor.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        torso_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=torso_opt, gamma=0.95)

        return [ra_opt, la_opt, rl_opt, ll_opt, left_opt, right_opt, legs_opt, torso_opt], [ra_sched, la_sched, rl_sched, ll_sched, left_sched, right_sched, legs_sched, torso_sched]

    def training_epoch_end(self, outputs):
        schedulers = self.lr_schedulers()
        for sch in schedulers:
            sch.step()

    def training_step(self, train_batch, batch_idx):

        opts = self.optimizers()
        for opt in opts:
            opt.zero_grad()

        inp_poses = train_batch['p2d_gt'].reshape(-1, 2, 17)

        inp_poses = inp_poses.reshape(-1, 34)

        left_split, right_split = split_data_left_right(inp_poses)
        legs_split = inp_poses.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)
        torso_split = inp_poses.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)

        legs_pred, _ = self.leg_lifter(legs_split)
        torso_pred, _ = self.torso_lifter(torso_split)
        left_pred, _ = self.left_lifter(left_split)
        right_pred, _ = self.right_lifter(right_split)

        #pred = combine_left_right_pred_1d(left_pred, right_pred, choice='right').reshape(-1, 17)
        pred = torch.cat((legs_pred, torso_pred), dim=1)
        pred[:, 0] = 0.0

        pred_depth = pred + config.depth

        train_3d_pose = torch.cat(
            (
            (inp_poses.reshape(-1, 2, 17) * pred_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_depth),
            dim=1).reshape(-1, 3, 17)

        train_3d_pose = train_3d_pose.reshape(-1, 3, 17) - train_3d_pose.reshape(-1, 3, 17)[:, :, [0]]

        train_left_arm_3d = train_3d_pose[:, :, 11:14].reshape(-1, 9)
        train_right_arm_3d = train_3d_pose[:, :, 14:].reshape(-1, 9)
        train_left_leg_3d = train_3d_pose[:, :, 4:7].reshape(-1, 9)
        train_right_leg_3d = train_3d_pose[:, :, 1:4].reshape(-1, 9)
        train_left_side_3d = torch.cat((train_3d_pose[:, :, 4:7], train_3d_pose[:, :, 11:14]), dim=2).reshape(-1, 18)
        train_right_side_3d = torch.cat((train_3d_pose[:, :, 1:4], train_3d_pose[:, :, 14:]), dim=2).reshape(-1, 18)
        train_both_legs_3d = train_3d_pose[:, :, 1:7].reshape(-1, 18)
        train_torso_3d = train_3d_pose[:, :, 7:].reshape(-1, 30)

        input_3d_no_left_arm = torch.cat((train_3d_pose[:, :, :11], train_3d_pose[:, :, 14:]), dim=2).reshape(-1, 42)
        input_3d_no_right_arm = train_3d_pose[:, :, :14].reshape(-1, 42)
        input_3d_no_left_leg = torch.cat((train_3d_pose[:, :, :4], train_3d_pose[:, :, 7:]), dim=2).reshape(-1, 42)
        input_3d_no_right_leg = torch.cat((train_3d_pose[:, :, :1], train_3d_pose[:, :, 4:]), dim=2).reshape(-1, 42)
        input_3d_no_torso = train_3d_pose[:, :, :7].reshape(-1, 21)
        input_3d_no_legs = torch.cat((train_3d_pose[:, :, :1], train_3d_pose[:, :, 7:]), dim=2).reshape(-1, 33)
        input_3d_no_right_side, input_3d_no_left_side = split_data_left_right_3d(train_3d_pose)

        left_arm_3d_predictions = self.left_arm_predictor(input_3d_no_left_arm)
        right_arm_3d_predictions = self.right_arm_predictor(input_3d_no_right_arm)
        left_leg_3d_predictions = self.left_leg_predictor(input_3d_no_left_leg)
        right_leg_3d_predictions = self.right_leg_predictor(input_3d_no_right_leg)
        torso_3d_predictions = self.torso_predictor(input_3d_no_torso)
        legs_3d_predictions = self.both_legs_predictor(input_3d_no_legs)
        left_side_3d_predictions = self.left_predictor(input_3d_no_left_side)
        right_side_3d_predictions = self.right_predictor(input_3d_no_right_side)


        self.losses.threed_loss_left_arm = (left_arm_3d_predictions - train_left_arm_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_arm = (right_arm_3d_predictions - train_right_arm_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_left_leg = (left_leg_3d_predictions - train_left_leg_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_leg = (right_leg_3d_predictions - train_right_leg_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_left_side = (left_side_3d_predictions - train_left_side_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_side = (right_side_3d_predictions - train_right_side_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_both_legs = (legs_3d_predictions - train_both_legs_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_torso = (torso_3d_predictions - train_torso_3d).square().sum(dim=1).mean()


        y_ang = (torch.rand((inp_poses.shape[0], 1), device=self.device) - 0.5) * 1.99 * np.pi
        z_ang = torch.zeros((inp_poses.shape[0], 1), device=self.device)
        Ry = euler_angles_to_matrix(torch.cat((z_ang, y_ang, z_ang), dim=1), 'XYZ')

        train_3d_pose = Ry.matmul(train_3d_pose)


        train_left_arm_3d = train_3d_pose[:, :, 11:14].reshape(-1, 9)
        train_right_arm_3d = train_3d_pose[:, :, 14:].reshape(-1, 9)
        train_left_leg_3d = train_3d_pose[:, :, 4:7].reshape(-1, 9)
        train_right_leg_3d = train_3d_pose[:, :, 1:4].reshape(-1, 9)
        train_left_side_3d = torch.cat((train_3d_pose[:, :, 4:7], train_3d_pose[:, :, 11:14]), dim=2).reshape(-1, 18)
        train_right_side_3d = torch.cat((train_3d_pose[:, :, 1:4], train_3d_pose[:, :, 14:]), dim=2).reshape(-1, 18)
        train_both_legs_3d = train_3d_pose[:, :, 1:7].reshape(-1, 18)
        train_torso_3d = train_3d_pose[:, :, 7:].reshape(-1, 30)

        input_3d_no_left_arm = torch.cat((train_3d_pose[:, :, :11], train_3d_pose[:, :, 14:]), dim=2).reshape(-1, 42)
        input_3d_no_right_arm = train_3d_pose[:, :, :14].reshape(-1, 42)
        input_3d_no_left_leg = torch.cat((train_3d_pose[:, :, :4], train_3d_pose[:, :, 7:]), dim=2).reshape(-1, 42)
        input_3d_no_right_leg = torch.cat((train_3d_pose[:, :, :1], train_3d_pose[:, :, 4:]), dim=2).reshape(-1, 42)
        input_3d_no_torso = train_3d_pose[:, :, :7].reshape(-1, 21)
        input_3d_no_legs = torch.cat((train_3d_pose[:, :, :1], train_3d_pose[:, :, 7:]), dim=2).reshape(-1, 33)
        input_3d_no_right_side, input_3d_no_left_side = split_data_left_right_3d(train_3d_pose)

        left_arm_3d_predictions = self.left_arm_predictor(input_3d_no_left_arm)
        right_arm_3d_predictions = self.right_arm_predictor(input_3d_no_right_arm)
        left_leg_3d_predictions = self.left_leg_predictor(input_3d_no_left_leg)
        right_leg_3d_predictions = self.right_leg_predictor(input_3d_no_right_leg)
        torso_3d_predictions = self.torso_predictor(input_3d_no_torso)
        legs_3d_predictions = self.both_legs_predictor(input_3d_no_legs)
        left_side_3d_predictions = self.left_predictor(input_3d_no_left_side)
        right_side_3d_predictions = self.right_predictor(input_3d_no_right_side)

        self.losses.threed_loss_left_arm += (left_arm_3d_predictions - train_left_arm_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_arm += (right_arm_3d_predictions - train_right_arm_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_left_leg += (left_leg_3d_predictions - train_left_leg_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_leg += (right_leg_3d_predictions - train_right_leg_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_left_side += (left_side_3d_predictions - train_left_side_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_side += (right_side_3d_predictions - train_right_side_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_both_legs += (legs_3d_predictions - train_both_legs_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_torso += (torso_3d_predictions - train_torso_3d).square().sum(dim=1).mean()


        y_ang = (torch.rand((inp_poses.shape[0], 1), device=self.device) - 0.5) * 1.99 * np.pi
        z_ang = torch.zeros((inp_poses.shape[0], 1), device=self.device)
        Ry = euler_angles_to_matrix(torch.cat((z_ang, y_ang, z_ang), dim=1), 'XYZ')

        train_3d_pose = Ry.matmul(train_3d_pose)


        train_left_arm_3d = train_3d_pose[:, :, 11:14].reshape(-1, 9)
        train_right_arm_3d = train_3d_pose[:, :, 14:].reshape(-1, 9)
        train_left_leg_3d = train_3d_pose[:, :, 4:7].reshape(-1, 9)
        train_right_leg_3d = train_3d_pose[:, :, 1:4].reshape(-1, 9)
        train_left_side_3d = torch.cat((train_3d_pose[:, :, 4:7], train_3d_pose[:, :, 11:14]), dim=2).reshape(-1, 18)
        train_right_side_3d = torch.cat((train_3d_pose[:, :, 1:4], train_3d_pose[:, :, 14:]), dim=2).reshape(-1, 18)
        train_both_legs_3d = train_3d_pose[:, :, 1:7].reshape(-1, 18)
        train_torso_3d = train_3d_pose[:, :, 7:].reshape(-1, 30)

        input_3d_no_left_arm = torch.cat((train_3d_pose[:, :, :11], train_3d_pose[:, :, 14:]), dim=2).reshape(-1, 42)
        input_3d_no_right_arm = train_3d_pose[:, :, :14].reshape(-1, 42)
        input_3d_no_left_leg = torch.cat((train_3d_pose[:, :, :4], train_3d_pose[:, :, 7:]), dim=2).reshape(-1, 42)
        input_3d_no_right_leg = torch.cat((train_3d_pose[:, :, :1], train_3d_pose[:, :, 4:]), dim=2).reshape(-1, 42)
        input_3d_no_torso = train_3d_pose[:, :, :7].reshape(-1, 21)
        input_3d_no_legs = torch.cat((train_3d_pose[:, :, :1], train_3d_pose[:, :, 7:]), dim=2).reshape(-1, 33)
        input_3d_no_right_side, input_3d_no_left_side = split_data_left_right_3d(train_3d_pose)

        left_arm_3d_predictions = self.left_arm_predictor(input_3d_no_left_arm)
        right_arm_3d_predictions = self.right_arm_predictor(input_3d_no_right_arm)
        left_leg_3d_predictions = self.left_leg_predictor(input_3d_no_left_leg)
        right_leg_3d_predictions = self.right_leg_predictor(input_3d_no_right_leg)
        torso_3d_predictions = self.torso_predictor(input_3d_no_torso)
        legs_3d_predictions = self.both_legs_predictor(input_3d_no_legs)
        left_side_3d_predictions = self.left_predictor(input_3d_no_left_side)
        right_side_3d_predictions = self.right_predictor(input_3d_no_right_side)

        self.losses.threed_loss_left_arm += (left_arm_3d_predictions - train_left_arm_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_arm += (right_arm_3d_predictions - train_right_arm_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_left_leg += (left_leg_3d_predictions - train_left_leg_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_leg += (right_leg_3d_predictions - train_right_leg_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_left_side += (left_side_3d_predictions - train_left_side_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_right_side += (right_side_3d_predictions - train_right_side_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_both_legs += (legs_3d_predictions - train_both_legs_3d).square().sum(dim=1).mean()
        self.losses.threed_loss_torso += (torso_3d_predictions - train_torso_3d).square().sum(dim=1).mean()


        self.losses.loss = self.losses.threed_loss_left_arm + \
                           self.losses.threed_loss_right_arm + self.losses.threed_loss_left_leg + self.losses.threed_loss_right_leg +\
                           self.losses.threed_loss_left_side + self.losses.threed_loss_right_side + self.losses.threed_loss_both_legs + \
                           self.losses.threed_loss_torso


        self.manual_backward(self.losses.loss)
        for opt in opts:
            opt.step()

        # logging
        for key, value in self.losses.__dict__.items():
            if key not in self.losses_mean.__dict__.keys():
                self.losses_mean.__dict__[key] = []

            self.losses_mean.__dict__[key].append(value.item())


    def validation_step(self, val_batch, batch_idx):

        if config.use_gt:
            test_poses_2dgt_normalized = val_batch['p2d_gt']
        else:
            test_poses_2dgt_normalized = val_batch['p2d_pred']

        test_3dgt_normalized = val_batch['poses_3d']

        inp_test_poses = test_poses_2dgt_normalized

        left_split, right_split = split_data_left_right(inp_test_poses)
        legs_split = inp_test_poses.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)
        torso_split = inp_test_poses.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)

        legs_pred, _ = self.leg_lifter(legs_split)
        torso_pred, _ = self.torso_lifter(torso_split)
        left_pred, _ = self.left_lifter(left_split)
        right_pred, _ = self.right_lifter(right_split)

        left_pred[:, 0] = 0.0
        right_pred[:, 0] = 0.0
        left_pred += config.depth
        right_pred += config.depth

        pred_lt = torch.cat((legs_pred, torso_pred), dim=1)
        pred_lt[:, 0] = 0.0

        pred_lt += config.depth

        pred_legs_3d = torch.cat(((legs_split.reshape(-1, 2, 7) * pred_lt[:, :7].reshape(-1, 1, 7).repeat(1, 2, 1)).reshape(-1, 14), pred_lt[:, :7]),
            dim=1).reshape(-1, 3, 7)

        pred_torso_3d = torch.cat(((torso_split.reshape(-1, 2, 10) * pred_lt[:, 7:].reshape(-1, 1, 10).repeat(1, 2, 1)).reshape(-1, 20), pred_lt[:, 7:]),
            dim=1).reshape(-1, 3, 10)

        pred_left_3d = torch.cat(((left_split.reshape(-1, 2, 11) * left_pred.reshape(-1, 1, 11).repeat(1, 2, 1)).reshape(-1, 22), left_pred),
            dim=1).reshape(-1, 3, 11)

        pred_right_3d = torch.cat(((right_split.reshape(-1, 2, 11) * right_pred.reshape(-1, 1, 11).repeat(1, 2, 1)).reshape(-1, 22), right_pred),
            dim=1).reshape(-1, 3, 11)


        pred_torso_3d = pred_torso_3d.reshape(-1, 3, 10) - pred_legs_3d.reshape(-1, 3, 7)[:, :, [0]]
        pred_legs_3d = pred_legs_3d.reshape(-1, 3, 7) - pred_legs_3d.reshape(-1, 3, 7)[:, : ,[0]]
        pred_left_3d = pred_left_3d.reshape(-1, 3, 11) - pred_left_3d.reshape(-1, 3, 11)[:, :, [0]]
        pred_right_3d = pred_right_3d.reshape(-1, 3, 11) - pred_right_3d.reshape(-1, 3, 11)[:, :, [0]]

        input_3d_no_left_arm = torch.cat((pred_legs_3d, pred_right_3d[:, :, 4:]), dim=2).reshape(-1, 42)
        input_3d_no_right_arm = torch.cat((pred_legs_3d, pred_left_3d[:, :, 4:]), dim=2).reshape(-1, 42)
        input_3d_no_left_leg = torch.cat((pred_right_3d[:, :, :4], pred_torso_3d), dim=2).reshape(-1, 42)
        input_3d_no_right_leg = torch.cat((pred_left_3d[:, :, :4], pred_torso_3d), dim=2).reshape(-1, 42)
        input_3d_no_torso = pred_legs_3d.reshape(-1, 21)
        input_3d_no_legs = torch.cat((pred_legs_3d[:, :, [0]], pred_torso_3d), dim=2).reshape(-1, 33)
        input_3d_no_right_side = pred_left_3d.reshape(-1, 33)
        input_3d_no_left_side = pred_right_3d.reshape(-1, 33)

        left_arm_3d_predictions = self.left_arm_predictor(input_3d_no_left_arm)
        right_arm_3d_predictions = self.right_arm_predictor(input_3d_no_right_arm)
        left_leg_3d_predictions = self.left_leg_predictor(input_3d_no_left_leg)
        right_leg_3d_predictions = self.right_leg_predictor(input_3d_no_right_leg)
        torso_3d_predictions = self.torso_predictor(input_3d_no_torso)
        legs_3d_predictions = self.both_legs_predictor(input_3d_no_legs)
        left_side_3d_predictions = self.left_predictor(input_3d_no_left_side)
        right_side_3d_predictions = self.right_predictor(input_3d_no_right_side)

        full_pose_la = combine_pose_and_limb(input_3d_no_left_arm, left_arm_3d_predictions, 'la')
        full_pose_ra = combine_pose_and_limb(input_3d_no_right_arm, right_arm_3d_predictions, 'ra')
        full_pose_ll = combine_pose_and_limb(input_3d_no_left_leg, left_leg_3d_predictions, 'll')
        full_pose_rl = combine_pose_and_limb(input_3d_no_right_leg, right_leg_3d_predictions, 'rl')
        full_pose_torso = torch.cat((input_3d_no_torso.reshape(-1, 3, 7), torso_3d_predictions.reshape(-1, 3, 10)), dim=2).reshape(-1, 51)
        full_pose_legs = torch.cat((input_3d_no_legs.reshape(-1, 3, 11)[:, :, :1], legs_3d_predictions.reshape(-1, 3, 6), input_3d_no_legs.reshape(-1, 3, 11)[:, :, 1:]), dim=2).reshape(-1, 51)
        full_pose_left = combine_left_right_occluded_3d(visible_part=input_3d_no_left_side, occluded_part=left_side_3d_predictions, part_occluded='left').reshape(-1, 51)
        full_pose_right = combine_left_right_occluded_3d(visible_part=input_3d_no_right_side, occluded_part=right_side_3d_predictions, part_occluded='right').reshape(-1, 51)

        global_full_pose_la = torch.cat((full_pose_la[:, 0:34], full_pose_la[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_ra = torch.cat((full_pose_ra[:, 0:34], full_pose_ra[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_ll = torch.cat((full_pose_ll[:, 0:34], full_pose_ll[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_rl = torch.cat((full_pose_rl[:, 0:34], full_pose_rl[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_torso = torch.cat((full_pose_torso[:, 0:34], full_pose_torso[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_legs = torch.cat((full_pose_legs[:, 0:34], full_pose_legs[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_left = torch.cat((full_pose_left[:, 0:34], full_pose_left[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()
        global_full_pose_right = torch.cat((full_pose_right[:, 0:34], full_pose_right[:, 34:51] + config.depth), dim=1).detach().cpu().numpy()

        self.losses.pa_la = 0
        self.losses.pa_ra = 0
        self.losses.pa_ll = 0
        self.losses.pa_rl = 0
        self.losses.pa_torso = 0
        self.losses.pa_legs = 0
        self.losses.pa_left = 0
        self.losses.pa_right = 0

        for eval_cnt in range(int(test_3dgt_normalized.shape[0])):
            err_la = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_la[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_la += err_la

            err_ra = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_ra[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_ra += err_ra

            err_ll = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_ll[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_ll += err_ll

            err_rl = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_rl[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_rl += err_rl

            err_torso = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_torso[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_torso += err_torso

            err_legs = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_legs[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_legs += err_legs

            err_left = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_left[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_left += err_left

            err_right = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      global_full_pose_right[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_right += err_right


        self.losses.pa_la /= test_3dgt_normalized.shape[0]
        self.losses.pa_ra /= test_3dgt_normalized.shape[0]
        self.losses.pa_ll /= test_3dgt_normalized.shape[0]
        self.losses.pa_rl /= test_3dgt_normalized.shape[0]
        self.losses.pa_torso /= test_3dgt_normalized.shape[0]
        self.losses.pa_legs /= test_3dgt_normalized.shape[0]
        self.losses.pa_left /= test_3dgt_normalized.shape[0]
        self.losses.pa_right /= test_3dgt_normalized.shape[0]

        # rotate to camera coordinate system
        test_poses_cam_frame_la = global_full_pose_la.reshape(-1, 3, 17)
        test_poses_cam_frame_ra = global_full_pose_ra.reshape(-1, 3, 17)
        test_poses_cam_frame_ll = global_full_pose_ll.reshape(-1, 3, 17)
        test_poses_cam_frame_rl = global_full_pose_rl.reshape(-1, 3, 17)
        test_poses_cam_frame_torso = global_full_pose_torso.reshape(-1, 3, 17)
        test_poses_cam_frame_legs = global_full_pose_legs.reshape(-1, 3, 17)
        test_poses_cam_frame_left = global_full_pose_left.reshape(-1, 3, 17)
        test_poses_cam_frame_right = global_full_pose_right.reshape(-1, 3, 17)

        self.losses.mpjpe_scaled_la = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_la, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_ra = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_ra, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_ll = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_ll, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_rl = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_rl, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_torso = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_torso, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_legs = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_legs, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_left = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_left, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.mpjpe_scaled_right = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame_right, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()


        wandb.log({'epoch': self.current_epoch})

        for key, value in self.losses_mean.__dict__.items():
            wandb.log({key: np.mean(value)})

        self.losses_mean = SimpleNamespace()

        for key, value in self.losses.__dict__.items():
            self.log(key, value.item(), prog_bar=True)



## load pretrained lifting networks
trained_leg_lifting_network = Leg_Lifter(use_batchnorm=False, num_joints=7).cuda()
trained_torso_lifting_network = Torso_Lifter(use_batchnorm=False, num_joints=10).cuda()
trained_left_lifting_network = Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.25).cuda()
trained_right_lifting_network = Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.25).cuda()

trained_leg_lifting_network.load_state_dict(torch.load('/home/aswarin/Desktop/3DV Paper/AAAI_Code/models/best_lifting_models/legs_lifter.pt'))
trained_torso_lifting_network.load_state_dict(torch.load('/home/aswarin/Desktop/3DV Paper/AAAI_Code/models/best_lifting_models/torso_lifter.pt'))
trained_left_lifting_network.load_state_dict(torch.load('/home/aswarin/Desktop/3DV Paper/AAAI_Code/models/best_lifting_models/final_best_left_lifter.pt'), strict=False)
trained_right_lifting_network.load_state_dict(torch.load('/home/aswarin/Desktop/3DV Paper/AAAI_Code/models/best_lifting_models/final_best_right_lifter.pt'), strict=False)

for param in trained_leg_lifting_network.parameters():
    param.requires_grad = False

for param in trained_torso_lifting_network.parameters():
    param.requires_grad = False

for param in trained_left_lifting_network.parameters():
    param.requires_grad = False

for param in trained_right_lifting_network.parameters():
    param.requires_grad = False

datafile = '../EVAL_DATA/correct_interesting_frames_h36m.pkl'

train_data = H36M_Data(datafile, train=True, get_pca=False, normalize_func=normalize_head, get_2dgt=True,
                       subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
test_data = H36M_Data(datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True, subjects=['S9', 'S11'])

test_loader = data.DataLoader(test_data, batch_size=10000, num_workers=0)
train_loader = data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
# model
model = Limb_Predictor(torso_lifter=trained_torso_lifting_network, leg_lifter=trained_leg_lifting_network,
                       left_lifter=trained_left_lifting_network, right_lifter=trained_right_lifting_network)
# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0, logger=False,
                     max_epochs=config.N_epochs)
trainer.fit(model, train_loader, test_loader)

torch.save(model.left_leg_predictor.state_dict(), 'models/occlusion_model_weights/left_leg_estimator.pt')
torch.save(model.right_leg_predictor.state_dict(), 'models/occlusion_model_weights/right_leg_estimator.pt')
torch.save(model.both_legs_predictor.state_dict(), 'models/occlusion_model_weights/both_legs_estimator.pt')
torch.save(model.left_predictor.state_dict(), 'models/occlusion_model_weights/left_side_estimator.pt')
torch.save(model.right_predictor.state_dict(), 'models/occlusion_model_weights/right_side_estimator.pt')
torch.save(model.right_arm_predictor.state_dict(), 'models/occlusion_model_weights/right_arm_estimator.pt')
torch.save(model.left_arm_predictor.state_dict(), 'models/occlusion_model_weights/left_arm_estimator.pt')
torch.save(model.torso_predictor.state_dict(), 'models/occlusion_model_weights/torso_estimator.pt')