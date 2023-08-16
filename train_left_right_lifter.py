import torch.nn
import torch.optim
from torch.utils import data
import pytorch_lightning as pl
from utils.models_def import Left_Right_Lifter
from types import SimpleNamespace
from utils.rotation_conversions import euler_angles_to_matrix
from utils.metrics import Metrics
from utils.metrics_batch import Metrics as mb
from utils.helpers import *
from utils.h36m_dataset_class import H36M_Data, MPI_INF_3DHP_Dataset
torch.manual_seed(42)
# https://github.com/VLL-HD/FrEIA
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import wandb

import argparse

parser = argparse.ArgumentParser(description='Train 2D INN with PCA')
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

wandb.init(project="LInKs")
wandb.run.name = "Left Right lifter " + wandb.run.name
project_folder = ''
data_folder = ''
config = wandb.config
config.learning_rate = 0.0002
config.BATCH_SIZE = 256
config.N_epochs = 100
num_bases = 22
config.num_bases = num_bases
config.use_elevation = True

config.weight_bl = float(args.bl)
config.weight_2d = float(args.rep2d)
config.weight_3d = float(args.rot3d)
config.weight_likeli = float(args.likelihood)
config.weight_velocity = float(args.velocity)
config.depth = float(args.translation)
config.use_gt = True

config.num_joints = 17

class LitLifter(pl.LightningModule):
    def __init__(self, left_inn2d, right_inn2d, full_inn2d):
        super(LitLifter, self).__init__()

        self.left_inn2d = left_inn2d.to(self.device)
        self.right_inn2d = right_inn2d.to(self.device)
        self.full_inn2d = full_inn2d.to(self.device)

        self.left_lifter = Attention_Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.25, num_heads=2).cuda()
        self.right_lifter = Attention_Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.25, num_heads=2).cuda()

        # self.left_lifter = Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False,).cuda()
        # self.right_lifter = Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False).cuda()

        # self.bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
        #                                          0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()

        self.bone_relations_mean = torch.Tensor([0.5180581, 1.73711136, 1.72285805, 0.5180552, 1.73710543,
                                                 1.72285651, 0.92087518, 0.98792375, 0.44812302, 0.44502545,
                                                 0.57462, 1.08121276, 0.9651687, 0.57461556, 1.08122523,
                                                 0.9651657]).cuda()  # human 3.6m relations mean

        #  self.bone_relations_mean = torch.Tensor([0.48123457, 1.83892552, 1.49699857, 0.48123457, 1.83579479,
        # 1.49699856, 0.90885878, 0.99415561, 0.34720909, 0.69462614,
        # 0.57956265, 1.21052741, 0.9251606 , 0.57302514, 1.21052743,
        # 0.92516058]).cuda() #all cameras MPI mean

        #  self.bone_relations_mean = torch.Tensor([0.48115763, 1.83961257, 1.49705786, 0.48115763, 1.83655297,
        # 1.49705784, 0.9086628 , 0.99419836, 0.34713946, 0.69448684,
        # 0.57953889, 1.21026625, 0.924973  , 0.57289866, 1.21026624,
        # 0.924973]) #vnect cameras MPI mean

        #  self.bone_relations_mean = torch.Tensor([0.48069107, 1.84637771, 1.49564841, 0.48069107, 1.84301997,
        # 1.4956484 , 0.90757932, 0.99706493, 0.34679742, 0.69380255,
        # 0.57843534, 1.20698327, 0.92306225, 0.5741528 , 1.20698326,
        # 0.92306223]) #vnect cameras interesting MPI mean

        self.automatic_optimization = False

        self.metrics = Metrics()

        self.losses = SimpleNamespace()
        self.losses_mean = SimpleNamespace()

    def forward(self, x):
        predict = self.depth_estimator(x)
        return predict

    def configure_optimizers(self):

        left_opt = torch.optim.Adam(self.left_lifter.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        left_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=left_opt, gamma=0.95)
        right_opt = torch.optim.Adam(self.right_lifter.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        right_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=right_opt, gamma=0.95)

        return [left_opt, right_opt], [left_sched, right_sched]

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch[0].step()
        sch[1].step()

    def training_step(self, train_batch, batch_idx):

        opt = self.optimizers()
        left_opt = opt[0]
        right_opt = opt[1]
        left_opt.zero_grad()
        right_opt.zero_grad()

        inp_poses = train_batch['p2d_gt']
        with torch.no_grad():
            z_2d, _ = self.full_inn2d(inp_poses)
            gaussian_noisy = add_noise(z_2d, noise_factor=0.2)
            drawn_samples, _ = self.full_inn2d(gaussian_noisy.cuda(), rev=True)
            drawn_samples = drawn_samples.reshape(-1, 2, 17)
            drawn_samples[:, :, [0]] = 0.0
            drawn_samples = drawn_samples.reshape(-1, inp_poses.shape[1])
            inp_samples = drawn_samples.data

            inp_poses = torch.concat((inp_poses, inp_samples), dim=0)


        left_inp_pose, right_inp_pose = split_data_left_right(inp_poses)

        # split data up into legs and torso


        left_pred, left_angle = self.left_lifter(left_inp_pose)
        right_pred, right_angle = self.right_lifter(right_inp_pose)

        props = (left_angle + right_angle) / 2

        pred_left = combine_left_right_pred_1d(left_pred, right_pred, choice='left').reshape(-1, 17)
        pred_right = combine_left_right_pred_1d(left_pred, right_pred, choice='right').reshape(-1, 17)

        pred_left[:, 0] = 0.0
        pred_right[:, 0] = 0.0


        x_ang_comp = torch.ones((inp_poses.shape[0], 1), device=self.device) * props
        y_ang_comp = torch.zeros((inp_poses.shape[0], 1), device=self.device)
        z_ang_comp = torch.zeros((inp_poses.shape[0], 1), device=self.device)

        euler_angles_comp = torch.cat((x_ang_comp, y_ang_comp, z_ang_comp), dim=1)
        R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')

        if config.use_elevation:
            # sample from learned distribution
            elevation = torch.cat((props.mean().reshape(1), props.std().reshape(1)))
            x_ang = (-elevation[0]) + elevation[1] * torch.normal(
                torch.zeros((inp_poses.shape[0], 1), device=self.device),
                torch.ones((inp_poses.shape[0], 1), device=self.device))
        else:
            # predefined distribution
            x_ang = (torch.rand((inp_poses.shape[0], 1), device=self.device) - 0.5) * 2.0 * (np.pi / 9.0)

        y_ang = (torch.rand((inp_poses.shape[0], 1), device=self.device) - 0.5) * 1.99 * np.pi
        z_ang = torch.zeros((inp_poses.shape[0], 1), device=self.device)
        Rx = euler_angles_to_matrix(torch.cat((x_ang, z_ang, z_ang), dim=1), 'XYZ')
        Ry = euler_angles_to_matrix(torch.cat((z_ang, y_ang, z_ang), dim=1), 'XYZ')
        if config.use_elevation:
            R = Rx @ (Ry @ R_comp)
        else:
            R = Rx @ Ry

        depth_right = pred_right + config.depth
        depth_left = pred_left + config.depth
        depth_left[depth_left < 1.0] = 1.0
        depth_right[depth_right < 1.0] = 1.0
        pred_3d_right = torch.cat(
            ((inp_poses.reshape(-1, 2, 17) * depth_right.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth_right),
            dim=1).reshape(-1, 3, 17)

        pred_3d_left = torch.cat(
            ((inp_poses.reshape(-1, 2, 17) * depth_left.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth_left),
            dim=1).reshape(-1, 3, 17)

        pred_3d_right = pred_3d_right.reshape(-1, 3, 17) - pred_3d_right.reshape(-1, 3, 17)[:, :, [0]]
        pred_3d_left = pred_3d_left.reshape(-1, 3, 17) - pred_3d_left.reshape(-1, 3, 17)[:, :, [0]]

        # """new 90 degree consistency loss"""
        # z_y_to_minus_x_pose = torch.stack((pred_3d_left[:, 2, :], pred_3d_left[:, 1, :], -pred_3d_left[:, 0, :]), dim=1).reshape(-1, 51)
        # minus_z_y_to_x_pose = torch.stack((-pred_3d_left[:, 2, :], pred_3d_left[:, 1, :], pred_3d_left[:, 0, :]), dim=1).reshape(-1, 51)
        # global_z_y_pose = torch.cat((z_y_to_minus_x_pose[:, 0:34], z_y_to_minus_x_pose[:, 34:51] + config.depth), dim=1)
        # global_minus_z_y_pose = torch.cat((minus_z_y_to_x_pose[:, 0:34], minus_z_y_to_x_pose[:, 34:51] + config.depth), dim=1)
        # z_y_2d = perspective_projection(global_z_y_pose)
        # minus_z_y_2d = perspective_projection(global_minus_z_y_pose)
        #
        # left_temp, right_temp = split_data_left_right(z_y_2d.reshape(-1, 34))
        #
        # pred_minus_x_left, _ = self.left_lifter(left_temp)
        # pred_minus_x_right, _ = self.right_lifter(right_temp)
        #
        # pred_minus_x = combine_left_right_pred_1d(pred_minus_x_left, pred_minus_x_right, choice='left').reshape(-1, 17)
        #
        # pred_minus_x[:, 0] = 0.0
        #
        # pred_minus_x_depth = pred_minus_x + config.depth
        #
        # pred_3d_minus_x = torch.cat(
        #     ((z_y_2d.reshape(-1, 2, 17) * pred_minus_x_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_minus_x_depth),
        #     dim=1).reshape(-1, 3, 17)
        #
        # pred_3d_minus_x = pred_3d_minus_x.reshape(-1, 3, 17) - pred_3d_minus_x.reshape(-1, 3, 17)[:, :, [0]]
        #
        # pred_3d_minus_x_comp = torch.stack((-pred_3d_minus_x[:, 2, :], pred_3d_minus_x[:, 1, :], pred_3d_minus_x[:, 0, :]),dim=1).reshape(-1, 51)
        # global_pred_3d_minus_x_comp = torch.cat((pred_3d_minus_x_comp[:, 0:34], pred_3d_minus_x_comp[:, 34:51] + config.depth), dim=1)
        #
        # inp_pose_comp = perspective_projection(global_pred_3d_minus_x_comp)
        #
        # self.losses.nintenty_degree_rot_left = (inp_pose_comp - inp_poses).abs().sum(dim=1).mean()
        #
        #
        # left_temp, right_temp = split_data_left_right(minus_z_y_2d)
        #
        # pred_x_left, _ = self.left_lifter(left_temp)
        # pred_x_right, _ = self.right_lifter(right_temp)
        #
        # pred_x = combine_left_right_pred_1d(pred_minus_x_left, pred_minus_x_right, choice='left').reshape(-1, 17)
        #
        # pred_x[:, 0] = 0.0
        # pred_x_depth = pred_x + config.depth
        #
        # pred_3d_x = torch.cat(
        #     ((minus_z_y_2d.reshape(-1, 2, 17) * pred_x_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_x_depth),
        #     dim=1).reshape(-1, 3, 17)
        #
        # pred_3d_x = pred_3d_x.reshape(-1, 3, 17) - pred_3d_x.reshape(-1, 3, 17)[:, :, [0]]
        #
        # pred_3d_x_comp = torch.stack((pred_3d_x[:, 2, :], pred_3d_x[:, 1, :], -pred_3d_x[:, 0, :]),dim=1).reshape(-1, 51)
        # global_pred_3d_x_comp = torch.cat((pred_3d_x_comp[:, 0:34], pred_3d_x_comp[:, 34:51] + config.depth), dim=1)
        #
        # inp_pose_comp = perspective_projection(global_pred_3d_x_comp)
        #
        # self.losses.nintenty_degree_rot_left += (inp_pose_comp - inp_poses).abs().sum(dim=1).mean()
        #
        # z_y_to_minus_x_pose = torch.stack((pred_3d_left[:, 2, :], pred_3d_left[:, 1, :], -pred_3d_left[:, 0, :]),
        #                                   dim=1).reshape(-1, 51)
        # minus_z_y_to_x_pose = torch.stack((-pred_3d_left[:, 2, :], pred_3d_left[:, 1, :], pred_3d_left[:, 0, :]),
        #                                   dim=1).reshape(-1, 51)
        # global_z_y_pose = torch.cat((z_y_to_minus_x_pose[:, 0:34], z_y_to_minus_x_pose[:, 34:51] + config.depth), dim=1)
        # global_minus_z_y_pose = torch.cat((minus_z_y_to_x_pose[:, 0:34], minus_z_y_to_x_pose[:, 34:51] + config.depth),
        #                                   dim=1)
        # z_y_2d = perspective_projection(global_z_y_pose)
        # minus_z_y_2d = perspective_projection(global_minus_z_y_pose)
        #
        # left_temp, right_temp = split_data_left_right(z_y_2d.reshape(-1, 34))
        #
        # pred_minus_x_left, _ = self.left_lifter(left_temp)
        # pred_minus_x_right, _ = self.right_lifter(right_temp)
        #
        # pred_minus_x = combine_left_right_pred_1d(pred_minus_x_left, pred_minus_x_right, choice='right').reshape(-1, 17)
        #
        # pred_minus_x[:, 0] = 0.0
        #
        # pred_minus_x_depth = pred_minus_x + config.depth
        #
        # pred_3d_minus_x = torch.cat(
        #     ((z_y_2d.reshape(-1, 2, 17) * pred_minus_x_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
        #      pred_minus_x_depth),
        #     dim=1).reshape(-1, 3, 17)
        #
        # pred_3d_minus_x = pred_3d_minus_x.reshape(-1, 3, 17) - pred_3d_minus_x.reshape(-1, 3, 17)[:, :, [0]]
        #
        # pred_3d_minus_x_comp = torch.stack(
        #     (-pred_3d_minus_x[:, 2, :], pred_3d_minus_x[:, 1, :], pred_3d_minus_x[:, 0, :]), dim=1).reshape(-1, 51)
        # global_pred_3d_minus_x_comp = torch.cat(
        #     (pred_3d_minus_x_comp[:, 0:34], pred_3d_minus_x_comp[:, 34:51] + config.depth), dim=1)
        #
        # inp_pose_comp = perspective_projection(global_pred_3d_minus_x_comp)
        #
        # self.losses.nintenty_degree_rot_right = (inp_pose_comp - inp_poses).abs().sum(dim=1).mean()
        #
        # left_temp, right_temp = split_data_left_right(minus_z_y_2d)
        #
        # pred_x_left, _ = self.left_lifter(left_temp)
        # pred_x_right, _ = self.right_lifter(right_temp)
        #
        # pred_x = combine_left_right_pred_1d(pred_minus_x_left, pred_minus_x_right, choice='right').reshape(-1, 17)
        #
        # pred_x[:, 0] = 0.0
        # pred_x_depth = pred_x + config.depth
        #
        # pred_3d_x = torch.cat(
        #     ((minus_z_y_2d.reshape(-1, 2, 17) * pred_x_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
        #      pred_x_depth),
        #     dim=1).reshape(-1, 3, 17)
        #
        # pred_3d_x = pred_3d_x.reshape(-1, 3, 17) - pred_3d_x.reshape(-1, 3, 17)[:, :, [0]]
        #
        # pred_3d_x_comp = torch.stack((pred_3d_x[:, 2, :], pred_3d_x[:, 1, :], -pred_3d_x[:, 0, :]), dim=1).reshape(-1,
        #                                                                                                            51)
        # global_pred_3d_x_comp = torch.cat((pred_3d_x_comp[:, 0:34], pred_3d_x_comp[:, 34:51] + config.depth), dim=1)
        #
        # inp_pose_comp = perspective_projection(global_pred_3d_x_comp)
        #
        # self.losses.nintenty_degree_rot_right += (inp_pose_comp - inp_poses).abs().sum(dim=1).mean()
        # """End 90 degree consistency loss"""


        rot_poses_right = (R.matmul(pred_3d_right)).reshape(-1, 51)
        rot_poses_left = (R.matmul(pred_3d_left)).reshape(-1, 51)

        ## lift from augmented camera and normalize
        global_pose_right = torch.cat((rot_poses_right[:, 0:34], rot_poses_right[:, 34:51] + config.depth), dim=1)
        global_pose_left = torch.cat((rot_poses_left[:, 0:34], rot_poses_left[:, 34:51] + config.depth), dim=1)
        rot_2d_right = perspective_projection(global_pose_right)
        rot_2d_left = perspective_projection(global_pose_left)

        norm_poses_left_side, _ = split_data_left_right(rot_2d_left)
        _ , norm_poses_right_side = split_data_left_right(rot_2d_right)

        """Putting both through the normalising flow"""

        z, log_jac_det = self.left_inn2d(norm_poses_left_side)
        likelis_right = 0.5 * torch.sum(z ** 2, 1) - log_jac_det

        self.losses.likeli_right = likelis_right.mean()

        z, log_jac_det = self.right_inn2d(norm_poses_right_side)
        likelis_left = 0.5 * torch.sum(z ** 2, 1) - log_jac_det

        self.losses.likeli_left = likelis_left.mean()

        # z, log_jac_det = self.full_inn2d(rot_2d_left)
        # full_pose_likelis = 0.5 * torch.sum(z ** 2, 1) - log_jac_det
        #
        # z, log_jac_det = self.full_inn2d(rot_2d_right)
        # full_pose_likelis += 0.5 * torch.sum(z ** 2, 1) - log_jac_det
        #
        # self.losses.full_likeli = full_pose_likelis.mean()

        self.losses.likeli = self.losses.likeli_left + self.losses.likeli_right #+ self.losses.full_likeli

        ## reprojection error
        inp_rot_left, _ = split_data_left_right(rot_2d_left)
        _, inp_rot_right = split_data_left_right(rot_2d_right)

        pred_rot_left, _ = self.left_lifter(inp_rot_left)
        pred_rot_right, _ = self.right_lifter(inp_rot_right)

        pred_rot_full_left = combine_left_right_pred_1d(pred_rot_left, pred_rot_right, choice='left').reshape(-1, 17)
        pred_rot_full_right = combine_left_right_pred_1d(pred_rot_left, pred_rot_right, choice='right').reshape(-1, 17)

        pred_rot_full_right[:, 0] = 0.0
        pred_rot_full_left[:, 0] = 0.0

        pred_rot_depth_right = pred_rot_full_right + config.depth
        pred_rot_depth_left = pred_rot_full_left + config.depth

        pred_rot_depth_left[pred_rot_depth_left < 1.0] = 1.0
        pred_rot_depth_right[pred_rot_depth_right < 1.0] = 1.0

        pred_3d_rot_left = torch.cat(((rot_2d_left[:, 0:34].reshape(-1, 2, 17) * pred_rot_depth_left.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth_left),dim=1)
        pred_3d_rot_right = torch.cat(((rot_2d_right[:, 0:34].reshape(-1, 2, 17) * pred_rot_depth_right.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth_right),dim=1)

        pred_3d_rot_left = pred_3d_rot_left.reshape(-1, 3, 17) - pred_3d_rot_left.reshape(-1, 3, 17)[:, :, [0]]
        pred_3d_rot_right = pred_3d_rot_right.reshape(-1, 3, 17) - pred_3d_rot_right.reshape(-1, 3, 17)[:, :, [0]]

        self.losses.L3d = (rot_poses_right - pred_3d_rot_right.reshape(-1, 51)).norm(dim=1).mean()
        self.losses.L3d += (rot_poses_left - pred_3d_rot_left.reshape(-1, 51)).norm(dim=1).mean()

        re_rot_3d_left = (R.permute(0, 2, 1) @ pred_3d_rot_left).reshape(-1, 51)
        re_rot_3d_right = (R.permute(0, 2, 1) @ pred_3d_rot_right).reshape(-1, 51)
        pred_rot_global_pose_left = torch.cat((re_rot_3d_left[:, 0:34], re_rot_3d_left[:, 34:51] + config.depth), dim=1)
        pred_rot_global_pose_right = torch.cat((re_rot_3d_right[:, 0:34], re_rot_3d_right[:, 34:51] + config.depth), dim=1)
        re_rot_2d_left = perspective_projection(pred_rot_global_pose_left)
        re_rot_2d_right = perspective_projection(pred_rot_global_pose_right)

        self.losses.rep_rot = (re_rot_2d_left - inp_poses).abs().sum(dim=1).mean()
        self.losses.rep_rot += (re_rot_2d_right - inp_poses).abs().sum(dim=1).mean()

        #pairwise deformation loss
        num_pairs = int(np.floor(pred_3d_left.shape[0] / 2))
        pose_pairs_left = pred_3d_left[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_right = pred_3d_right[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_re_rot_left = re_rot_3d_left[0:(2*num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_re_rot_right = re_rot_3d_right[0:(2*num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)

        self.losses.re_rot_3d = ((pose_pairs_left[:, 0] - pose_pairs_left[:, 1]) - (pose_pairs_re_rot_left[:, 0] - pose_pairs_re_rot_left[:, 1])).norm(dim=1).mean()
        self.losses.re_rot_3d += ((pose_pairs_right[:, 0] - pose_pairs_right[:, 1]) - (pose_pairs_re_rot_right[:, 0] - pose_pairs_re_rot_right[:, 1])).norm(dim=1).mean()

        ## bone lengths prior
        bl = get_bone_lengths_all(pred_3d_left.reshape(-1, 51))
        rel_bl = bl / bl.mean(dim=1, keepdim=True)
        self.losses.bl_prior = (self.bone_relations_mean - rel_bl).square().sum(dim=1).mean()
        bl = get_bone_lengths_all(pred_3d_right.reshape(-1, 51))
        rel_bl = bl / bl.mean(dim=1, keepdim=True)
        self.losses.bl_prior += (self.bone_relations_mean - rel_bl).square().sum(dim=1).mean()

        # #get symmetry loss
        # self.losses.left_sym_loss = symmetry_loss(pred_3d_left)
        # self.losses.right_sym_loss = symmetry_loss(pred_3d_right)

        # #get hip loss
        # self.losses.hip_loss = mirror_loss(pred_3d_left, 0, 1, 4)
        # self.losses.hip_loss += mirror_loss(pred_3d_right, 0, 1, 4)

        self.losses.loss = config.weight_likeli * self.losses.likeli + \
                           config.weight_2d * self.losses.rep_rot + \
                           config.weight_3d * self.losses.L3d + \
                           config.weight_velocity * self.losses.re_rot_3d

        self.losses.loss = self.losses.loss + config.weight_bl * (self.losses.bl_prior)

        self.manual_backward(self.losses.loss)
        left_opt.step()
        right_opt.step()


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

        inp_test_poses = test_poses_2dgt_normalized.reshape(-1, 2, 17)

        inp_test_left, inp_test_right = split_data_left_right(inp_test_poses)

        left_pred_test, _ = self.left_lifter(inp_test_left)
        right_pred_test, _ = self.right_lifter(inp_test_right)

        pred_test_left = combine_left_right_pred_1d(left_pred_test, right_pred_test, choice='left').reshape(-1, 17)
        pred_test_right = combine_left_right_pred_1d(left_pred_test, right_pred_test, choice='right').reshape(-1, 17)

        pred_test_left[:, 0] = 0.0
        pred_test_right[:, 0] = 0.0

        pred_test_depth_left = pred_test_left + config.depth
        pred_test_depth_right = pred_test_right + config.depth

        pred_test_poses_left = torch.cat(
            ((inp_test_poses.reshape(-1, 2, 17) * pred_test_depth_left.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
             pred_test_depth_left), dim=1).detach().cpu().numpy()

        pred_test_poses_right = torch.cat(
            ((inp_test_poses.reshape(-1, 2, 17) * pred_test_depth_right.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
             pred_test_depth_right), dim=1).detach().cpu().numpy()

        # rotate to camera coordinate system
        test_poses_cam_frame_left = pred_test_poses_left.reshape(-1, 3, 17)
        test_poses_cam_frame_right = pred_test_poses_right.reshape(-1, 3, 17)

        self.losses.pa_left = 0
        self.losses.pa_right = 0

        for eval_cnt in range(int(test_3dgt_normalized.shape[0])):
            err_left = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      pred_test_poses_left[eval_cnt].reshape(-1, 51),
                                      reflection='best')

            err_right = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      pred_test_poses_right[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa_left += err_left
            self.losses.pa_right += err_right

        self.losses.pa_left /= test_3dgt_normalized.shape[0]
        self.losses.pa_right /= test_3dgt_normalized.shape[0]

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



def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024,  dims_out))

## load pretrained INN
# a simple chain of operations is collected by ReversibleSequential
left_inn2d = Ff.SequenceINN(22)
right_inn2d = Ff.SequenceINN(22)
full_inn2d = Ff.SequenceINN(34)
for k in range(8):
    left_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    right_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    full_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

left_inn2d.load_state_dict(torch.load('no_sched_norm_flow_left_side_weights_h36m_v3_sampling.pt'))
right_inn2d.load_state_dict(torch.load('no_sched_norm_flow_right_side_weights_h36m_v3_sampling.pt'))
full_inn2d.load_state_dict(torch.load('models/norm_flow_full_pose_with_sampling.pt'))
# freeze all weights in INN
for param in left_inn2d.parameters():
    param.requires_grad = False

for param in right_inn2d.parameters():
    param.requires_grad = False

for param in full_inn2d.parameters():
    param.requires_grad = False

datafile = '../EVAL_DATA/correct_interesting_frames_h36m.pkl'

train_data = H36M_Data(datafile, train=True, get_pca=True, normalize_func=normalize_head, get_2dgt=True,
                                 subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
test_data = H36M_Data(datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True,
                                subjects=['S9', 'S11'])

test_loader = data.DataLoader(test_data, batch_size=10000, num_workers=0)
train_loader = data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
# model
model = LitLifter(left_inn2d=left_inn2d, right_inn2d=right_inn2d, full_inn2d=full_inn2d)

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0, logger=False,
                     max_epochs=config.N_epochs, )

trainer.fit(model, train_loader, test_loader)

torch.save(model.left_lifter.state_dict(), 'models/left_side_lifter_final.pt')
torch.save(model.right_lifter.state_dict(), 'models/right_side_lifter_final.pt')