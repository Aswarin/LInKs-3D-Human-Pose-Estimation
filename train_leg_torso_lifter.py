import torch.nn
import torch.optim
from torch.utils import data
import pytorch_lightning as pl
from lift_and_fill_models.utils.models_def import Leg_Lifter, Torso_Lifter
from types import SimpleNamespace
from lift_and_fill_models.utils.rotation_conversions import euler_angles_to_matrix
from lift_and_fill_models.utils.metrics import Metrics
from lift_and_fill_models.utils.metrics_batch import Metrics as mb
from lift_and_fill_models.utils.helpers import *
from lift_and_fill_models.utils.h36m_dataset_class import H36M_Data, MPI_INF_3DHP_Dataset
import torch
import torch.distributions.uniform as uniform

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
wandb.run.name = "Leg and Torso Lifter " + wandb.run.name
project_folder = ''
data_folder = ''
config = wandb.config
config.learning_rate = 0.0002
config.BATCH_SIZE = 256
config.N_epochs = 100

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
    def __init__(self, leg_inn_2d, torso_inn_2d, full_inn_2d):
        super(LitLifter, self).__init__()

        self.leg_inn_2d = leg_inn_2d.to(self.device)
        self.torso_inn_2d = torso_inn_2d.to(self.device)
        self.full_inn_2d = full_inn_2d.to(self.device)

        for param in self.left_lifter.parameters():
            param.requires_grad = False

        for param in self.right_lifter.parameters():
            param.requires_grad = False


        self.legs_lifter = Leg_Lifter(use_batchnorm=False, num_joints=7, use_dropout=False, d_rate=0.25).cuda()
        self.torso_lifter = Torso_Lifter(use_batchnorm=False, num_joints=10, use_dropout=False, d_rate=0.25).cuda()

        # self.bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
        #                                          0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()

        # self.bone_relations_mean = torch.Tensor([0.5180581, 1.73711136, 1.72285805, 0.5180552, 1.73710543,
        #                                          1.72285651, 0.92087518, 0.98792375, 0.44812302, 0.44502545,
        #                                          0.57462, 1.08121276, 0.9651687, 0.57461556, 1.08122523,
        #                                          0.9651657]).cuda()  # human 3.6m relations mean

        #  self.bone_relations_mean = torch.Tensor([0.48123457, 1.83892552, 1.49699857, 0.48123457, 1.83579479,
        # 1.49699856, 0.90885878, 0.99415561, 0.34720909, 0.69462614,
        # 0.57956265, 1.21052741, 0.9251606 , 0.57302514, 1.21052743,
        # 0.92516058]).cuda() #all cameras MPI mean

        #  self.bone_relations_mean = torch.Tensor([0.48115763, 1.83961257, 1.49705786, 0.48115763, 1.83655297,
        # 1.49705784, 0.9086628 , 0.99419836, 0.34713946, 0.69448684,
        # 0.57953889, 1.21026625, 0.924973  , 0.57289866, 1.21026624,
        # 0.924973]) #vnect cameras MPI mean

        self.bone_relations_mean = torch.Tensor([0.48069107, 1.84637771, 1.49564841, 0.48069107, 1.84301997,
        1.4956484 , 0.90757932, 0.99706493, 0.34679742, 0.69380255,
        0.57843534, 1.20698327, 0.92306225, 0.5741528 , 1.20698326,
        0.92306223]).cuda() #vnect cameras interesting MPI mean

        self.automatic_optimization = False

        self.metrics = Metrics()

        self.losses = SimpleNamespace()
        self.losses_mean = SimpleNamespace()

    def configure_optimizers(self):

        leg_optimizer = torch.optim.Adam(self.legs_lifter.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        leg_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=leg_optimizer, gamma=0.95)
        torso_optimizer = torch.optim.Adam(self.torso_lifter.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        torso_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=torso_optimizer, gamma=0.95)

        return [leg_optimizer, torso_optimizer], [leg_scheduler, torso_scheduler]

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch[0].step()
        sch[1].step()

    def training_step(self, train_batch, batch_idx):

        opt = self.optimizers()
        leg_opt = opt[0]
        torso_opt = opt[1]
        leg_opt.zero_grad()
        torso_opt.zero_grad()

        inp_poses = train_batch['p2d_gt']

        with torch.no_grad():
            z_2d, _ = self.full_inn_2d(inp_poses)
            gaussian_noisy = add_noise(z_2d, noise_factor=0.2)
            drawn_samples, _ = self.full_inn_2d(gaussian_noisy.cuda(), rev=True)
            drawn_samples = drawn_samples.reshape(-1, 2, 17)
            drawn_samples[:, :, [0]] = 0.0
            drawn_samples = drawn_samples.reshape(-1, inp_poses.shape[1])
            inp_samples_2D = drawn_samples.data

            inp_poses = torch.concat((inp_poses, inp_samples_2D), dim=0)



        # split data up into legs and torso
        inp_legs = inp_poses.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)
        inp_torso = inp_poses.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)

        legs_pred, legs_angle = self.legs_lifter(inp_legs)
        torso_pred, torso_angle = self.torso_lifter(inp_torso)

        props = (legs_angle + torso_angle) / 2
        #props = legs_angle

        pred = torch.cat((legs_pred, torso_pred), dim=1)
        pred[:, 0] = 0.0

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

        depth = pred + config.depth
        depth[depth < 1.0] = 1.0

        pred_3d = torch.cat(
            ((inp_poses.reshape(-1, 2, 17) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth),
            dim=1).reshape(-1, 3, 17)

        pred_3d = pred_3d.reshape(-1, 3, 17) - pred_3d.reshape(-1, 3, 17)[:, :, [0]]


        rot_poses = (R.matmul(pred_3d)).reshape(-1, 51)

        ## lift from augmented camera and normalize
        global_pose = torch.cat((rot_poses[:, 0:34], rot_poses[:, 34:51] + config.depth), dim=1)
        rot_2d = perspective_projection(global_pose)
        norm_poses = rot_2d


        torso_norm_poses = norm_poses.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)
        leg_norm_poses = norm_poses.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)


        z, log_jac_det = self.leg_inn_2d(leg_norm_poses)
        leg_likelis_1 = 0.5 * torch.sum(z ** 2, 1) - log_jac_det


        self.losses.leg_likeli = leg_likelis_1.mean()

        z, log_jac_det = self.torso_inn_2d(torso_norm_poses)
        torso_likelis_1 = 0.5 * torch.sum(z ** 2, 1) - log_jac_det


        self.losses.torso_likeli = torso_likelis_1.mean()

        self.losses.likeli = self.losses.torso_likeli + self.losses.leg_likeli

        ## reprojection error
        inp_legs_rot = norm_poses.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)
        inp_torso_rot = norm_poses.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)

        legs_pred_rot, _ = self.legs_lifter(inp_legs_rot)
        torso_pred_rot, _ = self.torso_lifter(inp_torso_rot)

        pred_rot = torch.cat((legs_pred_rot, torso_pred_rot), dim=1)
        pred_rot[:, 0] = 0.0

        pred_rot_depth = pred_rot + config.depth
        pred_rot_depth[pred_rot_depth < 1.0] = 1.0


        pred_3d_rot = torch.cat(
            ((norm_poses[:, 0:34].reshape(-1, 2, 17) * pred_rot_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth), dim=1)

        pred_3d_rot = pred_3d_rot.reshape(-1, 3, 17) - pred_3d_rot.reshape(-1, 3, 17)[:, :, [0]]

        self.losses.L3d = (rot_poses - pred_3d_rot.reshape(-1, 51)).norm(dim=1).mean()

        re_rot_3d = (R.permute(0, 2, 1) @ pred_3d_rot).reshape(-1, 51)
        pred_rot_global_pose = torch.cat((re_rot_3d[:, 0:34], re_rot_3d[:, 34:51] + config.depth), dim=1)
        re_rot_2d = perspective_projection(pred_rot_global_pose)
        norm_re_rot_2d = re_rot_2d

        self.losses.rep_rot = (norm_re_rot_2d - inp_poses).abs().sum(dim=1).mean()

        # pairwise deformation loss
        num_pairs = int(np.floor(pred_3d.shape[0] / 2))
        pose_pairs = pred_3d[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
        pose_pairs_re_rot_3d = re_rot_3d[0:(2 * num_pairs)].reshape(-1, 2, 51)
        self.losses.re_rot_3d = ((pose_pairs[:, 0] - pose_pairs[:, 1]) - (
                    pose_pairs_re_rot_3d[:, 0] - pose_pairs_re_rot_3d[:, 1])).norm(dim=1).mean()

        ## bone lengths prior
        bl = get_bone_lengths_all(pred_3d.reshape(-1, 51))
        rel_bl = bl / bl.mean(dim=1, keepdim=True)
        self.losses.bl_prior = (self.bone_relations_mean - rel_bl).square().sum(dim=1).mean()


        #hip angle error loss
        # self.losses.hip_angle_error = (pred_3d.reshape(-1, 3, 17)[:, :, 1] - pred_3d.reshape(-1, 3, 17)[:, :, 4]).square().sum(dim=1).mean()


        self.losses.loss = config.weight_likeli*self.losses.likeli + \
                           config.weight_2d*self.losses.rep_rot + \
                           config.weight_3d * self.losses.L3d + \
                           config.weight_velocity*self.losses.re_rot_3d


        self.losses.loss = self.losses.loss + config.weight_bl*self.losses.bl_prior

        self.manual_backward(self.losses.loss)
        torso_opt.step()
        leg_opt.step()


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
        inp_legs = inp_test_poses[:, :, :7].reshape(-1, 14)
        inp_torso = inp_test_poses[:, :, 7:].reshape(-1, 20)

        legs_pred_test, _ = self.legs_lifter(inp_legs)
        torso_pred_test, _ = self.torso_lifter(inp_torso)

        pred_test = torch.cat((legs_pred_test, torso_pred_test), dim=1)
        pred_test[:, 0] = 0.0

        pred_test_depth = pred_test + config.depth

        pred_test_poses = torch.cat(
            ((inp_test_poses.reshape(-1, 2, 17) * pred_test_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
             pred_test_depth), dim=1).detach().cpu().numpy()

        # rotate to camera coordinate system
        test_poses_cam_frame = pred_test_poses.reshape(-1, 3, 17)

        self.losses.pa = 0

        err_list = list()
        for eval_cnt in range(int(test_3dgt_normalized.shape[0])):
            err = self.metrics.pmpjpe(test_3dgt_normalized[eval_cnt].reshape(-1, 51).cpu().numpy(),
                                      pred_test_poses[eval_cnt].reshape(-1, 51),
                                      reflection='best')
            self.losses.pa += err
            err_list.append(err)

        self.losses.pa /= test_3dgt_normalized.shape[0]

        self.losses.mpjpe_scaled = mb().mpjpe(test_3dgt_normalized,
                                              torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device),
                                              num_joints=17,
                                              root_joint=0).mean().cpu().numpy()

        self.losses.auc = mb().AUC(test_3dgt_normalized, torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device),
                                         num_joints=17,
                                         root_joint=0).mean().cpu().numpy()

        self.losses.pck = mb().PCK(test_3dgt_normalized, torch.tensor(test_poses_cam_frame, device=test_3dgt_normalized.device),
                                         num_joints=17,
                                         root_joint=0).mean().cpu().numpy()

        wandb.log({'epoch': self.current_epoch})

        for key, value in self.losses_mean.__dict__.items():
            wandb.log({key: np.mean(value)})

        self.losses_mean = SimpleNamespace()

        for key, value in self.losses.__dict__.items():
            self.log(key, value.item(), prog_bar=True)


## load pretrained INN
# a simple chain of operations is collected by ReversibleSequential
leg_inn_2d = Ff.SequenceINN(14)
torso_inn_2d = Ff.SequenceINN(20)
full_inn_2d = Ff.SequenceINN(34)

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024,  dims_out))

for k in range(8):
    leg_inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    torso_inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    full_inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

torso_inn_2d.load_state_dict(torch.load('models/best_lifting_models/no_sched_norm_flow_torso_weights_h36m_v5.pt'))
leg_inn_2d.load_state_dict(torch.load('models/best_lifting_models/no_sched_norm_flow_leg_weights_h36m_v5.pt'))
full_inn_2d.load_state_dict(torch.load('models/norm_flow_full_pose_with_sampling.pt'))
# freeze all weights in INN
for param in leg_inn_2d.parameters():
    param.requires_grad = False
for param in torso_inn_2d.parameters():
    param.requires_grad = False
for param in full_inn_2d.parameters():
    param.requires_grad = False


datafile = '../EVAL_DATA/correct_interesting_frames_h36m.pkl'
test_datafile = '../EVAL_DATA/correct_interesting_frames_h36m.pkl'

train_data = H36M_Data(datafile, train=True, get_pca=True, normalize_func=normalize_head, get_2dgt=True,
                                 subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
test_data = H36M_Data(test_datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True,
                                subjects=['S9', 'S11'])

test_loader = data.DataLoader(test_data, batch_size=10000, num_workers=0)
train_loader = data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

# model
model = LitLifter(leg_inn_2d=leg_inn_2d, torso_inn_2d=torso_inn_2d, full_inn_2d = full_inn_2d)

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0, logger=False,
                     max_epochs=config.N_epochs, )

trainer.fit(model, train_loader, test_loader)

torch.save(model.legs_lifter.state_dict(), 'models/leg_lifter.pt')
torch.save(model.torso_lifter.state_dict(), 'models/torso_lifter.pt')